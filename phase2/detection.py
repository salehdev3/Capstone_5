import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


SPLITS = {"train": "train", "val_select": "val", "val_calibrate": "calibration", "test": "test"}
HUMAN_SOURCES = {"human_CCT20", "team_human_verified"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_recorded(path, workspace):
    path = Path(path)
    if path.is_absolute() and path.is_file():
        return path
    candidate = workspace / path
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Required Phase 1 file is unavailable: {path}")


def load_release(release_path):
    release_path = Path(release_path).resolve()
    workspace = release_path.parent.parent
    release = json.loads(release_path.read_text())
    if release.get("schema") != 2 or not release.get("audit_complete"):
        raise ValueError("A completed Phase 1 dataset release is required")
    manifest_name = next((name for name in release["files"] if Path(name).name == "manifest.json"), None)
    if manifest_name is None:
        raise ValueError("The release does not record manifest.json")
    manifest = resolve_recorded(manifest_name, workspace)
    if digest(manifest) != release["files"][manifest_name]:
        raise ValueError("The Phase 1 manifest differs from the frozen release")
    rows = json.loads(manifest.read_text())
    return rows, workspace, digest(manifest)


def validate_rows(rows, workspace):
    seen = defaultdict(dict)
    counts = Counter()
    eligible = []
    excluded = Counter()
    ids = set()
    for row in rows:
        ident = str(row["id"])
        split = row["split"]
        if split not in SPLITS or ident in ids:
            raise ValueError(f"Unknown split or duplicate ID: {ident}")
        ids.add(ident)
        for field in ("camera", "event", "pixel_sha256"):
            value = row.get(field)
            if value is None:
                continue
            if value in seen[field] and seen[field][value] != split:
                raise ValueError(f"Cross-split {field} leakage: {value}")
            seen[field][value] = split
        if row.get("incomplete_boxes"):
            excluded["incomplete_boxes"] += 1
            continue
        boxes = row.get("boxes", [])
        if row.get("animal") and not boxes:
            excluded["animal_without_box"] += 1
            continue
        if not row.get("animal") and boxes:
            raise ValueError(f"Empty frame contains boxes: {ident}")
        for box in boxes:
            if box.get("annotation_source") not in HUMAN_SOURCES:
                raise ValueError(f"Non-human annotation in {ident}")
            x1, y1, x2, y2 = box["xyxy"]
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError(f"Invalid normalized box in {ident}")
        image = resolve_recorded(row["path"], workspace)
        with Image.open(image) as source:
            source.verify()
        counts[split] += 1
        eligible.append((row, image))
    if not counts["train"] or not counts["val_select"] or not counts["test"]:
        raise ValueError("Train, selection validation, and test must contain eligible frames")
    return eligible, dict(counts), dict(excluded)


def yolo_line(box):
    x1, y1, x2, y2 = box["xyxy"]
    return f"0 {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} {x2-x1:.8f} {y2-y1:.8f}"


def prepare(release_path, output, sample_count=24, seed=42):
    rows, workspace, manifest_hash = load_release(release_path)
    eligible, counts, excluded = validate_rows(rows, workspace)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    for row, image in eligible:
        split = SPLITS[row["split"]]
        name = f"{row['id']}.jpg"
        target = output / "images" / split / name
        label = output / "labels" / split / f"{row['id']}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(image)
        label.write_text("\n".join(yolo_line(box) for box in row["boxes"]) + ("\n" if row["boxes"] else ""))
    rng = random.Random(seed)
    positive = [(row, path) for row, path in eligible if row["boxes"]]
    rng.shuffle(positive)
    sampled = positive[:min(sample_count, len(positive))]
    thumbs = []
    for row, image in sampled:
        with Image.open(image) as source:
            frame = ImageOps.exif_transpose(source).convert("RGB")
        draw = ImageDraw.Draw(frame)
        for box in row["boxes"]:
            x1, y1, x2, y2 = box["xyxy"]
            draw.rectangle((x1 * frame.width, y1 * frame.height, x2 * frame.width, y2 * frame.height), outline="lime", width=max(2, frame.width // 350))
        frame.thumbnail((400, 280))
        tile = Image.new("RGB", (420, 315), "white")
        tile.paste(frame, ((420 - frame.width) // 2, 25))
        ImageDraw.Draw(tile).text((10, 5), f"{row['id']} | {row['split']}", fill="black")
        thumbs.append(tile)
        audit_rows.append({"id": row["id"], "split": row["split"], "reviewer": "", "boxes_correct": "", "notes": ""})
    if thumbs:
        columns = 4
        sheet = Image.new("RGB", (columns * 420, ((len(thumbs) + columns - 1) // columns) * 315), "white")
        for index, tile in enumerate(thumbs):
            sheet.paste(tile, ((index % columns) * 420, (index // columns) * 315))
        sheet.save(output / "box_review.jpg", quality=90)
    with (output / "box_review.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "split", "reviewer", "boxes_correct", "notes"])
        writer.writeheader()
        writer.writerows(audit_rows)
    (output / "data.yaml").write_text(
        f"path: {output}\ntrain: images/train\nval: images/val\ncalibration: images/calibration\ntest: images/test\n"
        "names:\n  0: animal\n"
    )
    report = {"manifest_sha256": manifest_hash, "task": "single-class animal detection", "split_counts": counts,
              "excluded": excluded, "boxes": sum(len(row["boxes"]) for row, _ in eligible),
              "audit_samples": len(audit_rows), "validation_for_selection": "val_select",
              "validation_for_threshold": "val_calibrate", "locked_test": "test"}
    (output / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def verify_review(dataset):
    dataset = Path(dataset)
    with (dataset / "box_review.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(not row["reviewer"].strip() or row["boxes_correct"] not in {"yes", "no"} for row in rows):
        raise ValueError("Complete every box_review.csv decision and reviewer name")
    if any(row["boxes_correct"] == "no" for row in rows):
        raise ValueError("Resolve incorrect boxes in a new dataset release before training")
    return len(rows)


def train(dataset, epochs=30, image_size=640, batch=16, model="yolo11n.pt", device="0"):
    verify_review(dataset)
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Install phase2/requirements.txt in a GPU environment") from error
    dataset = Path(dataset).resolve()
    network = YOLO(model)
    return network.train(data=str(dataset / "data.yaml"), epochs=epochs, imgsz=image_size, batch=batch,
                         device=device, seed=42, project=str(dataset / "runs"), name="animal_detector",
                         exist_ok=False)


def main():
    parser = argparse.ArgumentParser(description="Prepare and train the Phase 2 animal detector")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--release", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--audit-samples", type=int, default=24)
    fit = sub.add_parser("train")
    fit.add_argument("--dataset", type=Path, required=True)
    fit.add_argument("--epochs", type=int, default=30)
    fit.add_argument("--image-size", type=int, default=640)
    fit.add_argument("--batch", type=int, default=16)
    fit.add_argument("--model", default="yolo11n.pt")
    fit.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.release, args.output, args.audit_samples), indent=2))
    else:
        train(args.dataset, args.epochs, args.image_size, args.batch, args.model, args.device)


if __name__ == "__main__":
    main()
