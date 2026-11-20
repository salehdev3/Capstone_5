"""Fresh-process real HTTP execution, used for the recorded fallback and clean replay."""

from common import *
from runtime import Pipeline, DEMO_IDS
import requests, time, cv2, numpy as np, os, sys, platform


def overlay(path, frame, target):
    from PIL import ImageDraw, ImageFont

    im = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(im)
    for o in frame["objects"]:
        x1, y1, x2, y2 = o["box"]
        d.rectangle(
            (x1 * im.width, y1 * im.height, x2 * im.width, y2 * im.height),
            outline="orange" if o["review"] else "red",
            width=3,
        )
        d.text(
            (max(0, x1 * im.width), max(0, y1 * im.height - 12)),
            f"{o['species']} {o['confidence']:.2f}",
            fill="yellow",
        )
        polygon = [
            (float(x * im.width), float(y * im.height))
            for x, y in o.get("polygon_normalized", [])
        ]
        if len(polygon) >= 3:
            d.line(polygon + [polygon[0]], fill="lime", width=3)
    im.save(target)


def run(root, cache):
    import torch

    root, out, rows, _ = context(root)
    dest = out / "http"
    dest.mkdir(exist_ok=True)
    p = Pipeline(
        root, cache, out / "policy.json", "cuda" if torch.cuda.is_available() else "cpu"
    )
    service = load(Path(__file__).parent / "legacy_service.py", "http_contracts")
    server, thread, base, instance = service.start_service(p)
    chosen = [{r["id"]: r for r in rows}[k] for k in DEMO_IDS]
    paths = [image_path(cache, r["id"]) for r in chosen]
    metadata = [
        dict(id=r["id"], event=r["event"], timestamp=r.get("timestamp")) for r in chosen
    ]
    try:
        began = time.perf_counter()
        response = requests.post(
            base + "/batch",
            files=[
                ("images", (path.name, path.read_bytes(), "image/jpeg"))
                for path in paths
            ],
            data={"metadata": json.dumps(metadata)},
            timeout=(10, 1200),
        )
        response.raise_for_status()
        result = response.json()
        assert [r["id"] for r in result["frames"]] == DEMO_IDS
        invalid = requests.post(
            base + "/batch",
            files=[("images", (paths[0].name, paths[0].read_bytes(), "image/jpeg"))],
            data={"metadata": "[]"},
            timeout=10,
        )
        assert invalid.status_code == 400
        save(
            dest / "service_test.json",
            dict(
                status=response.status_code,
                invalid_request_status=invalid.status_code,
                result=result,
                instance=instance,
                policy_sha256=p.policy_sha256,
                elapsed_seconds=time.perf_counter() - began,
            ),
        )
        coverage = dict(
            frames_with_detections=sum(bool(f["objects"]) for f in result["frames"]),
            classified_objects=sum(
                "species" in o for f in result["frames"] for o in f["objects"]
            ),
            matched_masks=sum(
                f["segmentation"]["matched_masks"] for f in result["frames"]
            ),
        )
        coverage["all_stages_exercised"] = (
            coverage["classified_objects"] > 0 and coverage["matched_masks"] > 0
        )
        save(
            dest / "demo_record.json",
            dict(
                kind="Executed fresh-process local HTTP request",
                images=5,
                split="val_select",
                coverage=coverage,
                policy_sha256=p.policy_sha256,
                selection="Same fixed five branch-coverage cases, including historical misses; not an accuracy sample",
            ),
        )
        save(
            dest / "clean_replay.json",
            dict(
                new_process=True,
                pid=os.getpid(),
                python=sys.version,
                policy_sha256=p.policy_sha256,
                dependencies_imported=True,
                http_status=response.status_code,
                scope="Fresh Python interpreter after dependency installation; not an independent environment-installation audit",
            ),
        )
        for path, f in zip(paths, result["frames"]):
            overlay(path, f, dest / (f["id"] + ".png"))
        print(
            "HTTP request:",
            response.status_code,
            "model-stage coverage:",
            coverage,
            flush=True,
        )
    finally:
        server.should_exit = True
        thread.join(timeout=10)
