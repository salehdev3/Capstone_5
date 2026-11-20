"""Run from a fresh Python process with the final-sprint source on PYTHONPATH."""
import shutil
import sys
import hashlib
from PIL import Image
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import PLAN, aggregate, context, csv_save, image_path, metrics, read, save, sha, temperature
from evaluate import benchmark, frame_score, mask_evaluate, reference, retained
from legacy_evaluation import event_scores
from runtime import Classifier, DEMO_IDS
from train import Crops, infer, samples
from candidate import FastPipeline


def main(root):
    root, old, rows, classes = context(root)
    baseline = old / "policy.json"
    assert sha(baseline) == "1ee0887f4e6ec960c421aff5d60af40d6c500698570bdeaa40b3fa06a1ae5328"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = old.parent / "fast_candidate" / stamp
    out.mkdir(parents=True)
    print("Saving to", out, flush=True)
    for name in ("crop_sam.py", "candidate.py", "run_candidate.py"):
        shutil.copy2(Path(__file__).with_name(name), out / name)
    cache = Path(tempfile.mkdtemp(prefix="camera_candidate_"))
    (cache / "images").mkdir()
    by_id = {r["id"]: r for r in rows}

    def stage_images(ids):
        from PIL import Image
        import hashlib
        ids = sorted(set(ids))
        print("Checking", len(ids), "source images", flush=True)
        for index, ident in enumerate(ids, 1):
            dest = image_path(cache, ident)
            if not dest.exists():
                shutil.copyfile(image_path(root, ident), dest)
                with Image.open(dest) as image:
                    assert hashlib.sha256(image.convert("RGB").tobytes()).hexdigest() == by_id[ident]["pixel_sha256"]
            if index % 100 == 0:
                print("Images", index, "/", len(ids), flush=True)

    base = read(baseline)
    assert sha(Path(base["species_classifier"]["path"])) == base["species_classifier"]["sha256"]
    assert read(old / "classifier/completed.json")["checkpoint_sha256"] == base["species_classifier"]["sha256"]
    val = np.load(old / "classifier/validation_logits.npz")
    z, y = val["logits"][:, :14], val["labels"]
    counts = np.asarray(read(old / "data_support.json")["training_counts"][:14])
    assert (counts > 0).all()
    logprior = np.log(counts / counts.sum())
    baseline_accuracy = metrics(z, y)["accuracy"]
    choices = []
    for strength in (0, .1, .25, .5):
        s = metrics(z - strength * logprior, y)
        if s["accuracy"] >= baseline_accuracy:
            choices.append((s["macro_f1"], strength))
    strength = max(choices)[1]
    offset = -strength * logprior
    print("Calibrating species scores, prior strength", strength, flush=True)
    cal = samples(rows, classes, "val_calibrate")
    stage_images(r["id"] for r in cal)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = Classifier(base["species_classifier"]["path"], device)
    assert list(net.classes) == list(classes)
    cz, cy = infer(net.net, DataLoader(Crops(cal, cache), batch_size=16), device)
    del net
    base.update(version="fast_candidate_v1", species_temperature=temperature(cz[:, :14]+offset, cy),
                species_logit_offset=offset.tolist(), baseline_sha256=sha(baseline),
                runtime_sha256={n: sha(out/n) for n in ("candidate.py", "crop_sam.py")})
    val_labels = reference(root, rows, "val_select")
    stage_images([p.stem for p in val_labels] + DEMO_IDS)
    candidates = []
    for mode, profile, size in (("frame", "int8", 448), ("crop", "float", 256), ("crop", "int8", 256), ("crop", "float", 192)):
        base["mask_mode"] = mode
        name = f"{mode}_{profile}_{size}"
        dest = out / f"validation_{name}"
        base["mask"]["sam"]["size"] = size
        base["encoder_precision"] = profile
        save(dest / "policy.json", base)
        try:
            p = FastPipeline(root, cache, dest / "policy.json")
        except (ImportError, AttributeError, NotImplementedError) as exc:
            if profile != "int8":
                raise
            print("Optional quantization unavailable", exc, flush=True)
            continue
        quality = mask_evaluate(p, val_labels, by_id, cache, dest / "masks")
        speed = benchmark(p, rows, cache, dest / "cpu")
        candidates.append(dict(name=name, size=size, iou=quality["mean_iou"], seconds=speed["p95"]))
        print("Candidate", candidates[-1], flush=True)
        del p
        if quality["mean_iou"] >= .7 and speed["p95"] <= 2:
            break
    save(out / "validation.json", dict(candidates=candidates, species=metrics(z+offset, y, base["species_temperature"])))
    feasible = [c for c in candidates if c["iou"] >= .7 and c["seconds"] <= 2]
    if not feasible:
        save(out / "status.json", dict(status="validation_gate_failed", candidates=candidates))
        print("Validation stopped before historical evaluation.", candidates, out, flush=True)
        return
    selected = max(feasible, key=lambda c: c["iou"])
    policy = out / "policy.json"
    shutil.copyfile(out / f"validation_{selected['name']}" / "policy.json", policy)
    p = FastPipeline(root, cache, policy)
    save(out / "freeze.json", dict(policy_sha256=sha(policy), independent_test=False, selected=selected))
    speed = benchmark(p, rows, cache, out / "cpu")
    test = retained(rows, "test")
    crops = samples(rows, classes, "test")
    labels = reference(root, rows, "test")
    stage_images([r["id"] for r in test+crops] + [p.stem for p in labels])
    predictions = {}
    scores = []
    for i, r in enumerate(test, 1):
        f = p.process_frame(image_path(cache, r["id"]), r, with_masks=False)
        predictions[r["id"]] = f
        scores.append(frame_score(r, [o["box"] for o in f["objects"]]))
        if i % 50 == 0:
            print("Historical detection", i, "/", len(test), flush=True)
    d = aggregate(scores)
    events, excluded = event_scores(rows, predictions)
    csv_save(out / "event_counts.csv", events)
    csv_save(out / "detection_per_image.csv", scores)
    m = mask_evaluate(p, labels, by_id, cache, out / "historical_masks")
    logits, labels_y = infer(p.species.net, DataLoader(Crops(crops, cache), batch_size=16), "cpu")
    s = metrics(logits[:, :14]+offset, labels_y, base["species_temperature"])
    save(out / "species_metrics.json", s)
    values = dict(animal_recall_min=d["animal_recall"], empty_removal_min=d["empty_removal"],
                  species_macro_f1_min=s["macro_f1"], species_ece_max=s["ece"], mask_iou_min=m["mean_iou"],
                  event_count_mae_max=float(np.mean([e["absolute_error"] for e in events])), p95_cpu_seconds_max=speed["p95"])
    assert values.keys() == PLAN["original_targets"].keys()
    result = [dict(metric=k, value=v, target=t, met=bool(v>=t if k.endswith("_min") else v<=t))
              for k,v in values.items() for t in [PLAN["original_targets"][k]]]
    csv_save(out / "acceptance.csv", result)
    decision = dict(decision="GO" if all(r["met"] for r in result) else "NO-GO",
                    independent_test=False, policy_sha256=sha(policy), acceptance=result)
    save(out / "decision.json", decision)
    print(decision, out, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
