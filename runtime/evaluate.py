"""Validation selection first; the historical development test is evaluated after freeze."""

from common import *
from runtime import Pipeline, Classifier, DEMO_IDS, sources
from train import Crops, samples, infer
import torch, cv2, time, gc, copy, resource
from torch.utils.data import DataLoader
from scipy.special import softmax


def retained(rows, split):
    return [
        r
        for r in rows
        if r["split"] == split
        and not (r["animal"] and (r.get("incomplete_boxes", True) or not r["boxes"]))
    ]


def entry(path, **kw):
    return dict(path=str(path), sha256=sha(path), **kw)


def frame_score(r, boxes):
    tp = len(pairs([b["xyxy"] for b in r["boxes"]], boxes))
    return dict(
        id=r["id"],
        animal=bool(r["animal"]),
        count=len(boxes),
        tp=tp,
        fp=len(boxes) - tp,
        fn=len(r["boxes"]) - tp,
    )


def reference(root, rows, split):
    labels = sorted((Path(root) / "segmentation/labels" / split).glob("*.txt"))
    assert len(labels) == 40
    old = read(Path(root) / "segmentation/reference_release.json")["mask_hashes"]
    byid = {r["id"]: r for r in rows}
    for p in labels:
        assert (
            sha(p) == old[p.stem] and byid[p.stem]["split"] == split
        ), "Reference release changed"
    return labels


def species_selection(root, cache):
    root, run, rows, classes = context(root)
    candidate = run / "classifier"
    valid = samples(rows, classes, "val_select")
    loader = DataLoader(Crops(valid, cache), batch_size=48, num_workers=2)
    records = []
    paths = [
        (
            candidate / "best.pt",
            "convnext_tiny_bg15",
            read(candidate / "policy.json")["species_temperature"],
        ),
        (root / "models/species_recovery_v1/best.pt", "resnet50", None),
        (root / "phase2_results/optimization_v3/species/best.pt", "resnet50", None),
    ]
    for path, architecture, t in paths:
        if not path.exists():
            continue
        if t is None:
            t = read(path.with_name("policy.json"))["temperature"]
        c = Classifier(path, "cuda", architecture)
        z, y = infer(c.net, loader, "cuda")
        score = metrics(z[:, :14], y, t)
        records.append(
            dict(
                model=entry(path, architecture=architecture),
                temperature=t,
                metrics=score,
            )
        )
        del c
        gc.collect()
        torch.cuda.empty_cache()
    best = max(records, key=lambda r: r["metrics"]["macro_f1"])
    save(
        run / "species_selection.json",
        dict(
            selected=best,
            candidates=records,
            split="val_select",
            metric="macro F1 over all 14 fixed species, including zero-support classes",
        ),
    )
    print(
        "Selected species model:",
        best["model"]["architecture"],
        best["metrics"]["accuracy"],
        best["metrics"]["macro_f1"],
        flush=True,
    )


def detector_selection(root, cache):
    from ultralytics import YOLO

    root, run, rows, classes = context(root)
    models = read(run / "sources.json")["models"]
    animal = Classifier(run / "classifier/best.pt", "cuda")
    temp = read(run / "classifier/policy.json")["animal_temperature"]
    chosen = retained(rows, "val_select")
    sweep = []
    confidences = [
        0.005,
        0.01,
        0.015,
        0.02,
        0.025,
        0.03,
        0.035,
        0.04,
        0.045,
        0.05,
        0.075,
        0.1,
        0.15,
        0.25,
        0.4,
        0.5,
    ]
    gates = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98]
    for name in [k for k in ("d_old", "d_new") if k in models]:
        spec = models[name]
        dest = run / "detector_cache" / name
        dest.mkdir(parents=True, exist_ok=True)
        identity = dict(
            detector_sha256=spec["sha256"],
            classifier_sha256=sha(run / "classifier/best.pt"),
            temperature=temp,
            manifest_sha256=MANIFEST_SHA,
            raw_conf=PLAN["raw_conf"],
            nms=PLAN["nms"],
            max_det=PLAN["max_boxes"],
            rect=False,
        )
        if (dest / "identity.json").exists():
            assert read(dest / "identity.json") == identity
        else:
            save(dest / "identity.json", identity)
        detector = YOLO(spec["cached"])
        predictions = []
        for i, r in enumerate(chosen, 1):
            p = dest / (r["id"] + ".json")
            if p.exists():
                record = read(p)
            else:
                im = cv2.imread(str(image_path(cache, r["id"])))
                d = detector.predict(
                    im,
                    conf=PLAN["raw_conf"],
                    iou=PLAN["nms"],
                    max_det=PLAN["max_boxes"],
                    imgsz=spec["size"],
                    rect=False,
                    device="cuda",
                    verbose=False,
                )[0]
                boxes = d.boxes.xyxyn.cpu().numpy().tolist()
                z = animal.logits(im, boxes)
                a = (1 - softmax(z / temp, axis=1)[:, 14]).tolist() if len(z) else []
                record = dict(
                    id=r["id"],
                    boxes=boxes,
                    scores=d.boxes.conf.cpu().numpy().tolist(),
                    animal_probability=a,
                )
                save(p, record)
            predictions.append(record)
            if i % 100 == 0:
                print(name, "validation", i, "/", len(chosen), flush=True)
        for conf in confidences:
            for gate in gates:
                scores = []
                for r, p in zip(chosen, predictions):
                    boxes = [
                        b
                        for b, s, a in zip(
                            p["boxes"], p["scores"], p["animal_probability"]
                        )
                        if s >= conf and a >= gate
                    ]
                    scores.append(frame_score(r, boxes))
                sweep.append(
                    dict(detector=name, conf=conf, animal_min=gate, **aggregate(scores))
                )
        del detector
        gc.collect()
        torch.cuda.empty_cache()

    def feasible(m, margin):
        return (
            m["animal_recall"] >= margin
            and m["empty_removal"] >= PLAN["original_targets"]["empty_removal_min"]
        )

    candidates = [m for m in sweep if feasible(m, PLAN["selection_recall_margin"])]
    rule = "validation recall >= .995 and empty removal >= .20"
    if not candidates:
        candidates = [m for m in sweep if feasible(m, 0.99)]
        rule = "validation recall >= .99 and empty removal >= .20"
    if not candidates:
        candidates = sweep
        rule = "No feasible validation point; rank recall, then empty removal. Targets are unchanged."
        key = lambda m: (m["animal_recall"], m["empty_removal"], m["f1"])
    else:
        key = lambda m: (m["empty_removal"], m["f1"], m["animal_recall"])
    best = max(candidates, key=key)
    selected = dict(
        **models[best["detector"]], conf=best["conf"], animal_min=best["animal_min"]
    )
    selected.pop("cached")
    csv_save(run / "threshold_sweep.csv", sweep)
    save(
        run / "detector_selection.json",
        dict(selected=selected, validation=best, selection_rule=rule),
    )
    print("Operating point:", best, flush=True)


def seed_policy(root):
    root, run, _, _ = context(root)
    s = read(run / "species_selection.json")["selected"]
    a = read(run / "classifier/policy.json")
    sources = read(run / "sources.json")
    return dict(
        version=PLAN["version"],
        manifest_sha256=MANIFEST_SHA,
        detector=read(run / "detector_selection.json")["selected"],
        animal_classifier=entry(
            run / "classifier/best.pt", architecture="convnext_tiny_bg15"
        ),
        animal_temperature=a["animal_temperature"],
        species_classifier=s["model"],
        species_temperature=s["temperature"],
        mask_helper_sha256=sources["mask_helper_sha256"],
        selection_split="val_select",
        calibration_split="val_calibrate",
        historical_test_is_independent=False,
    )


def mask_evaluate(pipeline, labels, byid, cache, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    predictions = []
    for i, label in enumerate(labels, 1):
        r = byid[label.stem]
        file = dest / (r["id"] + ".json")
        if file.exists():
            f = read(file)
            assert f["policy_sha256"] == pipeline.policy_sha256
            frame = f["frame"]
        else:
            frame = pipeline.process_frame(image_path(cache, r["id"]), r)
            save(file, dict(policy_sha256=pipeline.policy_sha256, frame=frame))
        polygons = [
            np.array(list(map(float, line.split()[1:]))).reshape(-1, 2)
            for line in label.read_text().splitlines()
            if line.strip()
        ]
        gt = raster(polygons, r["width"], r["height"])
        pred = raster(
            [
                o["polygon_normalized"]
                for o in frame["objects"]
                if "polygon_normalized" in o
            ],
            r["width"],
            r["height"],
        )
        inter = (gt & pred).sum()
        union = (gt | pred).sum()
        den = gt.sum() + pred.sum()
        records.append(
            dict(
                id=r["id"],
                iou=float(inter / union) if union else 1.0,
                dice=float(2 * inter / den) if den else 1.0,
                area_mae_fraction=abs(float(gt.mean()) - float(pred.mean())),
                label_sha256=sha(label),
            )
        )
        predictions.append(frame)
        if i % 10 == 0:
            print("Masks", dest.name, i, "/", len(labels), flush=True)
    result = dict(
        n=len(records),
        mean_iou=float(np.mean([r["iou"] for r in records])),
        mean_dice=float(np.mean([r["dice"] for r in records])),
        area_mae_fraction=float(np.mean([r["area_mae_fraction"] for r in records])),
        policy_sha256=pipeline.policy_sha256,
        split=byid[labels[0].stem]["split"],
        independent_test=False,
        definition="Per-image animal foreground union; arithmetic mean over all 40 original references, including misses; no GT prompts",
    )
    csv_save(dest / "per_image.csv", records)
    save(dest / "predictions.json", predictions)
    save(dest / "metrics.json", result)
    return result


def benchmark(pipeline, rows, cache, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    lookup = {r["id"]: r for r in rows}
    chosen = [lookup[k] for k in DEMO_IDS]
    lat = []
    for r in chosen[:3]:
        pipeline.process_frame(image_path(cache, r["id"]), r)
    for i in range(20):
        r = chosen[i % 5]
        f = pipeline.process_frame(image_path(cache, r["id"]), r)
        lat.append(dict(iteration=i + 1, id=r["id"], **f["timing"]))
        print(
            "Full pipeline",
            pipeline.device,
            i + 1,
            "/20",
            round(f["timing"]["total"], 3),
            "s",
            flush=True,
        )
    seconds = [r["total"] for r in lat]
    m = dict(
        p50=float(np.quantile(seconds, 0.5)),
        p95=float(np.quantile(seconds, 0.95)),
        p99=float(np.quantile(seconds, 0.99)),
        throughput=1 / float(np.mean(seconds)),
        repeats=20,
        warmup=3,
        batch=1,
        device=pipeline.device,
        scope="Fixed five validation functional cases repeated four times; full image decode/detection/animal filter/species/M5/refiner; excludes HTTP; GPU synchronized",
        peak_process_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        threads=torch.get_num_threads(),
        torch_version=str(torch.__version__),
        hardware=__import__("platform").platform(),
        gpu_name=torch.cuda.get_device_name(0) if pipeline.device != "cpu" else None,
        memory_scope="Cumulative process peak RSS",
        policy_sha256=pipeline.policy_sha256,
    )
    csv_save(dest / "latencies.csv", lat)
    save(dest / "benchmark.json", m)
    return m


def mask_selection(root, cache):
    root, run, rows, classes = context(root)
    labels = reference(root, rows, "val_select")
    byid = {r["id"]: r for r in rows}
    specs = read(run / "sources.json")["models"]
    base = seed_policy(root)
    records = []
    # A bounded quality/speed comparison of already trained guides and two pretrained refiners.
    for guide in [k for k in ["m_old", "m_v3", "m_v4"] if k in specs]:
        for profile, kind, mask_size, guide_conf in [
            ("direct010", "direct", 0, 0.10),
            ("direct025", "direct", 0, 0.25),
            ("direct035", "direct", 0, 0.35),
            ("direct050", "direct", 0, 0.50),
            ("sam_b", "sam_b", 640, 0.05),
            ("sam_t640", "sam_t", 640, 0.05),
            ("sam_t1024", "sam_t", 1024, 0.05),
        ]:
            name = guide + "_" + profile
            dest = run / "mask_selection" / name
            dest.mkdir(parents=True, exist_ok=True)
            p = copy.deepcopy(base)
            g = dict(specs[guide])
            g.pop("cached")
            m = dict(
                kind=kind,
                guide=g,
                guide_conf=guide_conf,
                primary_conf=0.4,
                quality=0.75 if kind == "sam_b" else 0.0,
            )
            if kind != "direct":
                m["sam"] = {k: v for k, v in specs[kind].items() if k != "cached"}
                m["sam"]["size"] = mask_size
            p["mask"] = m
            save(dest / "policy.json", p)
            pipeline = Pipeline(root, cache, dest / "policy.json", "cuda")
            score = mask_evaluate(pipeline, labels, byid, cache, dest)
            records.append(
                dict(
                    name=name,
                    kind=kind,
                    profile=profile if kind != "direct" else "direct",
                    policy=str(dest / "policy.json"),
                    **score
                )
            )
            del pipeline
            gc.collect()
            torch.cuda.empty_cache()
    finalists = [
        max(
            [r for r in records if r["profile"] == profile], key=lambda r: r["mean_iou"]
        )
        for profile in ["direct", "sam_b", "sam_t640", "sam_t1024"]
    ]
    for f in finalists:
        dest = Path(f["policy"]).parent / "cpu"
        receipt = dest / "benchmark.json"
        if receipt.exists():
            cpu = read(receipt)
            assert cpu["policy_sha256"] == sha(f["policy"])
        else:
            pipeline = Pipeline(root, cache, f["policy"], "cpu")
            cpu = benchmark(pipeline, rows, cache, dest)
            del pipeline
            gc.collect()
        f["cpu_p95"] = cpu["p95"]
    feasible = [f for f in finalists if f["mean_iou"] >= 0.8 and f["cpu_p95"] <= 2.0]
    if not feasible:
        feasible = [
            f for f in finalists if f["mean_iou"] >= 0.7 and f["cpu_p95"] <= 2.0
        ]
    selected = max(feasible or finalists, key=lambda f: f["mean_iou"])
    policy = read(selected["policy"])
    save(run / "policy.json", policy)
    csv_save(
        run / "mask_candidates.csv",
        [
            {k: r[k] for k in ("name", "kind", "n", "mean_iou", "mean_dice")}
            for r in records
        ],
    )
    save(
        run / "mask_selection.json",
        dict(
            selected=selected,
            candidates=records,
            finalists=finalists,
            selection="Prefer CPU p95 <=2 s with validation IoU >=.80, then >=.70; otherwise retain highest measured IoU. Fixed original targets remain unchanged.",
        ),
    )
    save(
        run / "freeze.json",
        dict(
            policy_sha256=sha(run / "policy.json"),
            frozen_at=datetime.now(timezone.utc).isoformat(),
            selection_split="val_select",
            test_used_for_selection=False,
            development_test_previously_exposed=True,
        ),
    )
    print("Frozen:", selected, flush=True)


def historical_test(root, cache):
    root, run, rows, classes = context(root)
    policy = run / "policy.json"
    assert sha(policy) == read(run / "freeze.json")["policy_sha256"]
    dest = run / "historical_test"
    dest.mkdir(exist_ok=True)
    p = Pipeline(root, cache, policy, "cuda")
    chosen = retained(rows, "test")
    records = []
    predictions = {}
    for i, r in enumerate(chosen, 1):
        target = dest / "detections" / (r["id"] + ".json")
        if target.exists():
            f = read(target)
            assert f["policy_sha256"] == p.policy_sha256
            frame = f["frame"]
        else:
            frame = p.process_frame(image_path(cache, r["id"]), r, with_masks=False)
            save(target, dict(frame=frame, policy_sha256=p.policy_sha256))
        records.append(frame_score(r, [o["box"] for o in frame["objects"]]))
        predictions[r["id"]] = frame
        if i % 50 == 0:
            print("Historical detection/count", i, "/", len(chosen), flush=True)
    # Slice analysis uses the same frozen predictions; no new threshold selection.
    slices = []
    for field in ("source", "camera", "grayscale_proxy"):
        values = sorted({str(r.get(field, "")) for r in chosen})
        for value in values:
            subset = [
                v for r, v in zip(chosen, records) if str(r.get(field, "")) == value
            ]
            slices.append(dict(field=field, value=value, **aggregate(subset)))
    csv_save(dest / "detection_slices.csv", slices)
    bins = {name: dict(instances=0, matched=0) for name in ["Small", "Medium", "Large"]}
    for r in chosen:
        boxes = [b["xyxy"] for b in r["boxes"]]
        matched = {
            i
            for i, j in pairs(
                boxes, [o["box"] for o in predictions[r["id"]]["objects"]]
            )
        }
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            area = (x2 - x1) * (y2 - y1) * 640**2
            name = "Small" if area < 32**2 else ("Medium" if area < 96**2 else "Large")
            bins[name]["instances"] += 1
            bins[name]["matched"] += int(i in matched)
    csv_save(
        dest / "detection_size_recall.csv",
        [
            dict(
                size=k,
                **v,
                recall=v["matched"] / v["instances"] if v["instances"] else None
            )
            for k, v in bins.items()
        ],
    )
    save(
        dest / "size_protocol.json",
        dict(
            definition="Normalized reference area x640^2; Small<32^2, Medium<96^2, Large>=96^2. Fixed baseline proxy, not exact COCO post-letterbox areas.",
            policy_sha256=p.policy_sha256,
        ),
    )
    save(
        dest / "detection_metrics.json",
        dict(
            **aggregate(records),
            policy_sha256=p.policy_sha256,
            split="test",
            independent_test=False,
            scope="Classifier-filtered retained complete-box frames; IoU>=.5 Hungarian matching"
        ),
    )
    csv_save(dest / "detection_per_image.csv", records)
    save(dest / "detection_predictions.json", list(predictions.values()))
    evaluation = load(
        Path(__file__).parent / "legacy_evaluation.py", "event_evaluation"
    )
    scores, excluded = evaluation.event_scores(rows, predictions)
    csv_save(dest / "event_counts.csv", scores)
    save(
        dest / "event_metrics.json",
        dict(
            mae=float(np.mean([r["absolute_error"] for r in scores])),
            events=len(scores),
            excluded_events=len(excluded),
            scope="Maximum simultaneous retained-frame count; complete-box events, not unique animals",
            policy_sha256=p.policy_sha256,
            independent_test=False,
        ),
    )
    save(dest / "excluded_events.json", excluded)
    loader = DataLoader(
        Crops(samples(rows, classes, "test"), cache), batch_size=48, num_workers=2
    )
    z, y = infer(p.species.net, loader, "cuda")
    m = metrics(z[:, :14], y, p.config["species_temperature"])
    save(
        dest / "species_metrics.json",
        dict(
            **m,
            classes=classes,
            checkpoint_sha256=p.config["species_classifier"]["sha256"],
            policy_sha256=p.policy_sha256,
            split="test",
            scope="Known-class human-box crops, not end-to-end species accuracy",
            independent_test=False
        ),
    )
    np.savez_compressed(dest / "species_logits.npz", logits=z, labels=y)
    labels = reference(root, rows, "test")
    mask_evaluate(p, labels, {r["id"]: r for r in rows}, cache, dest / "segmentation")
    benchmark(p, rows, cache, run / "gpu_runtime")
    del p
    gc.collect()
    torch.cuda.empty_cache()
    # AP from the native detector at a low confidence floor is a separate metric.
    from ultralytics import YOLO

    d = read(policy)["detector"]
    detector = YOLO(d["path"])
    report = detector.val(
        data=str(Path(cache) / "detection/data.yaml"),
        split="test",
        conf=0.001,
        iou=0.7,
        imgsz=d["size"],
        rect=False,
        batch=4,
        device=0,
        plots=True,
        save_json=True,
        project=str(dest),
        name="native_detector_ap",
        exist_ok=True,
        verbose=False,
    )
    save(
        dest / "detector_training_metadata.json",
        dict(
            train_args=detector.ckpt.get("train_args", {}),
            training_metrics=detector.ckpt.get("train_metrics", {}),
            weights_sha256=d["sha256"],
        ),
    )
    csv_save(
        dest / "detector_training_history.csv",
        __import__("pandas")
        .DataFrame(detector.ckpt.get("train_results", {}))
        .to_dict("records"),
    )
    save(
        dest / "native_detector_ap.json",
        dict(
            map50=float(report.box.map50),
            map50_95=float(report.box.map),
            scope="Native detector low-floor AP, before the separately evaluated learned background filter",
            policy_sha256=sha(policy),
            independent_test=False,
        ),
    )
    guide = read(policy)["mask"]["guide"]
    segmenter = YOLO(guide["path"])
    native_seg = segmenter.val(
        data=str(Path(cache) / "segmentation/data.yaml"),
        split="test",
        conf=0.001,
        iou=0.7,
        imgsz=guide["size"],
        rect=False,
        batch=4,
        device=0,
        plots=True,
        project=str(dest),
        name="native_segmenter_ap",
        exist_ok=True,
        verbose=False,
    )
    save(
        dest / "segmentation/native_mask_ap.json",
        dict(
            mask_map50=float(native_seg.seg.map50),
            mask_map50_95=float(native_seg.seg.map),
            scope="Native learned guide before SAM refinement, low confidence floor; distinct from accepted operating-mask AP",
            checkpoint_sha256=guide["sha256"],
            policy_sha256=sha(policy),
            independent_test=False,
        ),
    )
    save(
        dest / "segmenter_training_metadata.json",
        dict(
            train_args=segmenter.ckpt.get("train_args", {}),
            training_metrics=segmenter.ckpt.get("train_metrics", {}),
            weights_sha256=guide["sha256"],
        ),
    )
    csv_save(
        dest / "segmenter_training_history.csv",
        __import__("pandas")
        .DataFrame(segmenter.ckpt.get("train_results", {}))
        .to_dict("records"),
    )
    # COCO AP for the accepted operating-policy instance masks.
    audit = load(Path(__file__).parent / "legacy_audit.py", "mask_ap_evaluation")
    frames = read(dest / "segmentation/predictions.json")
    pred = dict(
        results=[
            dict(
                image=str(image_path(cache, f["id"])),
                objects=[
                    dict(
                        polygon_normalized=o["polygon_normalized"],
                        confidence=o["detector_confidence"],
                    )
                    for o in f["objects"]
                    if "polygon_normalized" in o
                ],
            )
            for f in frames
        ]
    )
    protocol = dict(reference_sha256={l.stem: sha(l) for l in labels})
    ap = audit.operating_mask_ap(
        pred, {r["id"]: r for r in rows}, root / "segmentation/labels/test", protocol
    )
    save(
        dest / "segmentation/operating_mask_ap.json",
        dict(ap, policy_sha256=sha(policy)),
    )


def cpu_runtime(root, cache):
    root, run, rows, _ = context(root)
    p = Pipeline(root, cache, run / "policy.json", "cpu")
    benchmark(p, rows, cache, run / "cpu_runtime")
