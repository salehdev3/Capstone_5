"""Detector export and measured raw/post-NMS parity at the frozen operating point."""

from common import *
from evaluate import retained, frame_score
import numpy as np, torch, cv2, shutil, platform, time


def run(root, cache):
    import onnxruntime as ort
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.ops import scale_boxes
    from ultralytics.utils.nms import non_max_suppression
    from scipy.optimize import linear_sum_assignment

    root, out, rows, _ = context(root)
    dest = out / "onnx"
    dest.mkdir(exist_ok=True)
    p = read(out / "policy.json")
    d = p["detector"]
    size = d["size"]
    weights = dest / "detector.pt"
    if not weights.exists() or sha(weights) != d["sha256"]:
        shutil.copyfile(d["path"], weights)
    torch.set_num_threads(4)
    model = YOLO(str(weights))
    model.model.cpu().eval()
    exported = model.export(
        format="onnx",
        imgsz=size,
        dynamic=False,
        simplify=False,
        nms=False,
        opset=17,
        device="cpu",
        half=False,
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(exported), sess_options=options, providers=["CPUExecutionProvider"]
    )
    assert session.get_inputs()[0].shape == [1, 3, size, size]
    model.model.fuse(verbose=False)
    model.model.float().eval()
    letterbox = LetterBox(new_shape=(size, size), auto=False, scaleup=True, stride=32)
    records = []
    scores = {"pytorch": [], "onnx": []}
    latencies = {"pytorch": [], "onnx": []}
    chosen = retained(rows, "val_select")
    for i, r in enumerate(chosen, 1):
        bgr = cv2.imread(str(image_path(cache, r["id"])))
        padded = letterbox(image=bgr)
        x = (
            np.ascontiguousarray(
                padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32
            )
            / 255
        )
        with torch.inference_mode():
            start = time.perf_counter()
            result = model.model(torch.from_numpy(x))
            native = (
                (result[0] if isinstance(result, (tuple, list)) else result)
                .cpu()
                .numpy()
            )
            latencies["pytorch"].append(time.perf_counter() - start)
        start = time.perf_counter()
        onnx = session.run(None, {session.get_inputs()[0].name: x})[0]
        latencies["onnx"].append(time.perf_counter() - start)
        a = []
        for raw in (native, onnx):
            det = (
                non_max_suppression(
                    torch.from_numpy(raw.copy()),
                    conf_thres=d["conf"],
                    iou_thres=PLAN["nms"],
                    max_det=PLAN["max_boxes"],
                    nc=1,
                )[0]
                .cpu()
                .numpy()
            )
            a.append(det)
        aa, bb = a
        if len(aa) and len(bb):
            lo = np.maximum(aa[:, None, :2], bb[None, :, :2])
            hi = np.minimum(aa[:, None, 2:4], bb[None, :, 2:4])
            inter = np.maximum(hi - lo, 0).prod(2)
            union = (
                np.maximum(aa[:, 2:4] - aa[:, :2], 0).prod(1)[:, None]
                + np.maximum(bb[:, 2:4] - bb[:, :2], 0).prod(1)[None, :]
                - inter
            )
            iou = inter / np.maximum(union, 1e-9)
            ia, ib = linear_sum_assignment(-iou)
            box_error = float(np.abs(aa[ia, :4] - bb[ib, :4]).max())
            score_error = float(np.abs(aa[ia, 4] - bb[ib, 4]).max())
            matched = int(((iou[ia, ib] >= 0.999) & (aa[ia, 5] == bb[ib, 5])).sum())
        else:
            matched = 0
            box_error = score_error = 0.0
        raw_pass = bool(
            native.shape == onnx.shape
            and np.allclose(native, onnx, atol=0.001, rtol=0.001)
        )
        decision_pass = bool(
            len(aa) == len(bb) == matched and box_error <= 0.1 and score_error <= 1e-4
        )
        records.append(
            dict(
                id=r["id"],
                raw_pass=raw_pass,
                decision_pass=decision_pass,
                raw_max_abs=float(np.max(np.abs(native - onnx))),
                matched=matched,
                pytorch_count=len(aa),
                onnx_count=len(bb),
                max_box_error=box_error,
                max_confidence_error=score_error,
            )
        )
        for backend, det in zip(("pytorch", "onnx"), a):
            boxes = scale_boxes(
                (size, size), torch.tensor(det[:, :4]), bgr.shape[:2]
            ).numpy() / np.array([r["width"], r["height"], r["width"], r["height"]])
            scores[backend].append(frame_score(r, boxes.tolist()))
        if i % 50 == 0:
            print(
                "ONNX parity",
                i,
                "/",
                len(chosen),
                "mismatches",
                sum(not v["decision_pass"] for v in records),
                flush=True,
            )
    m = {k: aggregate(v) for k, v in scores.items()}
    keys = ["precision", "recall", "f1", "animal_recall", "empty_removal"]
    delta = {k: abs(m["pytorch"][k] - m["onnx"][k]) for k in keys}
    summary = dict(
        frames=len(records),
        raw_pass=all(r["raw_pass"] for r in records),
        decision_pass=all(r["decision_pass"] for r in records),
        metrics=m,
        absolute_metric_differences=delta,
        metric_pass=all(v <= 1e-6 for v in delta.values()),
        raw_max_abs=max(r["raw_max_abs"] for r in records),
        detector_sha256=d["sha256"],
        onnx_sha256=sha(exported),
        policy_sha256=sha(out / "policy.json"),
        scope="FP32 detector only, common square preprocessing and common post-NMS; learned background filter, species classifier and SAM remain native PyTorch; val_select only",
        tolerances=dict(
            raw_rtol=0.001,
            raw_atol=0.001,
            box_pixels=0.1,
            confidence=0.0001,
            min_iou=0.999,
        ),
        input_size=size,
    )
    summary["passed"] = (
        summary["raw_pass"] and summary["decision_pass"] and summary["metric_pass"]
    )
    csv_save(dest / "per_image.csv", records)
    save(dest / "parity.json", summary)
    timing = []
    for backend, seconds in latencies.items():
        seconds = seconds[10:]
        timing.append(
            dict(
                backend=backend,
                p50=float(np.quantile(seconds, 0.5)),
                p95=float(np.quantile(seconds, 0.95)),
                p99=float(np.quantile(seconds, 0.99)),
                throughput=1 / float(np.mean(seconds)),
                warmup=10,
                repeats=len(seconds),
                scope="Model forward only, CPU, excludes preprocessing/NMS; not full-pipeline latency",
            )
        )
    csv_save(dest / "benchmark.csv", timing)
    save(
        dest / "environment.json",
        dict(
            torch=torch.__version__,
            onnxruntime=ort.__version__,
            python=platform.python_version(),
            device="cpu",
            threads=4,
        ),
    )
    print("ONNX parity:", summary["passed"], flush=True)
