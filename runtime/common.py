from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, math, random, importlib.util
import numpy as np
from PIL import Image

MANIFEST_SHA = "d3bfb6ba0c9d31412f9d603a6f63ae7c4890f458c5d2aa2a87d7653bb882bb53"
PLAN = dict(
    version="final_sprint_v1",
    seed=42,
    epochs=18,
    patience=5,
    batch=32,
    size=224,
    margin=0.05,
    max_boxes=20,
    raw_conf=0.005,
    nms=0.3,
    train_backgrounds_per_image=4,
    selection_recall_margin=0.995,
    original_targets=dict(
        animal_recall_min=0.99,
        empty_removal_min=0.2,
        species_macro_f1_min=0.65,
        species_ece_max=0.1,
        mask_iou_min=0.7,
        event_count_mae_max=1.0,
        p95_cpu_seconds_max=2.0,
    ),
    stretch_targets=dict(
        species_accuracy_min=0.8, species_macro_f1_min=0.8, mask_iou_min=0.8
    ),
    added_gpu_latency_target_seconds=2.0,
    historical_test_is_independent=False,
)


def sha(p):
    digest = hashlib.sha256()
    with Path(p).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def save(p, d):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + ".tmp")
    temp.write_text(
        json.dumps(
            d,
            indent=2,
            allow_nan=False,
            default=lambda v: (
                str(v)
                if isinstance(v, Path)
                else (_ for _ in ()).throw(TypeError(type(v).__name__))
            ),
        )
    )
    temp.replace(p)


def table(p):
    with Path(p).open() as f:
        return list(csv.DictReader(f))


def csv_save(p, rows):
    if not rows:
        return
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def load(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def context(root):
    root = Path(root)
    run = root / "phase2_results/final_sprint_v1"
    run.mkdir(parents=True, exist_ok=True)
    if (run / "plan.json").exists():
        assert (
            read(run / "plan.json") == PLAN
        ), "Configuration changed; use a new version"
    else:
        save(run / "plan.json", PLAN)
    manifest = root / "phase1_source/data/cct20_team_v2/manifest.json"
    assert sha(manifest) == MANIFEST_SHA
    rows = read(manifest)
    classes = read(root / "phase1_source/data/cct20_team_v2/species.json")
    assert len(classes) == 14
    for k in ("camera", "event", "pixel_sha256"):
        groups = {}
        for r in rows:
            groups.setdefault(r[k], set()).add(r["split"])
        assert all(len(v) == 1 for v in groups.values()), k
    return root, run, rows, classes


def crop(bgr, box):
    import cv2

    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = box
    dx = (x2 - x1) * 0.05
    dy = (y2 - y1) * 0.05
    l, t = min(w - 1, max(0, int((x1 - dx) * w))), min(h - 1, max(0, int((y1 - dy) * h)))
    r, b = min(w, max(l + 1, int((x2 + dx) * w))), min(
        h, max(t + 1, int((y2 + dy) * h))
    )
    im = cv2.cvtColor(bgr[t:b, l:r], cv2.COLOR_BGR2RGB)
    ih, iw = im.shape[:2]
    return cv2.resize(
        im,
        (224, 224),
        interpolation=cv2.INTER_AREA if ih > 224 or iw > 224 else cv2.INTER_LINEAR,
    )


def normalized(rgb):
    return (
        rgb.astype(np.float32) / 255 - np.array([0.485, 0.456, 0.406], np.float32)
    ) / np.array([0.229, 0.224, 0.225], np.float32)


def metrics(z, y, temperature=1.0):
    from scipy.special import softmax
    from sklearn.metrics import confusion_matrix, f1_score

    y = np.asarray(y, dtype=int)
    p = softmax(np.asarray(z) / temperature, axis=1)
    pred = p.argmax(1)
    confidence = p.max(1)
    correct = pred == y
    ece = 0.0
    for i in range(15):
        hit = (confidence >= i / 15) & (
            confidence < (i + 1) / 15 if i < 14 else confidence <= 1
        )
        if hit.any():
            ece += hit.mean() * abs(correct[hit].mean() - confidence[hit].mean())
    cm = confusion_matrix(y, pred, labels=list(range(p.shape[1])))
    support = cm.sum(1)
    return dict(
        n=len(y),
        accuracy=float(correct.mean()),
        macro_f1=float(
            f1_score(
                y,
                pred,
                labels=list(range(p.shape[1])),
                average="macro",
                zero_division=0,
            )
        ),
        ece=float(ece),
        confusion_matrix=cm.tolist(),
        support=support.tolist(),
        all_class_macro_f1_support_ceiling=float((support > 0).mean()),
    )


def temperature(z, y):
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp

    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=int)
    f = lambda t: float(
        np.mean(
            logsumexp(z / math.exp(t), axis=1) - (z / math.exp(t))[np.arange(len(y)), y]
        )
    )
    fit = minimize_scalar(f, bounds=(-3, 3), method="bounded")
    assert fit.success
    return float(math.exp(fit.x))


def model(pretrained=False):
    import torch
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

    net = convnext_tiny(
        weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    )
    net.classifier[2] = torch.nn.Sequential(
        torch.nn.Dropout(0.25), torch.nn.Linear(net.classifier[2].in_features, 15)
    )
    return net


def pairs(gt, pred):
    from scipy.optimize import linear_sum_assignment

    a = np.asarray(gt, float).reshape(-1, 4)
    b = np.asarray(pred, float).reshape(-1, 4)
    if not len(a) or not len(b):
        return []
    inter = np.maximum(
        np.minimum(a[:, None, 2:], b[None, :, 2:])
        - np.maximum(a[:, None, :2], b[None, :, :2]),
        0,
    ).prod(2)
    union = (
        np.maximum(a[:, 2:] - a[:, :2], 0).prod(1)[:, None]
        + np.maximum(b[:, 2:] - b[:, :2], 0).prod(1)[None, :]
        - inter
    )
    iou = inter / np.maximum(union, 1e-9)
    i, j = linear_sum_assignment(-(iou >= 0.5).astype(float) * 1000 - iou)
    return [(int(x), int(y)) for x, y in zip(i, j) if iou[x, y] >= 0.5]


def raster(polygons, w, h):
    from PIL import ImageDraw

    im = Image.new("1", (w, h))
    draw = ImageDraw.Draw(im)
    for p in polygons:
        if len(p) >= 3:
            draw.polygon([(x * w, y * h) for x, y in p], fill=1)
    return np.asarray(im, dtype=bool)


def aggregate(records):
    tp = sum(r["tp"] for r in records)
    fp = sum(r["fp"] for r in records)
    fn = sum(r["fn"] for r in records)
    positive = [r for r in records if r["animal"]]
    negative = [r for r in records if not r["animal"]]
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=tp / max(tp + fp, 1),
        recall=tp / max(tp + fn, 1),
        f1=2 * tp / max(2 * tp + fp + fn, 1),
        animal_recall=sum(r["count"] > 0 for r in positive) / max(len(positive), 1),
        empty_removal=sum(r["count"] == 0 for r in negative) / max(len(negative), 1),
        frames=len(records),
    )


def seed():
    import torch

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def image_path(cache, ident):
    return Path(cache) / "images" / (ident + ".jpg")
