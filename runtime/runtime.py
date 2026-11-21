"""Inference-only contracts. No reference labels or manifest are read by Pipeline."""

from common import *
import cv2, torch, time, shutil
from ultralytics import YOLO, SAM

DEMO_IDS = [
    "59d0e396-23d2-11e8-a6a3-ec086b02610b",
    "59a62b26-23d2-11e8-a6a3-ec086b02610b",
    "5860ed49-23d2-11e8-a6a3-ec086b02610b",
    "58629431-23d2-11e8-a6a3-ec086b02610b",
    "5888bc6a-23d2-11e8-a6a3-ec086b02610b",
]


def sources(root, cache):
    root, run, _, _ = context(root)
    release = root / "releases/Segmentation_M4_Guided_Recovery_v2"
    p = read(release / "refinement_policy.json")
    assert (
        sha(release / "predict_refined.py")
        == read(release / "artifact_checksums.json")["predict_refined.py"]
    )
    specs = {
        "d_old": (
            release / "detector.pt",
            "ac38aaffe735bf7a6666866d2410f3601a93c6f9829ae6f68b7cd0117d1135e8",
            640,
        ),
        "d_new": (
            root / "phase2_results/optimization_v3/detection/weights/best.pt",
            "d9403728c5b53ca0355f369e0ffcf55d6f3f13e9f711d48d59215a05a86d1601",
            960,
        ),
        "m_old": (
            release / "selected.pt",
            "f19fbfb60b2b2d1f08de7ae9c69e4a425caf3e3c264bff98212159ece656d39d",
            640,
        ),
        "m_v3": (
            root / "phase2_results/optimization_v3/segmentation/weights/best.pt",
            "60dfc13188fec981424105452db51e44a00ca5fd3530b8d6a2397653d460b4b2",
            960,
        ),
        "m_v4": (
            root / "phase2_results/mask_training_v4/segmenter/weights/best.pt",
            "bb89f9a6ec1b19a48c60de266f0cf08171ca5798bdd34e75996f1b0eed3eaaa0",
            960,
        ),
    }
    sam = release / "sam_b.pt"
    if not sam.exists():
        sam = root / "releases/Segmentation_M4_Guided/sam_b.pt"
    specs["sam_b"] = (
        sam,
        "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
        640,
    )
    tiny = Path(cache) / "sam2.1_t.pt"
    if not tiny.exists():
        import urllib.request

        tmp = tiny.with_suffix(".download")
        urllib.request.urlretrieve(
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_t.pt",
            tmp,
        )
        tmp.replace(tiny)
    specs["sam_t"] = (
        tiny,
        "3c1e81ca9b037dd39d70a014ddb9a813d6c4c4e12555420db7eaff31689bd4e3",
        1024,
    )
    available = {}
    missing = []
    for name, (path, digest, size) in specs.items():
        if not path.is_file():
            missing.append(name)
            continue
        assert sha(path) == digest, "Checkpoint differs: " + str(path)
        target = Path(cache) / "weights" / f"{name}.pt"
        target.parent.mkdir(exist_ok=True)
        if not target.exists() or sha(target) != digest:
            shutil.copyfile(path, target)
        available[name] = dict(
            path=str(path), cached=str(target), sha256=digest, size=size
        )
    assert all(
        k in available for k in ["d_old", "m_old", "sam_b", "sam_t"]
    ), "Original release weights are required"
    save(
        run / "sources.json",
        dict(
            models=available,
            optional_models_missing=missing,
            mask_helper_sha256=sha(release / "predict_refined.py"),
        ),
    )
    return available


class Classifier:
    def __init__(self, path, device, architecture="convnext_tiny_bg15"):
        self.path = Path(path)
        self.device = device
        self.architecture = architecture
        ck = torch.load(path, map_location="cpu", weights_only=True)
        self.classes = ck["classes"]
        assert len(self.classes) == 14
        if architecture == "convnext_tiny_bg15":
            self.net = model()
        else:
            self.net = load(
                Path(__file__).parent / "legacy_optimize.py", "legacy_species"
            ).species_model(self.classes, ck)
        self.net.load_state_dict(ck["state_dict"])
        self.net.to(device).eval()

    def logits(self, im, boxes):
        if not len(boxes):
            return np.empty(
                (0, 15 if self.architecture == "convnext_tiny_bg15" else 14)
            )
        x = np.stack([normalized(crop(im, b)).transpose(2, 0, 1) for b in boxes])
        out = []
        with torch.inference_mode():
            for i in range(0, len(x), 32):
                out.append(
                    self.net(torch.from_numpy(x[i : i + 32]).to(self.device))
                    .float()
                    .cpu()
                    .numpy()
                )
        return np.concatenate(out)


class CachedDetector:
    """Expose already-computed predictions to the unchanged mask helper."""

    def __init__(self, result):
        self.result = result

    def predict(self, *args, **kwargs):
        return [self.result]


class Pipeline:
    def __init__(self, root, cache, policy, device="cpu", masks=True):
        self.root = Path(root)
        self.cache = Path(cache)
        self.config = read(policy) if isinstance(policy, (str, Path)) else policy
        self.device = device
        torch.set_num_threads(4)
        self.policy_sha256 = (
            sha(policy)
            if isinstance(policy, (str, Path))
            else hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
        )
        c = self.config
        self.detector = YOLO(self.checked(c["detector"]))
        same_species_source = (
            c["species_classifier"]["sha256"] == c["animal_classifier"]["sha256"]
        )
        self.animal = (
            Classifier(self.checked(c["animal_classifier"]), device)
            if c["detector"]["animal_min"] > 0 or same_species_source
            else None
        )
        self.species = (
            self.animal
            if c["species_classifier"]["sha256"] == c["animal_classifier"]["sha256"]
            else Classifier(
                self.checked(c["species_classifier"]),
                device,
                c["species_classifier"]["architecture"],
            )
        )
        self.mask_enabled = masks
        self.segmenter = None
        self.sam = None
        if masks:
            self.segmenter = YOLO(self.checked(c["mask"]["guide"]))
            if c["mask"]["kind"] != "direct":
                self.sam = SAM(self.checked(c["mask"]["sam"]))
            release = self.root / "releases/Segmentation_M4_Guided_Recovery_v2"
            assert sha(release / "predict_refined.py") == c["mask_helper_sha256"]
            self.helper = load(release / "predict_refined.py", "verified_mask_helper")
            self.base_policy = read(release / "refinement_policy.json")

    def checked(self, entry):
        source = Path(entry["path"])
        # Preserve the SAM filename: Ultralytics uses it to select its architecture.
        # Content-addressed directories also avoid collisions between best.pt files.
        target = self.cache / "runtime_weights" / entry["sha256"][:16] / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or sha(target) != entry["sha256"]:
            assert sha(source) == entry["sha256"], "Model changed: " + str(source)
            shutil.copyfile(source, target)
        return str(target)

    def sync(self):
        if self.device != "cpu":
            torch.cuda.synchronize()

    def detect(self, im):
        c = self.config
        d = c["detector"]
        result = self.detector.predict(
            im,
            conf=d["conf"],
            iou=PLAN["nms"],
            max_det=PLAN["max_boxes"],
            imgsz=d["size"],
            rect=False,
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes.xyxyn.cpu().numpy()
        # Reject degenerate or out-of-frame detections before cropping.
        valid = (
            np.isfinite(boxes).all(axis=1)
            & (boxes[:, 0] < boxes[:, 2])
            & (boxes[:, 1] < boxes[:, 3])
            & (boxes[:, 0] < 1) & (boxes[:, 1] < 1)
            & (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
        )
        result = result[np.flatnonzero(valid).tolist()]
        boxes = result.boxes.xyxyn.cpu().numpy()
        if self.animal is None:
            return result, np.empty((len(boxes), 0))
        z = self.animal.logits(im, boxes)
        from scipy.special import softmax

        keep = (
            np.flatnonzero(
                (1 - softmax(z / c["animal_temperature"], axis=1)[:, 14])
                >= d["animal_min"]
            )
            if len(z)
            else np.array([], dtype=int)
        )
        result = result[keep.tolist()]
        z = z[keep]
        return result, z

    def process_frame(self, path, meta, with_masks=True):
        self.sync()
        start = time.perf_counter()
        im = cv2.imread(str(path))
        if im is None:
            raise ValueError("Unreadable image")
        result, z = self.detect(im)
        self.sync()
        det_end = time.perf_counter()
        boxes = result.boxes.xyxyn.cpu().numpy().tolist()
        scores = result.boxes.conf.cpu().numpy().tolist()
        if self.species is not self.animal:
            z = self.species.logits(im, boxes)
        from scipy.special import softmax

        probs = softmax(z[:, :14] / self.config["species_temperature"], axis=1)
        objects = []
        for box, det_score, p in zip(boxes, scores, probs):
            i = int(p.argmax())
            objects.append(
                dict(
                    box=box,
                    detector_confidence=float(det_score),
                    species=self.species.classes[i],
                    confidence=float(p[i]),
                    review=bool(p[i] < 0.6),
                )
            )
        self.sync()
        cls_end = time.perf_counter()
        polygons = []
        maskboxes = []
        if self.mask_enabled and with_masks and boxes:
            m = self.config["mask"]
            if m["kind"] == "direct":
                seg = self.segmenter.predict(
                    im,
                    conf=m["guide_conf"],
                    iou=0.7,
                    imgsz=m["guide"]["size"],
                    rect=False,
                    retina_masks=True,
                    device=self.device,
                    verbose=False,
                )[0]
                if seg.masks is not None:
                    polygons = [p.tolist() for p in seg.masks.xyn]
                    maskboxes = seg.boxes.xyxyn.cpu().numpy().tolist()
            else:
                p = dict(self.base_policy)
                p.pop("validation", None)
                p.update(
                    detector_input=self.config["detector"]["size"],
                    prompt_floor=self.config["detector"]["conf"],
                    segmenter_input=m["guide"]["size"],
                    segmenter_conf=m["guide_conf"],
                    sam_input=m["sam"]["size"],
                    sam_quality=m["quality"],
                    conf=m["primary_conf"],
                    max_prompts=PLAN["max_boxes"],
                )
                p["fallback"] = dict(
                    p["fallback"], confidence_floor=self.config["detector"]["conf"]
                )
                refined = self.helper.predict_image(
                    CachedDetector(result),
                    self.segmenter,
                    self.sam,
                    path,
                    p,
                    self.device,
                )
                polygons = [o["polygon_normalized"] for o in refined["objects"]]
                maskboxes = [o["box_xyxy_normalized"] for o in refined["objects"]]
            for i, j in pairs(boxes, maskboxes):
                objects[i].update(
                    polygon_normalized=polygons[j],
                    relative_body_area=float(
                        raster([polygons[j]], im.shape[1], im.shape[0]).mean()
                    ),
                )
        for o in objects:
            o["review"] = o["review"] or "relative_body_area" not in o
        self.sync()
        end = time.perf_counter()
        return dict(
            id=meta["id"],
            event=meta["event"],
            timestamp=meta.get("timestamp"),
            objects=objects,
            count=len(objects),
            segmentation=dict(
                matched_masks=sum("relative_body_area" in o for o in objects),
                policy_sha256=self.policy_sha256,
            ),
            timing=dict(
                detection_and_animal_filter=det_end - start,
                classification=cls_end - det_end,
                segmentation=end - cls_end,
                total=end - start,
            ),
        )

    def process_batch(self, paths, metadata):
        if len(paths) != len(metadata) or not paths:
            raise ValueError("One metadata record per image is required")
        frames = [self.process_frame(p, m) for p, m in zip(paths, metadata)]
        events = []
        for event in sorted({f["event"] for f in frames}):
            group = [f for f in frames if f["event"] == event]
            counts = {}
            for f in group:
                per_frame = {}
                for o in f["objects"]:
                    per_frame[o["species"]] = per_frame.get(o["species"], 0) + 1
                for k, v in per_frame.items():
                    counts[k] = max(counts.get(k, 0), v)
            if max(f["count"] for f in group):
                events.append(
                    dict(
                        event=event,
                        count=max(f["count"] for f in group),
                        species_counts=counts,
                    )
                )
        return dict(frames=frames, events=events)
