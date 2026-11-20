"""CPU checks of final-sprint contracts. Temporary/random weights are test fixtures only."""

import sys, json, tempfile, unittest, shutil
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/final_sprint"))
import common as c
import evaluate as ev
from runtime import Pipeline, CachedDetector


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = c.read(
            ROOT
            / "runs/v4_review_workspace/phase1_source/data/cct20_team_v2/manifest.json"
        )
        cls.classes = c.read(
            ROOT
            / "runs/v4_review_workspace/phase1_source/data/cct20_team_v2/species.json"
        )

    def test_frozen_support_and_split_scope(self):
        from train import samples

        self.assertEqual(len(self.rows), 6037)
        self.assertEqual(
            [
                len(ev.retained(self.rows, s))
                for s in ["train", "val_select", "val_calibrate", "test"]
            ],
            [3013, 1208, 467, 554],
        )
        y = [r["label"] for r in samples(self.rows, self.classes, "val_select")]
        self.assertEqual(len(set(y)), 9)
        z = np.zeros((len(y), 14))
        z[np.arange(len(y)), y] = 20
        score = c.metrics(z, y)
        self.assertAlmostEqual(score["macro_f1"], 9 / 14)
        self.assertEqual(score["accuracy"], 1.0)
        for key in ("camera", "event", "pixel_sha256"):
            group = {}
            for r in self.rows:
                group.setdefault(r[key], set()).add(r["split"])
            self.assertTrue(all(len(v) == 1 for v in group.values()))

    def test_matching_and_empty_decisions(self):
        box = [0.1, 0.1, 0.5, 0.5]
        self.assertEqual(c.pairs([box], [box, box]), [(0, 0)])
        self.assertEqual(c.pairs([], [box]), [])
        a = ev.frame_score(
            dict(id="positive", animal=True, boxes=[dict(xyxy=box)]), [box, box]
        )
        b = ev.frame_score(dict(id="empty", animal=False, boxes=[]), [])
        m = c.aggregate([a, b])
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 1, 0))
        self.assertEqual(m["animal_recall"], 1)
        self.assertEqual(m["empty_removal"], 1)

    def test_reference_hashes_and_no_dropped_frames(self):
        self.assertEqual(
            len(
                ev.reference(ROOT / "runs/v4_review_workspace", self.rows, "val_select")
            ),
            40,
        )

    def test_tampered_reference_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            shutil.copytree(
                ROOT / "runs/v4_review_workspace/segmentation", base / "segmentation"
            )
            label = next((base / "segmentation/labels/val_select").glob("*.txt"))
            label.write_text(label.read_text() + "\n")
            with self.assertRaises(AssertionError):
                ev.reference(base, self.rows, "val_select")

    def test_mask_union(self):
        p = [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]]
        self.assertTrue(
            np.array_equal(c.raster([p], 100, 80), c.raster([p, p], 100, 80))
        )
        self.assertFalse(c.raster([], 100, 80).any())

    def test_cpu_target_not_replaced_by_gpu(self):
        import artifacts

        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / "historical_test/segmentation").mkdir(parents=True)
            (out / "cpu_runtime").mkdir()
            (out / "gpu_runtime").mkdir()
            c.save(out / "policy.json", {"fixture": True})
            c.save(
                out / "historical_test/detection_metrics.json",
                dict(
                    animal_recall=0.98,
                    empty_removal=0.5,
                    precision=0.9,
                    recall=0.9,
                    f1=0.9,
                ),
            )
            c.save(
                out / "historical_test/species_metrics.json",
                dict(accuracy=0.85, macro_f1=0.48, ece=0.05),
            )
            c.save(
                out / "historical_test/segmentation/metrics.json", dict(mean_iou=0.82)
            )
            c.save(out / "historical_test/event_metrics.json", dict(mae=0.3))
            c.save(out / "cpu_runtime/benchmark.json", dict(p95=9.0))
            c.save(out / "gpu_runtime/benchmark.json", dict(p95=0.4))
            original = artifacts.context
            artifacts.context = lambda _: (out, out, [], [])
            try:
                result = artifacts.verdict(out)
            finally:
                artifacts.context = original
            self.assertFalse(
                next(
                    r["met"]
                    for r in result
                    if r["metric"] == "pipeline_cpu_p95_seconds"
                )
            )
            self.assertEqual(
                c.read(out / "decision.json")["performance_decision"], "NO-GO"
            )
            self.assertTrue(c.read(out / "improvement_targets.json")["gpu_latency_met"])

    def test_document_renderer(self):
        from artifacts import document
        from pypdf import PdfReader

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.pdf"
            document(
                path,
                "Fixture",
                [
                    dict(
                        title="Measured evidence",
                        text=["Temporary formatting test only."],
                        tables=[
                            (["Metric", "Value"], [["IoU", ".82"], ["CPU p95", "9.0"]])
                        ],
                    )
                ],
            )
            self.assertEqual(len(PdfReader(path).pages), 1)
            self.assertIn("Measured evidence", PdfReader(path).pages[0].extract_text())


class ActualInference(unittest.TestCase):
    def test_single_detector_forward_and_real_http(self):
        import torch, requests
        from runtime import Classifier
        from legacy_service import start_service

        torch.set_num_threads(4)
        with tempfile.TemporaryDirectory(prefix="final-sprint-test-") as temp:
            root = Path(temp)
            cache = root / "cache"
            cache.mkdir()
            (root / "releases").mkdir()
            release = Path("/workspace/exports/Segmentation_M4_Guided_Recovery_v2")
            (root / "releases/Segmentation_M4_Guided_Recovery_v2").symlink_to(
                release, target_is_directory=True
            )
            classes = c.read(
                ROOT
                / "runs/v4_review_workspace/phase1_source/data/cct20_team_v2/species.json"
            )
            net = c.model()
            checkpoint = root / "random_test_fixture.pt"
            torch.save(dict(classes=classes, state_dict=net.state_dict()), checkpoint)
            del net

            def ent(p, **kw):
                return dict(path=str(p), sha256=c.sha(p), **kw)

            classifier = ent(checkpoint, architecture="convnext_tiny_bg15")
            policy = dict(
                version="TEST_FIXTURE_ONLY",
                detector=ent(
                    release / "detector.pt", size=640, conf=0.045, animal_min=0
                ),
                animal_classifier=classifier,
                species_classifier=classifier,
                animal_temperature=1.0,
                species_temperature=1.0,
                mask_helper_sha256=c.sha(release / "predict_refined.py"),
                mask=dict(
                    kind="sam_t",
                    guide=ent(release / "selected.pt", size=640),
                    sam=ent(ROOT / "runs/assistant_segmentation/sam2.1_t.pt", size=640),
                    guide_conf=0.05,
                    primary_conf=0.4,
                    quality=0.0,
                ),
            )
            pipeline = Pipeline(root, cache, policy, "cpu")
            calls = []
            predict = pipeline.detector.predict

            def wrapper(*args, **kwargs):
                calls.append(1)
                return predict(*args, **kwargs)

            pipeline.detector.predict = wrapper
            path = (
                ROOT
                / "runs/assistant_segmentation/transfer/dataset/images/val_select/59d0e396-23d2-11e8-a6a3-ec086b02610b.jpg"
            )
            meta = dict(id=path.stem, event="test-fixture")
            frame = pipeline.process_frame(path, meta)
            self.assertEqual(len(calls), 1)
            self.assertGreater(frame["count"], 0)
            self.assertGreater(frame["segmentation"]["matched_masks"], 0)
            self.assertTrue(
                all(0 <= o.get("relative_body_area", 0) <= 1 for o in frame["objects"])
            )
            server, thread, base, _ = start_service(pipeline)
            try:
                response = requests.post(
                    base + "/batch",
                    files=[("images", (path.name, path.read_bytes(), "image/jpeg"))],
                    data={"metadata": json.dumps([meta])},
                    timeout=90,
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["frames"][0]["id"], path.stem)
                self.assertEqual(len(calls), 2)
                invalid = requests.post(
                    base + "/batch",
                    files=[("images", (path.name, path.read_bytes(), "image/jpeg"))],
                    data={"metadata": "[]"},
                    timeout=10,
                )
                self.assertEqual(invalid.status_code, 400)
            finally:
                server.should_exit = True
                thread.join(timeout=10)


if __name__ == "__main__":
    unittest.main()
