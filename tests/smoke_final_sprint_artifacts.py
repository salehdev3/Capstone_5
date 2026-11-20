"""Temporary fixtures exercise complete PDF/slide/video generation; no model scores are published."""

import sys, tempfile, shutil, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/final_sprint"))
from common import save, csv_save, read, sha
from runtime import DEMO_IDS
import artifacts, video
from PIL import Image
from pypdf import PdfReader
import cv2


class GeneratedArtifacts(unittest.TestCase):
    def test_complete_document_and_video_paths(self):
        with tempfile.TemporaryDirectory(prefix="final-sprint-artifacts-") as temp:
            root = Path(temp)
            out = root / "outputs"
            out.mkdir()
            cache = root / "cache"
            (cache / "images").mkdir(parents=True)
            rows = read(
                ROOT
                / "runs/v4_review_workspace/phase1_source/data/cct20_team_v2/manifest.json"
            )
            classes = read(
                ROOT
                / "runs/v4_review_workspace/phase1_source/data/cct20_team_v2/species.json"
            )
            context = lambda _: (root, out, rows, classes)
            release = Path("/workspace/exports/Segmentation_M4_Guided_Recovery_v2")
            image = (
                ROOT
                / "runs/assistant_segmentation/transfer/dataset/images/val_select"
                / f"{DEMO_IDS[0]}.jpg"
            )
            fixture = dict(
                version="TEST_FIXTURE_ONLY",
                detector=dict(path=str(release / "detector.pt"), size=640, conf=0.045),
                species_classifier=dict(architecture="convnext_tiny_bg15"),
            )
            fixture["detector"]["animal_min"] = 0.0
            save(out / "policy.json", fixture)
            save(out / "freeze.json", dict(policy_sha256=sha(out / "policy.json")))
            for name, value in {
                "historical_test/detection_metrics.json": dict(
                    animal_recall=0.97,
                    empty_removal=0.5,
                    precision=0.8,
                    recall=0.85,
                    f1=0.82,
                ),
                "historical_test/species_metrics.json": dict(
                    accuracy=0.74, macro_f1=0.48, ece=0.05
                ),
                "historical_test/segmentation/metrics.json": dict(
                    mean_iou=0.74, mean_dice=0.80, area_mae_fraction=0.02
                ),
                "historical_test/event_metrics.json": dict(
                    mae=0.3, events=288, excluded_events=47
                ),
                "historical_test/native_detector_ap.json": dict(
                    map50=0.9, map50_95=0.57
                ),
                "historical_test/segmentation/operating_mask_ap.json": dict(
                    mask_map50=0.8, mask_map50_95=0.6
                ),
                "cpu_runtime/benchmark.json": dict(
                    p50=0.5, p95=9.0, p99=9.1, throughput=0.3
                ),
                "gpu_runtime/benchmark.json": dict(
                    p50=0.1, p95=0.3, p99=0.4, throughput=5.0
                ),
                "onnx/parity.json": dict(
                    frames=1208,
                    raw_pass=True,
                    decision_pass=True,
                    metric_pass=True,
                    passed=True,
                ),
                "mask_selection.json": dict(
                    selected=dict(name="TEST_FIXTURE", mean_iou=0.82),
                    candidates=[
                        dict(name="TEST_FIXTURE", mean_iou=0.82, mean_dice=0.88)
                    ],
                ),
                "data_support.json": dict(training_counts=[1] * 15),
                "http/demo_record.json": dict(coverage=dict(all_stages_exercised=True)),
            }.items():
                save(out / name, value)
            csv_save(
                out / "classifier/history.csv",
                [
                    dict(
                        epoch=1,
                        train_subset_accuracy=0.8,
                        val_accuracy=0.75,
                        accuracy_gap=0.05,
                    )
                ],
            )

            def gallery(*args):
                target = out / "failure_gallery"
                target.mkdir()
                shutil.copyfile(image, target / "det_fixture.png")
                shutil.copyfile(image, target / "mask_fixture.png")

            frames = []
            for ident in DEMO_IDS:
                shutil.copyfile(image, cache / "images" / (ident + ".jpg"))
                (out / "http").mkdir(exist_ok=True)
                shutil.copyfile(image, out / "http" / (ident + ".png"))
                frames.append(
                    dict(
                        id=ident,
                        count=1,
                        objects=[],
                        segmentation=dict(matched_masks=1),
                        timing=dict(total=0.3),
                    )
                )
            save(
                out / "http/service_test.json",
                dict(
                    status=200,
                    result=dict(frames=frames, events=[dict(event="test-fixture")]),
                ),
            )
            with patch.object(artifacts, "context", context), patch.object(
                artifacts, "failure_gallery", gallery
            ):
                artifacts.build(root, cache)
            report = PdfReader(out / "submission/Phase2_Report.pdf")
            slides = PdfReader(out / "submission/Final_Presentation.pdf")
            self.assertGreaterEqual(len(report.pages), 10)
            self.assertEqual(len(slides.pages), 14)
            text = "\n".join(p.extract_text() for p in report.pages)
            self.assertIn("NO-GO", text)
            self.assertIn("9.000", text)
            self.assertIn("no new independent", text)
            with patch.object(video, "context", context):
                video.run(root, cache)
            provenance = read(out / "submission/video_provenance.json")
            self.assertEqual(provenance["decoded_frames"], 50)
            self.assertEqual(provenance["inference_frames"], 5)
            for name in ["Demo_Fallback.mp4", "Detector_Video.mp4"]:
                cap = cv2.VideoCapture(str(out / "submission" / name))
                self.assertTrue(cap.isOpened())
                okay, frame = cap.read()
                self.assertTrue(okay)
                self.assertEqual(frame.shape[:2], (768, 1024))
                cap.release()
            destination = Path("/tmp/final_sprint_visual_smoke")
            destination.mkdir(exist_ok=True)
            # Visually inspectable formatting fixtures, outside the repository and source archive.
            for name in ["Phase2_Report.pdf", "Final_Presentation.pdf"]:
                shutil.copyfile(out / "submission" / name, destination / name)


if __name__ == "__main__":
    unittest.main()
