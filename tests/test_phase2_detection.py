import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from phase2.detection import digest, prepare, verify_review


class DetectionPreparationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "handoff").mkdir()
        (self.root / "data").mkdir()
        rows = []
        for index, split in enumerate(("train", "val_select", "val_calibrate", "test")):
            image = self.root / "data" / f"sample{index}.jpg"
            Image.new("RGB", (80, 60), "white").save(image)
            rows.append({"id": f"sample{index}", "split": split, "camera": f"camera{index}",
                         "event": f"event{index}", "pixel_sha256": str(index), "path": str(image),
                         "animal": 0 if index == 1 else 1,
                         "boxes": [] if index == 1 else [{"xyxy": [0.1, 0.2, 0.5, 0.8],
                                                              "annotation_source": "human_CCT20"}],
                         "incomplete_boxes": False})
        self.manifest = self.root / "data" / "manifest.json"
        self.manifest.write_text(json.dumps(rows))
        self.release = self.root / "handoff" / "dataset_release.json"
        self.release.write_text(json.dumps({"schema": 2, "audit_complete": True,
                                            "files": {"data/manifest.json": digest(self.manifest)}}))

    def tearDown(self):
        self.temp.cleanup()

    def test_prepare_and_review_gate(self):
        output = self.root / "detector"
        report = prepare(self.release, output, sample_count=3)
        self.assertEqual(report["split_counts"], {"train": 1, "val_select": 1, "val_calibrate": 1, "test": 1})
        self.assertEqual((output / "labels" / "train" / "sample0.txt").read_text().strip(),
                         "0 0.30000000 0.50000000 0.40000000 0.60000000")
        self.assertEqual((output / "labels" / "val" / "sample1.txt").read_text(), "")
        self.assertTrue((output / "images" / "test" / "sample3.jpg").is_symlink())
        with self.assertRaises(ValueError):
            verify_review(output)
        review = output / "box_review.csv"
        with review.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row["reviewer"] = "Reviewer"
            row["boxes_correct"] = "yes"
        with review.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        self.assertEqual(verify_review(output), len(rows))

    def test_reject_changed_release_and_leakage(self):
        self.manifest.write_text(self.manifest.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "differs"):
            prepare(self.release, self.root / "changed")
        self.release.write_text(json.dumps({"schema": 2, "audit_complete": True,
                                            "files": {"data/manifest.json": digest(self.manifest)}}))
        rows = json.loads(self.manifest.read_text())
        rows[3]["camera"] = rows[0]["camera"]
        self.manifest.write_text(json.dumps(rows))
        self.release.write_text(json.dumps({"schema": 2, "audit_complete": True,
                                            "files": {"data/manifest.json": digest(self.manifest)}}))
        with self.assertRaisesRegex(ValueError, "leakage"):
            prepare(self.release, self.root / "leaked")


if __name__ == "__main__":
    unittest.main()
