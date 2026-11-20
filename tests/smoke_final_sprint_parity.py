"""CPU export/parity on a temporary repeated-image fixture, not the full project evaluation."""

import sys, tempfile, unittest, shutil
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/final_sprint"))
from common import read, save, sha
import parity


class ExportParity(unittest.TestCase):
    def test_real_export_and_nms(self):
        with tempfile.TemporaryDirectory(prefix="final-sprint-parity-fixture-") as temp:
            root = Path(temp)
            out = root / "outputs"
            out.mkdir()
            cache = root / "cache"
            (cache / "images").mkdir(parents=True)
            manifest = read(
                ROOT
                / "runs/v4_review_workspace/phase1_source/data/cct20_team_v2/manifest.json"
            )
            ident = "59d0e396-23d2-11e8-a6a3-ec086b02610b"
            row = next(r for r in manifest if r["id"] == ident)
            shutil.copyfile(
                ROOT
                / "runs/assistant_segmentation/transfer/dataset/images/val_select"
                / (ident + ".jpg"),
                cache / "images" / (ident + ".jpg"),
            )
            weights = Path(
                "/workspace/exports/Segmentation_M4_Guided_Recovery_v2/detector.pt"
            )
            save(
                out / "policy.json",
                dict(
                    version="TEST_FIXTURE_ONLY",
                    detector=dict(
                        path=str(weights), sha256=sha(weights), conf=0.045, size=640
                    ),
                ),
            )
            with patch.object(parity, "context", lambda _: (root, out, [row] * 12, [])):
                parity.run(root, cache)
            result = read(out / "onnx/parity.json")
            self.assertEqual(result["frames"], 12)
            self.assertTrue(result["passed"])
            self.assertEqual(result["absolute_metric_differences"]["f1"], 0.0)
            self.assertTrue((out / "onnx/detector.onnx").is_file())


if __name__ == "__main__":
    unittest.main()
