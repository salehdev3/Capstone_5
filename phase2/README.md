# Phase 2: camera-trap detection, segmentation and integration

Project 5 requires all three components: animal detection (M4), verified-mask segmentation (M5), and an end-to-end batch service (M8). Our four-person team does not take the open-world extension. The official Phase 2 report source is in `docs/phase2_report/official_template/`; Section 5 was removed as instructed. Its remaining prompts are intentionally unfilled until the corresponding experiment exists.

## Current milestone: detection data and training entry point

The detection preparation command consumes the frozen Phase 1 `handoff/dataset_release.json` and its `manifest.json`. It checks the recorded manifest hash, split separation, image readability, normalized box coordinates and human provenance. It excludes animal frames with missing or incomplete boxes instead of treating them as negative images. Empty frames with complete annotations become negative training examples. It makes a single-class `animal` YOLO dataset, symlinks private images, and creates a visual `box_review.jpg` plus `box_review.csv`.

Run from the repository root in the same Colab or local workspace that contains the Phase 1 `handoff/` and `data/` folders:

```bash
python -m phase2.detection prepare \
  --release handoff/dataset_release.json \
  --output runs/phase2_detection
```

Review the generated contact sheet and fill `reviewer` and `boxes_correct` (`yes` or `no`) for each row of `runs/phase2_detection/box_review.csv`. If a box is wrong, correct the source annotations in a new frozen release and regenerate the detector dataset; the training gate will reject any `no` row.

Install the Phase 2 packages in a GPU runtime, then fine-tune the detector:

```bash
pip install -r phase2/requirements.txt
python -m phase2.detection train --dataset runs/phase2_detection --device 0
```

The detector uses `train` for weight updates and `val_select` for training-time checkpoint selection. `val_calibrate` is reserved for confidence/NMS operating-point selection; the locked `test` split is evaluated only after that policy is fixed. The current code prepares data and starts fine-tuning. Threshold sweep, test metrics and galleries are not implemented yet. No detector result is claimed in this repository.

For the next milestones, produce visually checked SAM-prompted masks on a documented subset, fine-tune a segmentation model, and evaluate against independently verified ground-truth masks. Then package the Phase 1 classifier, detector and segmenter into a batch pipeline with event-level counts, timestamps and a review gallery. Export the principal model to ONNX, check numerical parity, expose an HTTP endpoint, benchmark with warm-up on the served hardware, and document go/no-go criteria and failures. Record the actual resolved package versions and W&B run links when those experiments execute.

The `runs/` and `data/` directories are private and ignored by Git. Do not upload camera-trap images, team photos, model weights or private locations to the repository.
