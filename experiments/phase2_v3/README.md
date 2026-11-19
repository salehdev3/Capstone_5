# Phase 2 optimization v3

GPU experiments with preserved baseline, fixed camera/event splits and validation-only selection. Accuracy > .90, macro-F1 > .90, detector AP50 > .90 and mask IoU > .90 are aspirational evaluation goals, not manufactured stopping conditions or guarantees. Existing acceptance thresholds remain recorded separately. Zero overfitting cannot be certified from finite data.

Run the Colab notebook in order. Preparation verifies original pixels, builds local detection/segmentation datasets and audits mask geometry. Model training saves resumable checkpoints to Drive. Classification starts from the recovered ResNet-50 and fine-tunes the last residual stage with class-balanced loss. Detection and segmentation use larger pretrained YOLO11s models. Early stopping, moderate augmentation and held-out selection constrain training. Calibration uses val_calibrate only. All measured outputs are stored under phase2_results/optimization_v3.

The standalone segmentation candidate removes the second detector/SAM pass, but may sacrifice mask quality. Selection records overlap against the existing standalone model and the archived full-pipeline validation baseline. No new weights replace the deployed release automatically. Benchmark new candidates with their quality measurements before deciding on deployment.

The script does not alter the exposed historical test or use it to select models. Fresh independent camera/event data are required for an independent >90% claim. Historical comparisons, full ONNX parity, HTTP fresh-runtime replay, event counts, report/presentation and acceptance audit must be rerun with the selected release before it replaces v2. See ROADMAP.md. A training run alone does not close these deliverables.

The training mask copy reuses the six documented training-only removals from Recovery v2 and checks both original and derived hashes. Validation masks are copied unchanged. Other geometric findings are recorded for inspection, not silently repaired.
