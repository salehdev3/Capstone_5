# Detector ONNX decision and metric parity

Run `notebooks/Member4_ONNX_Decision_Parity.ipynb` in Colab after mounting the project's Drive. CPU is sufficient. It uses the fixed detector, confidence .045 and NMS IoU .30; no training, threshold tuning or historical test rerun.

The full evaluation uses all **1,208 retained selection-validation frames**, including 91 empty images. Images are checked against the frozen manifest's decoded-pixel hashes. Both backends receive the identical square FP32 1x3x640x640 tensor and use identical external NMS. This matches a fixed-shape export contract (`rect=False`), and does not establish equality to a native runtime that uses different rectangular padding.

Predeclared tolerances: same detection counts/classes, matched IoU >= .999, maximum box-coordinate error <= .1 input pixels, confidence error <= .0001, and absolute operating-metric differences <= 1e-6. Raw tensors also use rtol=atol=.001. Hungarian matching associates backend outputs; confidence-ordered greedy IoU >= .5 matching measures detection precision/recall/F1 against the frozen references. Frame animal recall and empty removal are also compared. AP is not calculated by this tool.

Outputs are saved in a new timestamped Drive directory under `phase2_results/member4_recovery_v2/onnx_decision_parity/`. Existing evaluations are preserved. A parity failure is retained and reported; thresholds are never adjusted to obtain a pass. This compares the detector only; classifier, M5 and SAM remain PyTorch.

`local_validation_40/`, when present, contains an initial run on the 40 available reviewed-mask validation images, a positive-only subset. It cannot measure empty-frame removal or replace full validation. The remaining source images could not be downloaded in the assistant environment because both public storage endpoints returned HTTP 403. Use the full Colab run for the complete evaluation evidence.

## Full Colab result supplied on 19 November 2026

The user's execution output reports PASS: 1,208/1,208 frames, 1,626/1,626 matched detections, no decision mismatches and zero differences in precision, recall, F1, animal recall and empty removal. Both backends produce TP/FP/FN 1027/599/97. Maximum box error is .000518798828125 input pixels; confidence error is 4.5299530029296875e-06. Raw maximum absolute error .00946044921875 passes the combined relative/absolute tolerance; it is not an absolute-only .001 check.

`full_validation_20261119T070317993000Z/` contains the four original uploaded outputs, SHA256 checksums and `verification.json`. Cross-file checks confirm all 1,208 frozen validation IDs exactly once, reference counts and labels, equal per-image backend decisions, reconstructed operating metrics and matching detector/ONNX hashes with the prior raw-tensor export evidence. This is consistency verification, not an inference rerun. The earlier `full_validation_reported.json` is retained as a labelled transcription and points to the original exports. These validation parity values use fixed square preprocessing and do not replace the earlier detector test/acceptance metrics.
