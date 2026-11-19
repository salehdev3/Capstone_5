# Phase 2 v3 measured validation results

| Component | Metric | Value |
| --- | --- | --- |
| Species | Accuracy | 0.768683 |
| Species | Macro-F1 | 0.377834 |
| Detector | Animal-frame recall | 0.991047 |
| Detector | Empty removal | 0.274725 |
| Direct segmentation | Mean IoU | 0.655042 |

Validation-selected experimental candidate; masks are direct learned masks, not Recovery-v2 SAM masks. Fresh deployment evaluation required.

The released v2 report remains the measured deployment baseline until candidate HTTP execution, count/latency evaluation, ONNX parity and acceptance results are available.