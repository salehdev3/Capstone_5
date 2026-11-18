# Detector ONNX export parity

The assistant executed static FP32 export at input 640, ONNX opset 17, with no NMS, using the M4 checkpoint pinned by Recovery v2. All raw-output checks passed at rtol/atol 0.001 on zero, constant and seeded-random tensors; maximum absolute difference 0.000396728515625. This is assistant CPU evidence, not a measurement of the user's Colab runtime or full-pipeline parity.

The complete segmentation pipeline still includes PyTorch M4/M5 and SAM. Neither a detector-forward benchmark nor this raw tensor comparison establishes end-to-end speed or segmentation equivalence. The original presentation's later integration cells must be updated separately to invoke recovery v2.

## Recovery-v2 integration

`tools/member4_integration_colab.py` initializes PyTorch M4 detection at the original 0.045 floor and NMS 0.3, the Phase-1 frozen/full calibrated species classifier, and the exact released M4 + M5 + SAM Recovery v2 segmentation helper. It preserves the evaluated segmentation policy. Accepted segmentation masks are matched one-to-one to detector boxes. Detection/event counts remain independent of mask filtering; unmatched masks and absent body-area values are explicit. The helper repeats M4 inference and this cost is included in total latency.

Species inference requires the actual Phase-1 checkpoint, its ordered classes and matching temperature policy. Missing checkpoints block readiness; Phase-1 result tables and detector weights cannot substitute. The loader can retrieve a missing calibration policy only when the checkpoint SHA matches one of the original Phase-1 frozen/full models.

Component checks confirm exact Phase-1 preprocessing on three crops, one-to-one mask matching, event maximum counting and readiness blocking when classifier files are absent. Full species and end-to-end inference were not verified locally because the trained classifier is unavailable in the assistant workspace. HTTP service, complete pipeline timing, deployment decision and evidence audit still need execution/update.

## Local HTTP demo

`tools/member4_service_colab.py` uses the initialized pipeline and reads three validation events directly from the frozen manifest, avoiding session-dependent `included`/`IMAGES`. Uses a dynamically bound loopback port, instance-specific health checks and serialized inference. Actual requests, metadata, outputs, overlays and elapsed time are saved under timestamped `member4_recovery_v2/http_demos/` in Drive. Legacy `service_test.json` and `demo_record.json` point to the executed demo. This is not a video recording or a latency benchmark.

Local component checks passed using a synthetic pipeline probe, including real loopback HTTP. Actual neural inference and user-runtime performance remain to be executed in Colab.
