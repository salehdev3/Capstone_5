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

## Application evaluation

`tools/member4_evaluation_colab.py` loads the initialized pipeline and updated demo. Its module binds classifier and segmentation metrics to the actual installed model/policy hashes, recomputes fixed-detector counts, and uses unchanged original acceptance targets (including the Phase-1 species F1/ECE criteria). It does not read old generic segmentation_metrics.json or rely on stale in-memory metrics/test_records.

Count inference uses 554 retained supervised test frames; 288 events have complete supervision over every retained frame, and 47 events are excluded. Counts use the same unchanged detector path as the pipeline, independent of classification/mask filtering, avoiding unnecessary classifier/SAM execution across the full count set. Empty-event errors remain included. Input image hashes and detector-policy identity are cached; source pixels must match the frozen manifest.

Full-pipeline timing separately executes three warmups and 20 validation-case replays, includes all inference stages (including repeated M4 within masks), and excludes HTTP transport. The fixed functional-case mixture is not representative production traffic, and 20-sample p95/p99 are descriptive. Memory is cumulative process peak RSS.

Outputs are timestamped under member4_recovery_v2/evaluations. Legacy report filenames point to the current evaluation. NO-GO indicates unmet preset deployment performance criteria, not a course-completion grade. Local event logic checks passed on synthetic cases and frozen metadata; actual model scores/timings were not generated in the assistant workspace.

## Version-aware evidence audit

`tools/member4_audit_colab.py` checks installed model identities, current application and full-stage HTTP evidence, the 230-mask review/reference release, current-version failure-review fields, export parity, history and historical test records. It creates a separate v2 failure-review CSV with blank human-review fields; old v1 review is not silently transferred. Missing report, presentation, video and uploaded W&B link remain explicit incomplete items. File presence is not content approval.

When available, accepted-policy predictions and frozen mask references are evaluated with COCO instance-mask AP. This is AP of the deployed accepted outputs, including confidence/quality/fallback filtering, not a low-floor native model AP sweep. No test images are rerun or used to change policy. An unavailable AP result remains a missing item, not a fabricated score.

The curated Drive ZIP includes actual learned detector/M5/species checkpoints, their policies, available evidence, demo overlays and package checksums. The public pretrained SAM weight is referenced by official URL/checksum rather than duplicated. No raw dataset, credentials or arbitrary old-run contents are archived. Local packaging does not depend on W&B success; a separate explicit upload function backs up the completed ZIP. Rerun the audit after resolving missing items to produce a fresh timestamped package.

Local checks used synthetic mask fixtures and an intentionally incomplete filesystem: reference-bound AP, honest incomplete status, preserved missing reviews, checkpoint inclusion and checksum-verified ZIP creation all passed. Actual user Drive audit and final mask AP still require execution in Colab.
