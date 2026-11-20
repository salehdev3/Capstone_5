# V4 mask improvement on approved training updates

Uses the actual merged 70-review override package, but applies **56 training updates only**. All 40 validation reference files remain byte-identical to the previous release for selection and model comparison. The 14 approved validation updates remain available separately for a future versioned reference assessment. One unresolved original validation reference is tracked and retained; it is not excluded to increase the score. Other training references retain the baseline copy, including the six previously documented training-only removals.

Warm-start the actually trained V3 YOLO11s-seg checkpoint (SHA pinned) with AdamW lr .0001, 40 epochs maximum, patience 8, no mosaic/mixup, moderate augmentation. Do not alter deployed models automatically. GPU is required by training/comparison; CPU is sufficient for preparation.

Compare V3, V4 best and V4 last direct-mask outputs on the same original validation references. Then rerun the old full M4+M5+SAM-B policy and compare four M4+new-M5+SAM2.1-t settings (640/1024 input, quality gate 0/.75). No human boxes/reference masks are used as inference prompts: the original helper uses detector boxes and learned-mask interior points. Human reference masks are used only for scoring. Early selection is development evidence; no new test is run.

A 90% IoU target is recorded and reported from actual output, never guaranteed. The selected pipeline remains experimental until fresh classifier/count/HTTP/CPU-latency/ONNX/acceptance checks. GPU mask-helper timing is not full CPU pipeline latency. Dataset changes do not create new classifier samples; the separate collection/review workflow is still required for species coverage.

Run notebooks/Phase2_Mask_Improvement_V4.ipynb. Review package excludes learned weights and original photographs; save actual trained weights in Drive and back them up separately. Code compiles and preparation is tested with the actual 56 approved updates and all unchanged validation labels; GPU training and comparison have not executed in the assistant environment.
