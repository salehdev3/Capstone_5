# Measurement and release sequence

1. Preserve baseline v2 report, all original metrics and checksum inventory under docs/final_review. V3 writes only optimization_v3 results and local cache.
2. Prepare and verify pixels, grouped splits, class support and mask geometry. Any semantic reference correction needs a named reviewer and new reference version. Preparation records geometric findings without automatic changes. Current mask population remains 150 train / 40 validation; small data can limit gains.
3. Train detector and segmenter with moderate augmentation, validation early stopping and periodic checkpoints. Train ResNet's last residual stage with class-weighted loss, fixed earlier stages, frozen BN running statistics and early stopping by macro-F1. Resume an interrupted run with unchanged sources/configuration.
4. Compare baseline and candidate detector settings on all 1,208 retained validation images. Maximize empty removal subject to recall >= .99 and empty removal >= .20. If no configuration is feasible, retain a diagnostic choice and do not call it acceptance success. The 90% stretch goals are reported separately.
5. Compare baseline standalone masks and candidate best/last weights at predetermined thresholds on the same 40 original validation references. Report direct-mask IoU and baseline full M4+M5+SAM .823969663 alongside it. Quality loss is not a speed improvement alone. Inspect overlays/reference quality before deciding whether a direct model should replace SAM.
6. Freeze selected model hashes and classifier calibration; run real local HTTP requests, overlay generation and full CPU benchmark in the supplied final cell. A GPU training run is not a CPU latency benchmark. Raw model accuracy is not end-to-end crop classification accuracy.
7. For a new independent score, collect separately reviewed camera/event-disjoint images with boxes, species and masks. Existing test data have already been inspected and remain historical/development evidence. Any fixed-model historical rerun is labelled accordingly and cannot justify a new untouched-test claim.
8. After actual results: export the selected detector with fresh weight-specific ONNX checks, evaluate event counts/rare species/size bins, review failure cases, apply fixed acceptance targets, rebuild model card/report/slides and archive the new package. The existing audit remains authoritative for v2; v3 is not automatically accepted by changing paths or labels.

## Fitting diagnostics

Species history records full validation macro-F1 and accuracy, unaugmented fixed training-subset accuracy and their gap. Detection/segmentation selection adds unaugmented training-subset measurements. Low training and validation scores can indicate capacity, optimization or label problems; a widening gap suggests generalization problems. These diagnostics require context and do not certify absence of overfitting. Repeated selection can overfit validation even with early stopping.

## Execution

Run `notebooks/Phase2_Optimization_V3.ipynb` in Colab with GPU. Start at setup/prepare after a runtime reset, then rerun the interrupted stage. Persisted completed stages return without retraining. Training downloads official YOLO11s pretrained weights; classifier recovery weights are loaded from Drive. No claim of new metrics is made until these stages have actually executed.
