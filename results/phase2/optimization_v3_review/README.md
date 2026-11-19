# Measured V3 experiment review

Actual user-executed GPU training, validation selection, CPU benchmark and HTTP execution. REVIEW_SUMMARY.json identifies the uploaded ZIP checksum and consistency checks. The large per-NMS caches and training-image mosaics remain in local runs/v3_review; no learned weight binaries were included in the ZIP.

| Validation result | Baseline | New candidate |
| --- | ---: | ---: |
| Detector AP50 | .908999 | .927799 |
| Detector AP50–95 | .523142 | .547386 |
| Species accuracy | .761566 | .768683 |
| Species all-14 macro-F1 | .368544 | .377834 |
| Best standalone mask mean IoU in this run | .606309 | .655042 |
| Full v2 SAM validation IoU | .823970 | Different pipeline |
| CPU functional-case p95 seconds | 9.071338 | 2.085286 |

The triage rule selected the original detector at confidence .05/NMS .30 with square preprocessing. The fast CPU candidate uses this detector plus the new classifier and direct M5 masks. The speed gain trades away SAM mask quality; v2 historical acceptance/test results remain unchanged.

Species selected epoch 4: training-subset accuracy .949219 vs validation .768683. The gap and later trajectory indicate overfitting. Mask train-subset IoU .808803 vs validation .655042 is another fitting diagnostic. These are subset comparisons.

Five species have zero validation references (badger, deer, fox, rodent, skunk). All-14 macro-F1 with the declared zero-F1 convention cannot exceed 9/14=.642857. Supported-class F1 is diagnostic only, never a replacement for the original metric. Training fox has one crop.

Next: add reviewed training class coverage and separate camera/event-disjoint assessment data; inspect the 73 recorded mask geometry findings without changing the old reference release; test improved M5 guidance with SAM in a new version; measure quality and latency together. More epochs alone do not address these findings. New independent data are needed for an independent >90% generalization claim.
