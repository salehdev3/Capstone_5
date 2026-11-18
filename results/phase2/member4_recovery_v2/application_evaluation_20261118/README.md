# Integrated application development evaluation

The uploaded acceptance CSV and decision JSON agree on all seven metrics, values, unchanged thresholds, directions and outcomes. Four criteria pass and three fail. Decision: **NO-GO for unattended deployment against these criteria**. This is not an assessment of course submission completeness.

| Metric | Measured | Target | Outcome |
|---|---:|---:|---|
| Frame animal recall | 97.60% | >=99% | Fail |
| Empty-frame removal | 52.83% | >=20% | Pass |
| Mask foreground-union IoU | 73.99% | >=70% | Pass |
| Event-count MAE | 0.31944 | <=1 | Pass |
| Full-pipeline CPU p95 | 9.07134 s | <=2 s | Fail |
| Species macro-F1 | 0.48252 | >=0.65 | Fail |
| Species ECE | 0.05128 | <=0.10 | Pass |

Count evaluation covers 288 fully supervised events over retained frames, excluding 47 events with incomplete positive box supervision. The detector identity matches the released M4 model and original confidence 0.045 / NMS 0.3. Count is maximum simultaneous detections over retained frames, not unique individuals or an entire original source sequence.

The CPU benchmark has p50 0.48264 s, p95 9.07134 s, p99 9.08620 s, throughput 0.28817 images/s, 3 warmups and 20 sequential replays. It is a fixed validation branch-coverage mixture with fast empty/missed frames and slower detected-animal frames; median speed must not be presented as latency for every animal frame. Image read and all inference stages are included; HTTP transport is excluded. Repeated detector execution within Recovery v2 is included. The 2354.72 MB RSS is cumulative process peak, not incremental per-model memory.

The scopes differ across metrics and are explicitly preserved in acceptance.csv. Segmentation is a 40-image exposed development test; the retrained species model is evaluated on historical Phase-1 test crops. These results do not establish independent end-to-end generalization.

Use the current system as a research prototype with human review. Remaining improvements are animal recall, rare-class species recognition and full-pipeline CPU latency; do not change acceptance thresholds to make the existing decision pass.

Verification covers cross-file consistency and recomputation of all seven threshold comparisons. Raw event-count rows and raw timing samples were not supplied in this upload, so event MAE and latency quantiles were not independently recalculated here. The first HTTP attempt is preserved separately; full-stage HTTP execution was later reported by the user, and its updated evidence should also be included in the final audit.
