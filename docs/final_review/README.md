# Verified project evidence

Review of the current measured baseline before further optimization. No results or acceptance thresholds were changed.

| Area | Verified evidence |
| --- | --- |
| Frozen grouped dataset | 6,037 frozen frames; camera/event/decoded-pixel split separation checked |
| Detector training and operating-point evidence | 40 training epochs, 15 archived threshold combinations and native low-floor test AP |
| Separate Small/Medium/Large recall | Small/Medium/Large populations 5/154/349; matched totals reconstruct 450/508 detections from fixed cached outputs |
| Human annotation review records | 100 animal-box image reviews and 230 reference-mask reviews with named reviewers |
| Segmentation evaluation and operating-mask AP | Recovery-v2 development evaluation: 40 images; IoU 0.739877; accepted-output mask AP archived |
| Reviewed detector and mask examples | 12 reviewed detector examples and 12 reviewed Recovery-v2 mask examples |
| New classifier calibration/reload and historical evaluation | New recovery classifier: 12 epochs; 508 historical crops; calibration and reload evidence bound to checkpoint |
| Executed full-stage HTTP batch | HTTP 200 on 5 fixed validation cases; detection, classification and matched masks exercised |
| Event-count application measure | 288 supervised retained events; per-event errors reconstruct archived count MAE |
| Warmup and full-pipeline benchmark | 20 full-pipeline timings after 3 warmups; p50/p95/p99 and throughput reconstructed |
| Full validation raw/decision/operating-metric parity | 1,208 validation frames; 1,626 matched detections; original exports/hash identities consistent |
| Fixed criteria and documented decision | All 7 fixed criterion calculations agree with the documented decision; values preserved |
| Compiled English report and matching source | English template report PDF, 13 referenced figure files and matching editable source archive |
| Code/notebook static reproducibility | Notebook cells parse, critical loader hashes match modules, Phase 1 report preserved; runtime replay is a separate task |
| Archived W&B receipt | Committed W&B artifact receipt and run URL archived; no new upload claimed |

The machine-readable register records exact checks and evidence paths. Runtime replay and final human deliverables are tracked separately in `required_actions.json`. Preserve the baseline hashes when starting new optimization experiments. Run `python tools/final_submission_review.py` from the repository to reproduce this review.
