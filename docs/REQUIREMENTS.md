# Phase 1 requirements traceability

CCT20 is the curated, annotated subset of Caltech Camera Traps used here; it is not the entire 243,100-image collection described in the project example. The metadata summary and Dataset Card record actual downloaded and retained counts.

Source: supplied **Capstone_Project_Guidelines_UPDATED.pdf**, Phase 1 scope in Tables 1–2, M1–M3 deliverables in Table 5, and Project 5. The original four workstreams describe the whole project; this team's four-person split is adapted to Phase 1.

| Requirement | Owner | Produced evidence |
| --- | --- | --- |
| Public Project 5 dataset and licence | 1 | CCT20 metadata/source and Dataset Card |
| Clean corrupt files and duplicates | 1 | cleaning_report.json, manifest, image integrity hashes |
| Leakage-safe stratification | 1 | camera groups, whole events, cross-split duplicate quarantine, site_split.json |
| Class balance per split after cleaning | 1 | class_balance_proportions.csv, species_crop_support.csv, plots |
| Label audit and error upper bound | 1 | human-reviewed label_audit.csv and audit_report.json |
| Blur/quality inspection | 1 | inventory.csv and reviewer-completed blur_review.csv |
| Augmentation data loader | 1–3 | identical OpenCV code, preview batches, train-only augmentation |
| Dataset Card | 1 | runs/data/DATASET_CARD.md with actual observations |
| Full-frame empty/nonempty task | 2–3 | triage runs and conservative removal threshold |
| Species classification | 2–3 | human-box crop runs, class support and unknown-class coverage |
| Classifier from scratch | 2 | ConvNN checkpoints and logged training histories |
| Pretrained frozen and full fine-tuning | 3 | ResNet-50 frozen/full checkpoints and histories |
| Curves and best checkpoint | 2–3 | history.csv, learning_loss.png, learning_accuracy.png, best.pt |
| Accuracy/top-k, precision/recall/F1, confusion matrix | 2–3 | test_metrics.json, confusion.png, per-class output |
| Calibration | 2–3 | temperature fitted on val_calibrate, ECE before/after, reliability.png |
| Test accuracy confidence interval | 2–3 | camera bootstrap plus explicitly labelled independent-sample Wilson |
| Triage operating point | 2–3 | val_calibrate threshold, recall/empty removal, threshold sweep |
| Error gallery and improvement plan | 2–3 | private_errors.jpg if errors exist; manually completed ERROR_ANALYSIS.md |
| W&B all runs | 2–4 | local/online training and locked-test logs; dashboard links after sync |
| Common metric across runs | 4 | comparison.csv; selection from validation macro-F1 |
| Predeclared acceptance criteria | 1,4 | frozen config targets and measured acceptance.csv; failures reported honestly |
| Reproducible code and requirements | All | unified notebook and Python export, per-section environment snapshots, fresh-process check and actual replay evidence |
| Report and repository | 4 | REPORT_DRAFT.md, unified source notebook, selected results and weight hashes |

Performance targets in the starter configuration are proposed team goals, not promised results. They must be agreed before training. A failed performance target does not mean the experiment failed to execute.

## Completion limits

The code creates reports from actual executions and leaves human fields pending. Label audit, independent review, analysis, team contributions, W&B account/link completion and a real replay cannot be automated honestly. No trained weights, CCT20 performance results, completed Dataset Card or final submission report are included in this initial implementation package.

Frozen release files should not be edited after training starts. A data correction requires a new release and retraining all comparison runs; record the issue. Do not tune on test. Unknown species reduce overall coverage and are explicitly counted rather than silently removed from reported scope.

Detection, segmentation, ONNX export and HTTP deployment belong to later work and are outside this package.
