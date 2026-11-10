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

Performance targets in the shared configuration are proposed team goals, not promised results. They must be agreed before training. A failed performance target does not mean the experiment failed to execute.

## Completion limits

Reports combine recorded experiment outputs with signed team reviews. The repository includes experiment metrics, figures, a Dataset Card and the Phase 1 report. Model weights and full execution artifacts are stored separately. Human review records and independent training runs provide evidence distinct from automated code validation.

Frozen release files should not be edited after training starts. A data correction requires a new release and retraining all comparison runs; record the issue. Do not tune on test. Unknown species reduce overall coverage and are explicitly counted rather than silently removed from reported scope.

Detection, segmentation, ONNX export and HTTP deployment belong to later work and are outside this package.

## Project-specific scope

Section 8.5, Project 5, page 19, Phase 1 item 1.2 explicitly requires both frame empty/non-empty classification and species classification on crops, with per-class recall on unseen locations and calibration. Crops come from human-annotated boxes in CCT20. These are classification inputs, not detector predictions. Both tasks must remain in the Phase 1 comparison.

## Official report submission

- One team member submits one compiled PDF using the official Overleaf template: `TeamName_Phase1.pdf`.
- Complete every section and table, replace grey prompts, and store figures in `figures/`.
- Section 1 includes working repository, W&B project and published Dataset Card links, accessible to instructors.
- Every figure identifies its data split, model input resolution and decision threshold, or states argmax / not applicable.
- The notebook's Markdown report and ZIP contain evidence for writing the report; they do not replace the required PDF.
- Table 6 allocates 12 points to M1, 20 to M2–M3 and 8 to reproducibility.

## Team-collected data contribution

Updated protocol: Riyadh training supplementation and Tabuk test, split by confirmed city. CCT20 retains validation/calibration. The same cleaning, leakage checks, augmentation and train-only label audit apply to the combined release. Team labels and boxes are separately human-verified by Saleh. Record retained counts, exclusions, collection dates, category mapping, imbalance and limits in the report; retrain all six comparisons. The new workspace preserves the old experiment outputs. Five held-out local images remain before duplicate cleaning after removing image 1 from the collection. The local report is a subset of the combined test, not an additional independent test.
