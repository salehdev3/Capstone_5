# Dataset Card — Project 5 Phase 1

## Overview

This Phase 1 dataset supports two image-classification tasks: (1) full-frame classification as **empty** or **animal**, and (2) species classification on crops from human-annotated animal boxes. The latter uses existing human boxes; the project does not train a detector in Phase 1.

The source is a selected subset of [Caltech Camera Traps (CCT20)](https://lila.science/datasets/caltech-camera-traps), supplemented by team photographs from Riyadh and Tabuk. The CCT20 subset is largely US wildlife imagery. The local collection is a small convenience sample and should not be treated as representative of Saudi camera-trap deployment.

**Source citation:** Beery, Van Horn and Perona, *Recognition in Terra Incognita*, ECCV 2018. The source is listed under CDLA-Permissive in the project records; users must check the current source terms and the course redistribution policy before reuse. Original photographs and review galleries are kept outside the code repository.

## Dataset construction

| Stage | Images |
|---|---:|
| CCT20 images selected | 6,022 |
| Team photographs included before audit | 19 |
| Combined images downloaded and initially kept | 6,041 |
| Images retained after human-audit quarantine | 6,037 |

The 19 team photographs comprise **14 Riyadh training images** and **5 Tabuk test images**. The first team photograph was removed before this 19-image release. Original pigeon and chicken labels remain in collection metadata; both map to the shared **bird** category for classification. Team metadata and provenance are recorded in `team_metadata.csv` and `team_collection.json`.

All selected images downloaded successfully. Initial cleaning found no corrupt downloads, exact duplicate IDs, or near-duplicate pairs across partitions. Four images were subsequently excluded through the human-audit quarantine; the initial cleaning count of 6,041 and final modeling count of 6,037 refer to different stages. See `cleaning_report.json` and `audit_report.json` for recorded decisions. Images are not discarded solely because of a low Laplacian blur score.

Images are prepared with a maximum side length of 1,024 pixels. Classifiers use **224 × 224 RGB** input with ImageNet normalization. Random flips and brightness/contrast changes are applied to training data only. Shared processing code is in `preprocessing.py`.

## Splits and leakage controls

CCT20 images are split by **camera group** using `StratifiedGroupKFold` with seed **42** and approximate joint image-label stratification. Whole camera events/bursts stay in one split. The team images follow a city protocol: Riyadh supplements training and Tabuk is held out for local testing; both validation sets use CCT20 only. This is a project-specific split, not the official CCT benchmark split.

| Split | Frames | Camera groups |
|---|---:|---:|
| Train | 3,534 | 13 |
| Validation for model selection | 1,337 | 3 |
| Validation for calibration | 527 | 2 |
| Test | 639 | 4 |

The final manifest passed checks for unique IDs and absence of ID, camera, event, and pixel-hash overlap between splits. The full per-class frame frequencies and crop support are in `class_balance_proportions.csv` and `species_crop_support.csv`. Class percentages may sum to more than 100% because a frame can contain multiple categories.

Class frequencies differ substantially between groups. For example, the training split includes 764 raccoon and 727 opossum frame occurrences, whereas fox has one and badger has four. The test split has 53 empty and 586 animal frames. Several rare species have no validation examples. Group separation was prioritized over exact class balance.

## Species crop support

Counts are annotated animal crops; percentages use total crops in each split. A frame may contain multiple crops.

| Species | Train n (%) | Selection n (%) | Calibration n (%) | Test n (%) |
|---|---:|---:|---:|---:|
| Badger | 4 (0.1%) | 0 (0.0%) | 0 (0.0%) | 1 (0.2%) |
| Bird | 116 (4.1%) | 3 (0.3%) | 45 (9.3%) | 9 (1.8%) |
| Bobcat | 286 (10.1%) | 25 (2.2%) | 78 (16.1%) | 57 (11.2%) |
| Cat | 142 (5.0%) | 247 (22.0%) | 35 (7.2%) | 48 (9.4%) |
| Coyote | 288 (10.2%) | 29 (2.6%) | 37 (7.7%) | 169 (33.3%) |
| Deer | 11 (0.4%) | 0 (0.0%) | 0 (0.0%) | 3 (0.6%) |
| Dog | 117 (4.2%) | 33 (2.9%) | 73 (15.1%) | 29 (5.7%) |
| Fox | 1 (0.0%) | 0 (0.0%) | 0 (0.0%) | 1 (0.2%) |
| Opossum | 622 (22.1%) | 489 (43.5%) | 99 (20.5%) | 87 (17.1%) |
| Rabbit | 317 (11.2%) | 164 (14.6%) | 61 (12.6%) | 4 (0.8%) |
| Raccoon | 709 (25.2%) | 14 (1.2%) | 28 (5.8%) | 28 (5.5%) |
| Rodent | 38 (1.3%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| Skunk | 73 (2.6%) | 0 (0.0%) | 21 (4.3%) | 19 (3.7%) |
| Squirrel | 95 (3.4%) | 120 (10.7%) | 6 (1.2%) | 53 (10.4%) |
| **Total crops** | **2,819 (100%)** | **1,124 (100%)** | **483 (100%)** | **508 (100%)** |

## Label audit

One eligible frame was sampled from each of **10 randomly chosen training events**. Reviewers found **1 label error**. The observed error rate is **10%**, and the recorded one-sided 95% upper bound is **39.4%**. The bound assumes independent sampled events; shared cameras may make it optimistic. Flagged training events were quarantined and the original review decisions retained. The sample is too small to establish a precise dataset-wide error rate.

## Intended use and limitations

The dataset supports Phase 1 comparisons between a CNN trained from scratch and frozen/full fine-tuning of a pretrained backbone. Validation selects the models and calibration policy; test data is reserved for final evaluation. Positive frames without human boxes contribute to full-frame triage only. Human/vehicle-labelled, conflicting, or unlabelled frames are excluded by metadata.

The team collection is small, geographically narrow, and uneven in class mix. Its five Tabuk test frames are correlated and provide descriptive local evidence only. Some species are absent or nearly absent from held-out camera groups. Grayscale appearance is only a proxy for infrared or night capture, not verified ground truth. No GPS coordinates are published. Avoid sharing photos with identifiable people without appropriate review and permission.

## Reproducibility references

The evaluated data release has manifest hash `d3bfb6ba0c9d31412f9d603a6f63ae7c4890f458c5d2aa2a87d7653bb882bb53`. The repository records the shared configuration, preprocessing code, split checks, dataset audit, and environment requirements. The full source images are not committed. Test metrics and confidence intervals are produced by the classification notebooks, not by this dataset card.
