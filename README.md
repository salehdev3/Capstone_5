# Automated Triage of Camera-Trap Imagery

Project 5 — AI251P Applied Computer Vision. Phase 1: data preparation (M1), CNN classification (M2), transfer learning and experiment tracking (M3).

Phase 2 development is on branch `v2`. See [Phase 2 execution and status](phase2/README.md), the [official Phase 2 report source](docs/phase2_report/official_template/main.tex), and the [Phase 2 model card](docs/phase2_model_card.md). Phase 1 artifacts remain available in `v1`.

## Project files

- [Unified presentation](notebooks/Phase1_Final_Presentation.ipynb): data preparation, scratch CNN, frozen/full ResNet-50, evaluation and reporting in one session.
- [Python export](Phase1_Final_Presentation.py): executable code from the unified notebook.
- [Requirements mapping](docs/REQUIREMENTS.md): deliverables and evaluation criteria.
- [Dependencies](requirements.in): package declarations; each execution records installed versions.
- `preprocessing.py`: the single implementation imported by data preparation, training and checkpoint inference.
- `tools/build_presentation.py`: refreshes the Python export and source snapshot from the unified notebook.
- `tests/`: CPU synthetic validation.

## Execution

[Open in Colab](https://colab.research.google.com/github/salehdev3/Capstone_5/blob/v1/notebooks/Phase1_Final_Presentation.ipynb).

Select a GPU with compatible PyTorch and torchvision. Execute the four sections in order in the same session. The notebook downloads a CCT20 subset, requests label and blur reviews, freezes the dataset release, runs six classification experiments and assembles evidence for the report. No archive transfer between members is required.

The setup cell loads `preprocessing.py` from the local repository. When opened alone in Colab, it downloads a pinned copy from this repository and verifies its SHA-256 before importing it. Checkpoints record the actual module hash; evaluation rejects a different preprocessing version. The submission package includes the module.

Both tasks share the dataset release: empty/animal classification on full frames and species classification on human-box crops. Cameras and bursts remain disjoint across train, val_select, val_calibrate and test. Select on val_select and calibrate on val_calibrate before evaluating locked models on test.

The W&B Account Connection cell reads WANDB_API_KEY from Colab Secrets and enables online tracking before configuration freeze. Sync any older offline logs and record run links. Complete team observations, independent reviews, error analysis and replay evidence before presenting the report as complete. Keep raw images, weights and private galleries outside Git.

## Validation

After editing the notebook, run `python tools/build_presentation.py`, then `MPLBACKEND=Agg python tests/smoke_presentation.py` with the project dependencies installed.

Validation exercises synthetic scratch training, calibration, evaluation, checkpoint reload, random-weight ResNet gradient paths and report packaging. CCT20 training, performance results and human review remain required for submission.

## Phase 1 scope and submission

Project 5, Section 8.5, page 19 of the updated guidelines explicitly requires: “Classify frames as empty or non-empty and classify species on crops.” Both tasks are retained. Human-annotated boxes supply the crops; detector training, segmentation, ONNX export and HTTP deployment are Phase 2 work and are not implemented here. `triage` is the name of the required empty/animal task, not a trial execution mode.

The image-quality review includes editable reviewer, decision and observation fields. Saved entries reload on reopening the cell. `keep` records acceptable image quality; `flag` records a quality concern for discussion. A blur flag alone does not remove an image. The label audit remains a separate required review. Reviews cannot be changed after the dataset release is frozen.

Submit one PDF per team, compiled from the official Overleaf template and named `TeamName_Phase1.pdf`. Complete every section and table, replace grey prompts and put figures in `figures/`. Section 1 must contain accessible repository, W&B project and Dataset Card links. Member 4 records these links in `report_inputs.json`. Every reported figure must identify its split, input resolution and threshold (or argmax / not applicable).

The notebook generates `REPORT_DRAFT.md` and an evidence archive, not the completed Overleaf PDF. Publish the generated Dataset Card and sync offline W&B runs before checking instructor access. Complete the report with actual experiment results and team analysis. The grading allocation is M1: 12, M2–M3: 20, reproducibility: 8 points.

Run `python tests/test_blur_review.py` to validate review persistence and frozen-release protection.

## Current team-data training protocol (v2)

The notebook now creates `CameraTrap_Phase1_Team_Mixed_v2`. Team data enter before shared duplicate checks and the training label audit. The confirmed city defines the split: 14 Riyadh images train; 5 Tabuk images are held out, with image 1 removed from the collection, leaving 5 before cleaning. All comparison models must be retrained with the new frozen release. CCT20 supplies validation and calibration. The retained Tabuk subset is reported separately and is also included in combined test metrics; do not count it twice.

Member 1 first looks for the private `data/team_collected_riyadh` folder locally. If absent, it downloads the folder from the private v1 branch using a GitHub token stored in Colab Secrets as `GITHUB_TOKEN` with notebook access enabled. The token needs read access to this private repository; it is not printed or stored in the project output. For the separate local report, set `LOCAL_DATA_ROOT` to the same folder. The default paths expect the data folder beside the shared preprocessing module. Local metadata, provenance, exclusions and category mapping are frozen with the release. Pigeon and chicken map to bird, with original species preserved. The site IDs for local frames represent conservative city groups, not actual camera counts. This small, imbalanced sample is not representative; bonus credit remains the instructor's assessment.

## Training time budget

New dataset configurations use `max_training_seconds=900`, GPU mixed precision and two data-loader workers on CUDA. CPU training is blocked for the presentation configuration; synthetic validation may still use CPU. Set a GPU runtime before starting. The timer starts after model setup and W&B initialization and is checked between training batches. When it expires, the current partial epoch is validated, the best checkpoint is retained, and training ends. History records elapsed time, sample count and partial epochs. Validation already in progress, checkpoint writing, calibration and final test evaluation can extend wall time beyond 15 minutes; this is not a strict end-to-end deadline. Each of the six experiments has its own budget. Shortened runs may have lower accuracy. Runtime policy is included in the checkpoint cache signature.

Existing frozen dataset releases retain their configuration. For a run that has not started, create the frozen release using the updated notebook. Do not modify a release while training is active or mix old and time-budgeted results.
