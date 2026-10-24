# Automated Triage of Camera-Trap Imagery

Project 5 — AI251P Applied Computer Vision. Phase 1: data preparation (M1), CNN classification (M2), transfer learning and experiment tracking (M3).

## Project files

- [Unified presentation](notebooks/Phase1_Final_Presentation.ipynb): data preparation, scratch CNN, frozen/full ResNet-50, evaluation and reporting in one session.
- [Python export](Phase1_Final_Presentation.py): executable code from the unified notebook.
- [Requirements mapping](docs/REQUIREMENTS.md): deliverables and evaluation criteria.
- [Dependencies](requirements.in): package declarations; each execution records installed versions.
- `preprocessing.py`: shared preprocessing reference checked by tests.
- `tools/build_presentation.py`: refreshes the Python export and source snapshot from the unified notebook.
- `tests/`: CPU synthetic validation.

## Execution

[Open in Colab](https://colab.research.google.com/github/salehdev3/Capstone_5/blob/main/notebooks/Phase1_Final_Presentation.ipynb).

Select a GPU with compatible PyTorch and torchvision. Execute the four sections in order in the same session. The notebook downloads a CCT20 subset, requests label and blur reviews, freezes the dataset release, runs six classification experiments and assembles the report. No archive transfer between members is required.

Both tasks share the dataset release: empty/animal classification on full frames and species classification on human-box crops. Cameras and bursts remain disjoint across train, val_select, val_calibrate and test. Select on val_select and calibrate on val_calibrate before evaluating locked models on test.

W&B defaults offline. Sync actual logs and record run links. Complete team observations, independent reviews, error analysis and replay evidence before presenting the report as complete. Keep raw images, weights and private galleries outside Git.

## Validation

After editing the notebook, run `python tools/build_presentation.py`, then `MPLBACKEND=Agg python tests/smoke_presentation.py` with the project dependencies installed.

Validation exercises synthetic scratch training, calibration, evaluation, checkpoint reload, random-weight ResNet gradient paths and report packaging. CCT20 training, performance results and human review remain required for submission.
