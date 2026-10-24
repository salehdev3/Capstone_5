# Project 5 — Automated Triage of Camera-Trap Imagery

The consolidated Phase 1 presentation is [Phase1_Final_Presentation.ipynb](notebooks/Phase1_Final_Presentation.ipynb), with a matching [Python cell export](Phase1_Final_Presentation.py). It runs data preparation, scratch training, transfer learning and reporting in one workspace. Human audit gates remain interactive; execute on real data before presenting project results. The original role notebooks remain available for individual work.

Rebuild the consolidated files with `python tools/build_presentation.py`. Validate their actual cells with `python tests/smoke_presentation.py` in the prepared CPU environment. This validates synthetic execution, not a completed CCT20 experiment.

**Phase 1: M1–M3 only.** This is the implementation package, not a claim that training or human review has finished. Begin with [START_HERE_AR.md](START_HERE_AR.md).

| Member | Notebook | Responsibility |
| --- | --- | --- |
| 1 | `notebooks/Phase1_Member_1_Data.ipynb` | CCT20 preparation, stratified camera split, auditing, Dataset Card |
| 2 | `notebooks/Phase1_Member_2_CNN.ipynb` | Scratch CNN, two classification tasks |
| 3 | `notebooks/Phase1_Member_3_TransferLearning.ipynb` | Frozen and full ResNet-50 fine-tuning, both tasks |
| 4 | `notebooks/Phase1_Member_4_Report.ipynb` | Comparison, independent checks, report and submission repository |

Run **1 → (2 and 3) → 4**. All notebooks are self-contained with visible Python cells, and do not require Drive or `preprocessing.py`; that file is an exact shared-code reference. The two tasks are full-frame empty/animal classification and species classification using human-annotated boxes. No detector is trained in Phase 1.

Use a GPU Colab runtime for training. Install cells expect Colab's existing compatible PyTorch/torchvision. Run Member1 from a fresh session; it downloads CCT20 annotations and an approximately 6,000-frame subset. Internet is required for data and pretrained ResNet weights. The subset size, 12-epoch cap and acceptance targets are team defaults, not instructor-prescribed values. Review them before freezing Member1's release. Human label and blur reviews and Dataset Card observations are required; the notebook stops if incomplete.

Download Member1's private ZIP. In new sessions, upload it using Colab Files and set `ARCHIVES` to its absolute path in Member2/3. An empty list reuses an existing workspace. Member2/3 produce private result ZIPs; Member4 imports all three. Do not unzip archives into arbitrary folders: the provided import cell creates the expected paths and checks conflicts. Do not use old Phase1 archives with this schema-2 release.

Keep the same dataset and training config for all six experiments. Train uses augmentation, val_select selects checkpoints, val_calibrate fits temperature and the triage threshold, and test measures the locked models. Results must not be used to tune and rerun on the same test set. Unknown held-out species are reported separately; top-k with k equal to class count is uninformative and explicitly identified.

W&B defaults to offline logging. Sync the logs or use online mode with your own account, and record the resulting run links. No credentials belong in notebooks. Each role records its actual `pip freeze` and runtime after installation. The included `requirements.in` lists dependencies; it is not an executed Colab lockfile.

Member4 requires the four source notebooks under the workspace's `notebooks/`, genuine review notes, team names and repository URL. It exports a report and repository containing code and selected numerical evidence, excluding raw photos, checkpoints and private error galleries. Keep executed notebooks, model weights and private archives separately for course assessment. Check for identifying information before sharing your report.

See `docs/LAB_MAPPING.md`, `docs/REQUIREMENTS.md` and `docs/VALIDATION.md`. The `tests/` CPU smoke test uses disposable synthetic data and makes no CCT20 performance claim.

## Open in Google Colab

| Member | Colab notebook |
| --- | --- |
| 1 — Data | [Open Member 1](https://colab.research.google.com/github/salehdev3/Capstone_5/blob/main/notebooks/Phase1_Member_1_Data.ipynb) |
| 2 — CNN | [Open Member 2](https://colab.research.google.com/github/salehdev3/Capstone_5/blob/main/notebooks/Phase1_Member_2_CNN.ipynb) |
| 3 — Transfer learning | [Open Member 3](https://colab.research.google.com/github/salehdev3/Capstone_5/blob/main/notebooks/Phase1_Member_3_TransferLearning.ipynb) |
| 4 — Report | [Open Member 4](https://colab.research.google.com/github/salehdev3/Capstone_5/blob/main/notebooks/Phase1_Member_4_Report.ipynb) |

Repository: https://github.com/salehdev3/Capstone_5

Opening a notebook does not execute it or provide the private datasets/result archives. Save your own executed copy in Colab. The repository contains source and local smoke-test documentation; actual CCT20 training, human review and the final report remain to be completed by the team.
