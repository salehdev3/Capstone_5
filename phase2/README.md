# Project 5 — Phase 2

Submission: 22 November 2026.

## Team ownership

| Member | Name | Workstream |
| --- | --- | --- |
| 1 | Saud Abdullah Alfaiz | Data and classification |
| 2 | Saleh Mohammed Alanazi | Detection |
| 3 | Yousef Nafi Alanazi | Segmentation |
| 4 | Mohannad Mohammed Alqahtani | Integration |

## Files and status

- `notebooks/Phase2_Final_Presentation.ipynb`: English presentation notebook.
- `notebooks/Phase2_Final_Presentation_English.ipynb`: identical English copy.
- `notebooks/Phase2_Data_Recovery.ipynb`: initial recovery notebook.
- `phase2/review_ui.py`: persistent 100-image review, one image per page.
- `tools/audit_phase2_inputs.py`: checks recovered Phase 1 workspace identity.
- `results/phase2/member1/`: uploaded recovery and review evidence, original release candidate and verification summary.
- `docs/phase2_report/official_template/`: unfilled official report template.

Member 1 evidence records 6,037 recovered images matching both original file and
pixel hashes. The 100 training reviews are recorded as verified by Saud. Two
observations mention limited animal visibility and should be confirmed before
freezing the training release. The original candidate remains `pending`; the
uploaded source evidence is preserved unchanged.

Detection data: train 3,013; selection 1,208; calibration 467; test 554.
795 incomplete-positive frames are excluded from detection supervision.

Detection, segmentation and integration have not been executed or validated end
to end. Before full training, complete the template-alignment changes: retain all
Phase 1 acceptance criteria, detection error categories and size bins, segmentation
baseline and validation threshold selection, real-image ONNX decision/metric
parity, independent manifest-driven service and deployment-path benchmarking.
Species weights and verified segmentation masks are still required.

Notebook source syntax and deterministic metric checks do not establish training
performance or deployment readiness. No fabricated results are included.
Raw images, weights, uploaded archives and credentials remain outside this change.
