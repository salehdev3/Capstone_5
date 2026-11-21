# Automated Triage of Camera-Trap Imagery

AI251P · Project 5 · Phase 1 and Phase 2

| File | Use |
| --- | --- |
| [Final presentation](presentation/Final_Presentation.pdf) | Slides for a 15–20 minute presentation including the live demonstration |
| [Official project notebook](https://colab.research.google.com/github/salehdev3/Capstone_5/blob/project/official_file_project.ipynb) | Phase 1 classification, Phase 2 detection and segmentation, final evaluation and demonstration |
| [Live demo](https://colab.research.google.com/github/salehdev3/Capstone_5/blob/project/demo.ipynb) | Run a real HTTP request with the frozen CPU pipeline |
| [Phase 1 report](reports/Phase1_Report.pdf) | Original submitted report |
| [Phase 2 report](reports/Phase2_Report.pdf) · [Overleaf source](reports/Phase2_Report_Overleaf.zip) | Final integrated results, component experiments, failure cases and model card |

## Running the demonstration

Open the demo in a fresh Colab runtime. Mount the team Google Drive account with the full `MyDrive/CameraTrap_Phase2` project folder. The evaluation policy and its matching checkpoints live in that Drive. The loader verifies the policy and model digests. Run both cells in order. The default **cct_test** demonstration sends the first 15 CCT20 test frames by image ID. Select **team_test** to run the team-collected test images. Set **DEMO_INPUT** to **photos** for personal images or **video** for one clip. Video mode samples up to five frames at one-second intervals as one event. Each uploaded photo becomes a separate event.

The response contains frame detections, species scores, segmentation polygons when available, event counts and stage timing. Each run writes a new directory under `CameraTrap_Phase2/presentation_demos/` containing the HTTP response, annotated frames and **Demo_Fallback.mp4**. Run the demonstration once before presenting to save the fallback in Drive.

The official notebook defaults to presentation mode. Optional training and evaluation switches require the original datasets, annotations and Drive run dependencies. Phase 2 component training replay uses a separate `CameraTrap_Phase2_training_replay` directory. Phase 1 training uses its original notebook workflow.

## Evaluated release

`phase2_results/integrated_species_cv/20261120T224149142114Z`

The four original acceptance criteria passed on the historical data. The guided mask policy reached mean IoU 0.662 on 40 historical images against a target of 0.70. The integrated full-pipeline CPU p95 was 1.879 s. See the Phase 2 report for operating points, metric definitions and cohort sizes. The detector export parity evidence applies to the documented component operating point.

## Presentation timing

| Topic | Speaker | Minutes |
| --- | --- | --- |
| Problem, users and data | Saud | 4 |
| Phase 1 comparison and improvement plan | Saud | 2 |
| Phase 2 detection | Saleh | 3 |
| Phase 2 segmentation | Yousef | 3 |
| Live demonstration | Mohannad | 2–3 |
| Benchmarks and decision | Mohannad | 2–3 |
