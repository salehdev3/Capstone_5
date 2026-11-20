# Final sprint — one-cell Colab workflow

Run `notebooks/Phase2_Final_Sprint_One_Cell.ipynb` with a GPU and the existing project Drive account. It installs dependencies and performs a bounded, resumable run. A target of above 80% is measured, not assumed; MAE, ECE and latency retain their own units and original directions. Original acceptance criteria and Phase 1 submissions are unchanged.

The workflow verifies the original 6,037-image manifest and decoded pixels; reuses verified trained YOLO candidates; trains one regularized ConvNeXt-Tiny classifier with a background output; calibrates on `val_calibrate`; selects species/triage/mask configurations only on validation; freezes the policy; measures the historical development test; runs detector ONNX parity on all 1,208 retained validation images; measures full CPU/GPU timing; sends a real HTTP request in a fresh interpreter; and generates a measured English report, final slides, actual-response replay, labelled video-sequence demonstration, model card and learned-weight evidence ZIP. It optionally backs up the ZIP to W&B using a Colab secret.

Existing completed classifier checkpoints are compared on validation so that a new classifier is not automatically substituted if it scores worse. The new classifier's background output remains a separately calibrated filtering candidate. Threshold zero disables filtering as a candidate. Detector counts are the accepted boxes after the fixed filter, independently of mask acceptance.

Mask comparison uses the original frozen 40 validation masks, three available trained guides, direct confidence sweeps, SAM-B and SAM2.1-Tiny at 640/1024. The highest-quality configuration in each family is benchmarked on CPU. The selected configuration prefers the original CPU p95 target together with the overlap goal; it does not relabel GPU times as CPU times. Test outcomes never trigger automatic retuning. Original references are not silently rewritten to match predictions.

The frozen validation has only 9 of the 14 classes. All-14 macro-F1 therefore cannot exceed 9/14 on that validation set. The denominator stays fixed. Class support and diagnostic training/validation gaps are reported; neither absent overfitting nor an 80–90% outcome is guaranteed. Existing test sets are historical/development evidence, not independent new tests.

All automatic stages store logs, checkpoints and receipts under `CameraTrap_Phase2/phase2_results/final_sprint_v1`. A Colab disconnect requires rerunning the same cell. Preparation re-verifies/restores the disposable local cache; completed stages are reused. A code identity guard prevents silently mixing different code with an existing version. Missing source files or mismatching hashes stop with a concrete error. W&B is optional and bounded to four minutes so it cannot prevent local packaging.

Report/slides/video and the new automated failure-analysis drafts require team visual review. No new named human approval, independent generalization, live screen recording, natural-video accuracy, or full-stack ONNX export is invented. Artifact presence and unattended-deployment GO/NO-GO are separate. The recorded fallback is explicitly an execution replay of an actual HTTP response.

## Reproduction

Inputs: existing `phase1_source/data/cct20_team_v2/{manifest,species}.json`, all original `images/*.jpg`, original `segmentation/{labels,reference_release.json}`, Recovery-v2 release/helper/weights, and any completed V3/V4 candidate weights. The two larger detector/guide candidates are optional; the original release and SAM-B are required. Public SAM2.1-Tiny and pretrained ConvNeXt weights download from their publishers.

The archive contains the measured source snapshot, learned checkpoints, release hashes, results, report, slides and videos. Source images and public SAM checkpoints remain outside the archive. `release_manifest.json` documents relocation; existing Drive paths are used by the installed release. The source requirements and executed Python/torch/backend versions are recorded. An existing archived report and Phase 1 comparison CSV are historical reference assets, not replacement measurements.
