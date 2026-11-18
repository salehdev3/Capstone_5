# Species classifier recovery for Phase 2

Retrains a separately versioned frozen Microsoft ResNet-50 classifier using the exact Phase-1 camera/event split and 14-class vocabulary. The original Phase-1 checkpoint is unavailable. This workflow does not manufacture old weights, reuse old results for a new model, or modify the submitted report.

The metadata ZIP contains only the original manifest, vocabulary/configuration and preprocessing source; no photographs or model weights. The manifest SHA256 is `d3bfb6ba0c9d31412f9d603a6f63ae7c4890f458c5d2aa2a87d7653bb882bb53`.

Recipe: input 224, seed 42, frozen pretrained ResNet-50 backbone in eval mode, Adam lr 0.001, batch 32, up to 12 epochs, patience 4, no class weighting/label smoothing/weight decay, original flip and color-jitter augmentation. Backbone weights are downloaded from the public `microsoft/resnet-50` checkpoint.

Differences from the historical run: a new execution/library/hardware environment, zero DataLoader workers for resumption, and no 900-second cutoff so CPU training can finish complete epochs. Original metrics are not guaranteed. Partial epochs are rerun from the last completed epoch after an interruption. GPU/CPU switches can change numerical results.

Training uses 2,819 crops, checkpoint selection 1,124 validation crops, and temperature calibration 483 separate calibration crops. Test uses 508 known-class crops and is run only after model and policy hashes are fixed. Historical test data has been reported before, so the new result is labelled historical evaluation, not fresh independent testing.

Output: `phase2_results/species_recovery_v1/` contains best/last checkpoints, history, image hashes, calibration, reload verification and measured results. Completed checkpoints are installed into `models/species_recovery_v1/`. Missing default `models/species/` aliases are filled; differing existing files are preserved. No detector, segmenter or submitted Phase-1 results are changed.

Use the accompanying Colab notebook to install frozen metadata, train/calibrate, evaluate the historical test and initialize Member 4 with the new model. Training writes a resumable `last.pt` after every complete epoch and a selected `best.pt` on validation improvement. Checkpoints remain in Drive, not Git. The optional W&B backup cell explicitly uploads the actual new model as an artifact.

Local verification covers frozen metadata/group disjointness, safe RNG checkpoint roundtrip, closed-set metrics/calibration and strict ResNet-50 state-dict compatibility with the deployment architecture on synthetic inputs. Full pretrained training was not executed in the assistant workspace: complete training images reside in the user's Drive. No recovery accuracy has been measured yet.
