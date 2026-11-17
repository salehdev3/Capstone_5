# Phase 2 input recovery

Submission date: 22 November 2026 (confirmed by the team).

## Historical recovery notes

Update, 17 November 2026: Colab recovery now reports all 6,037 images exactly
matching Phase 1. The 100 training reviews are recorded. See
`results/phase2/member1/` and `phase2/README.md` for current evidence and limits.
The following inventory describes the earlier local recovery stage.

## Initial inventory

The repository contains the Phase 1 report, six experiment summaries, calibration
policies, shared preprocessing and 19 local team images with review metadata.
The uploaded `Phase1_Results_For_Report.zip` restored the original manifest,
split assignments, human boxes, frozen dataset release, calibration policies and
experiment evidence. All 19 files referenced by the dataset release match their
recorded SHA-256 hashes; the manifest matches the evaluated Phase 1 release.
`Phase1_Submission.zip` contains only seven local-evaluation evidence files.
Neither uploaded archive contains image files or model weights.

The report archive is privately extracted at
`runs/phase1_recovered/CameraTrap_Phase1_Team_Mixed_v2/`.
Eighteen team images were reconstructed from local originals using the original
preprocessing recipe and verified against both recorded processed-file MD5 and
decoded-pixel SHA-256. One local image could not be reproduced exactly and was
not installed as a matching recovered file. See
`runs/phase2/local_image_recovery.json` for per-image results.
The original manifest is unchanged; only the known original Colab path prefix is
rebased at lookup time by the audit tool.

Remaining recovery inputs: all 6,018 processed CCT20 images, the one unmatched
processed team image, and the selected triage/species `best.pt` checkpoints.
Both public CCT20 download mirrors returned a proxy tunnel HTTP 403 from this
environment, so direct image recovery could not proceed here.

Detection candidates from the frozen manifest:

| Split | Boxed positives | Empty | Positives needing box review |
| --- | ---: | ---: | ---: |
| train | 2,694 | 319 | 521 |
| val_select | 1,117 | 91 | 129 |
| val_calibrate | 454 | 13 | 60 |
| test | 501 | 53 | 85 |

These counts describe metadata candidates, not visual verification of completeness.
Preserve all original split assignments.

Restore the original `CameraTrap_Phase1_Team_Mixed_v2` directory privately under
`runs/phase1_recovered/` (ignored by Git). Keep the internal relative paths:

```text
CameraTrap_Phase1_Team_Mixed_v2/
  data/cct20_team_v2/          # manifest, images, annotations and review evidence
  handoff/                   # dataset_release.json and referenced frozen files
  runs/data/                 # original Dataset Card
  runs/member3/triage/frozen/ # best.pt, policy.json, summary.json
  runs/member3/species/frozen/# best.pt, policy.json, summary.json
```

Prefer the entire original workspace. The Phase1_Submission ZIP produced by the
notebook is an evidence/code package: it deliberately excludes raw images and
weights, so it may not be sufficient. Do not rerun dataset preparation to replace
missing frozen files: this would not establish the identity of the evaluated release.

## Verify after recovery

From the repository root:

```bash
python tools/audit_phase2_inputs.py \
  --workspace runs/phase1_recovered/CameraTrap_Phase1_Team_Mixed_v2 \
  --output runs/phase2/input_audit.json
```

Exit code 1 means missing or mismatched inputs; 0 means these input checks pass.
The script does not execute the notebook or load checkpoint pickle contents.
It checks frozen-file hashes, the manifest identity recorded by Phase 1, split
separation, image presence and selected checkpoint hashes/calibration policies.
It also counts boxed positives, empties and positives requiring box review.
Image decoding, annotation completeness and visual verification remain the next step.

The expected manifest SHA-256 is
`d3bfb6ba0c9d31412f9d603a6f63ae7c4890f458c5d2aa2a87d7653bb882bb53`.
Checkpoint hashes and class order come from the committed experiment summaries.
Do not mix inputs from a different training release.

## Next step after recovery

Build detection annotations from verified human boxes while preserving all original
split assignments. Animal-positive frames with incomplete or missing boxes need
review or exclusion from detection supervision, never negative labels. Preserve
timestamps and event IDs for later event counting. Review provenance separately
from box geometry. No new split or training has been started at this stage.
