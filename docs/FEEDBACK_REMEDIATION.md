# Evaluation and reproducibility procedures

Historical experiment metrics are identified by their dataset hashes and recorded training configuration. CPU validation, human label review and GPU training are documented separately.

## Evaluation outputs

- Report includes both triage confusion matrices, argmax versus calibrated operating decisions, calibration split and empirical recall constraint.
- Full-test error totals replace gallery-only totals; future evaluation exports `test_operating_predictions.csv` and `test_error_patterns.csv` with the decision rule.
- Report defines camera-cluster bootstrap intervals and both species macro-F1 denominators, and includes the five-frame Tabuk results.
- New dataset releases require at least 100 completed training-event reviews. The reported Phase 1 audit contains 10 training events.
- Illumination analysis compares error rates across day, night and uncertain groups; training interventions are evaluated on validation data.
- Privacy statements now acknowledge tracked team images. Consent and repository visibility still need owner verification.
- `requirements-cpu-validation.txt` pins the tested local Python 3.12 CPU validation environment, not the original Colab training environment.

## Experiment artifact collection

1. Retrieve `species_crop_support.csv` from the frozen release matching manifest SHA-256 `d3bfb6ba0c9d31412f9d603a6f63ae7c4890f458c5d2aa2a87d7653bb882bb53`. Publish its train, val_select, val_calibrate and test counts. Test counts alone cannot recover the other partitions.
2. Retrieve the original `handoff/requirements_member*.txt` and runtime JSON from the experiment artifacts. CPU validation and GPU experiment environments use separate version records.
3. Record the run IDs for the 28 October W&B rerun attributed to Mohannad. Compare dataset hash, code revision, configuration and per-task metrics side by side. Image-audit records and training-replay records have distinct scopes.
4. Each new release requires at least 100 completed sampled training-event reviews. Existing frozen releases must not be edited. Apply corrections before freezing and retrain all compared models when data changes.
5. Run a controlled GPU experiment comparing weighted loss with the existing unweighted baseline. Choose on val_select; calibrate temperature and triage threshold on val_calibrate. Do not optimize against known test results. Acceptance criteria are evaluated against measured results.

## CPU validation

Use Python 3.12 in a virtual environment:

```sh
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-cpu-validation.txt
MPLBACKEND=Agg python tests/smoke_presentation.py
python tests/test_blur_review.py
```

After editing the notebook, run `python tools/build_presentation.py` before validation. To build the corrected report, run `pdflatex -interaction=nonstopmode -halt-on-error main.tex` twice from `docs/phase1_report/official_template/`. The saved PDF is `docs/phase1_report/Project5_Phase1.pdf`.

## Execution workflow

New releases default to inverse-frequency class-weighted loss. Existing frozen configurations and historical metrics remain unchanged. All six comparison models must use the same new release. The notebook now exports threshold-specific triage matrices/plots and full error counts, and includes species support, audit bounds, both macro-F1 definitions, operating policy, camera CI method and Tabuk evidence in the generated report. Internal QA remains in `handoff/qa_release.json`; scientific results are written to the report. Saved team/report input fields survive rerunning the input cell.

The team identifies Mohannad as the replay operator; see `evidence/replay_attribution.json` under the report directory. Compare actual downloaded original and rerun result directories, each containing `triage/{scratch,frozen,full}` and `species/{scratch,frozen,full}`:

```sh
python tools/compare_replay.py --original ORIGINAL_RESULTS --replay RERUN_RESULTS --reviewer Mohannad --run-url ACTUAL_WANDB_RUN_URL --replay-date 2026-10-28 --output docs/phase1_report/evidence/replay
```

This exports a side-by-side CSV and evidence JSON, checks all six experiments and matching provenance, and returns nonzero for differing numbers. Run records document the operator and execution history; the comparison checks numerical agreement.

For illumination analysis, use the entire `test_operating_predictions.csv`, not an error gallery:

```sh
python tools/analyze_illumination.py --predictions RUN/test_operating_predictions.csv --review review/illumination.csv --output review/illumination_metrics
```

The first execution creates a review template without overwriting existing work. A named human labels every sample day/night/uncertain, then reruns the same command. It rejects partial reviews, preserves uncertain cases and computes rates over all samples in each group. Illumination labels characterize errors; they must not be used to tune against the held-out test.
