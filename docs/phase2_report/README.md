# Phase 2 report

English report following the supplied four-member Phase 2 template (M4, M5, M8).

- `Phase2_Report.pdf`: compiled report.
- `main.tex` and `figures/`: editable Overleaf source; compiler pdfLaTeX.
- `Phase2_Report_Overleaf.zip`: import-ready source archive.
- `evidence/`: supplied machine-readable results and extracted training histories used in figures and tables. `input_provenance.json` identifies the uploaded input archive.

The submitted Phase 1 report is unchanged. The recovered species classifier is a newly trained Phase 2 checkpoint, not the original Phase 1 weights. Recovery-v2 mask test results are development evidence after earlier test inspection; they are not an independent unseen test. The report retains the measured NO-GO deployment verdict.

Final presentation and recorded demo video remain separate deliverables. ONNX decision/operating-metric parity passes on all 1,208 validation frames with 1,626 matched detections. Original exported summary, per-image CSV, metrics CSV, environment JSON, SHA256 checksums and independent consistency verification are archived in `docs/onnx_parity/full_validation_20261119T070317993000Z/`. Full-test error taxonomy and verified day/night analysis are not claimed as measured. Review the provisional self-assessment before submission.

Build: run `pdflatex -interaction=nonstopmode -halt-on-error main.tex` twice.

Image panels are derived camera-trap evaluation illustrations. Dataset attribution: Caltech Camera Traps / LILA, CDLA-Permissive, https://lila.science/datasets/caltech-camera-traps .

The separate Small/Medium/Large detector recall table is derived from the saved fixed operating-point outputs; original CSV, extraction protocol and population/aggregate verification are archived under `evidence/detection_size_breakdown/`. No training or historical inference was repeated.

The V3 experiment section compares actual new validation results and candidate CPU/HTTP evidence. It distinguishes standalone masks from the deployed SAM pipeline and explains the fixed all-class macro-F1 support ceiling. Archived experiment summaries are under `results/phase2/optimization_v3_review/`. No new historical test metrics replace v2.
