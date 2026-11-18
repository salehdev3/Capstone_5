# Phase 2 report

English report following the supplied four-member Phase 2 template (M4, M5, M8).

- `Phase2_Report.pdf`: compiled report.
- `main.tex` and `figures/`: editable Overleaf source; compiler pdfLaTeX.
- `Phase2_Report_Overleaf.zip`: import-ready source archive.
- `evidence/`: supplied machine-readable results and extracted training histories used in figures and tables. `input_provenance.json` identifies the uploaded input archive.

The submitted Phase 1 report is unchanged. The recovered species classifier is a newly trained Phase 2 checkpoint, not the original Phase 1 weights. Recovery-v2 mask test results are development evidence after earlier test inspection; they are not an independent unseen test. The report retains the measured NO-GO deployment verdict.

Final presentation and recorded demo video remain separate deliverables. Post-NMS ONNX decision parity, full-test error taxonomy and verified day/night analysis are not claimed as measured. Review the provisional self-assessment before submission.

Build: run `pdflatex -interaction=nonstopmode -halt-on-error main.tex` twice.

Image panels are derived camera-trap evaluation illustrations. Dataset attribution: Caltech Camera Traps / LILA, CDLA-Permissive, https://lila.science/datasets/caltech-camera-traps .
