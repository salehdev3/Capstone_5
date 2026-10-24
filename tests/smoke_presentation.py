""
from pathlib import Path
import ast
import runpy
import tempfile
import os
import nbformat

ROOT = Path(__file__).resolve().parents[1]
read = nbformat.read
merged = read(ROOT/'notebooks/Phase1_Final_Presentation.ipynb', 4)
nbformat.validate(merged)
for cell in merged.cells:
    if cell.cell_type == 'code':
        ast.parse(cell.source)
        assert 'from google.colab import files' not in cell.source
        assert 'ARCHIVES = ' not in cell.source
ast.parse((ROOT/'Phase1_Final_Presentation.py').read_text())
nbformat.validate(merged)

runpy.run_path(str(ROOT/'tests/smoke_notebooks.py'), run_name='__main__')
print('PASS: merged presentation cells, source snapshot, and Python export')
