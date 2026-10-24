"""Exercise merged presentation cells with the existing synthetic workflow."""
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
cwd = Path.cwd()
try:
    with tempfile.TemporaryDirectory() as directory:
        os.chdir(directory)
        exec(merged.cells[2].source, {})
        exec(merged.cells[2].source, {})
        saved = read(Path(directory)/'CameraTrap_Phase1_Team/notebooks/Phase1_Final_Presentation.ipynb', 4)
        nbformat.validate(saved)
finally:
    os.chdir(cwd)

def presentation_read(path, *args, **kwargs):
    source = read(path, *args, **kwargs)
    for cell in merged.cells:
        origin = cell.metadata.get('source_section', {})
        if origin.get('notebook') == Path(path).name:
            source.cells[origin['cell']] = cell
    return source

nbformat.read = presentation_read
try:
    runpy.run_path(str(ROOT/'tests/smoke_notebooks.py'), run_name='__main__')
finally:
    nbformat.read = read
print('PASS: merged presentation cells, source snapshot, and Python export')
