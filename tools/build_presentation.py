from pathlib import Path
import ast
import nbformat

ROOT = Path(__file__).resolve().parents[1]
NAME = 'Phase1_Final_Presentation'
path = ROOT/'notebooks'/f'{NAME}.ipynb'
notebook = nbformat.read(path, 4)
for cell in notebook.cells:
    if cell.cell_type == 'code':
        ast.parse(cell.source)
        cell.outputs = []
        cell.execution_count = None
nbformat.validate(notebook)
nbformat.write(notebook, path)
export = ''.join('\n'+cell.source+'\n' for cell in notebook.cells if cell.cell_type == 'code')
ast.parse(export)
(ROOT/f'{NAME}.py').write_text(export)
print(f'Exported {len(notebook.cells)} cells.')
