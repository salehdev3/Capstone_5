from pathlib import Path
import ast
import json
import nbformat

ROOT = Path(__file__).resolve().parents[1]
NAME = 'Phase1_Final_Presentation'
path = ROOT/'notebooks'/f'{NAME}.ipynb'
notebook = nbformat.read(path, 4)
notebook.cells[0].source = notebook.cells[0].source.replace('ملف Python المصاحب يستخدم خلايا # %% ويمكن فتحه في محرر يدعم Python Interactive؛', 'ملف Python المصاحب يحتوي الكود التنفيذي للأقسام بالتسلسل؛')
cells = [cell for i, cell in enumerate(notebook.cells) if i != 2]
payload = json.dumps([{'cell_type': c.cell_type, 'source': c.source, 'metadata': {}} for c in cells], ensure_ascii=False)
bootstrap = notebook.cells[2].source
assignment = next(node for node in ast.parse(bootstrap).body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'presentation_cells' for t in node.targets))
literal = ast.get_source_segment(bootstrap, assignment.value.args[0])
notebook.cells[2].source = bootstrap.replace(literal, repr(payload), 1)
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
