from pathlib import Path
import ast
import os
import tempfile
import nbformat
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
notebook = nbformat.read(ROOT / 'notebooks/Phase1_Final_Presentation.ipynb', 4)
source = next(c.source for c in notebook.cells if 'blur_review' in c.metadata.get('tags', []))
tree = ast.parse(source)
tree.body = tree.body[:-1]
namespace = {'Path': Path, 'pd': pd}
exec(compile(tree, '<blur_review>', 'exec'), namespace)
namespace['display'] = lambda *args: None
with tempfile.TemporaryDirectory() as directory:
    os.chdir(directory)
    data = Path('data')
    data.mkdir()
    image = data / 'image.png'
    Image.new('RGB', (20, 20)).save(image)
    path = data / 'blur_review.csv'
    pd.DataFrame([dict(id='sample', path=str(image), blur_score=514.37,
                       reviewer='', decision='', notes='')]).to_csv(path, index=False)
    original = path.read_bytes()
    reviewer, rows, button, output = namespace['blur_review_widget'](data)
    button.click()
    assert path.read_bytes() == original
    reviewer.value = 'Reviewer One'
    button.click()
    assert path.read_bytes() == original
    rows[0][1].value = 'flag'
    button.click()
    assert path.read_bytes() == original
    rows[0][2].value = 'Motion obscures the animal.'
    button.click()
    saved = pd.read_csv(path)
    assert saved.loc[0, 'reviewer'] == 'Reviewer One'
    assert saved.loc[0, 'decision'] == 'flag'
    reviewer2, rows2, button2, _ = namespace['blur_review_widget'](data)
    assert reviewer2.value == 'Reviewer One' and rows2[0][1].value == 'flag'
    assert rows2[0][2].value == 'Motion obscures the animal.'
    rows2[0][1].value = 'keep'
    button2.click()
    assert pd.read_csv(path).loc[0, 'decision'] == 'keep'
    Path('handoff').mkdir()
    Path('handoff/dataset_release.json').write_text('{}')
    frozen = path.read_bytes()
    rows2[0][1].value = 'flag'
    button2.click()
    assert path.read_bytes() == frozen
print('PASS: required review fields, keep/flag persistence, reload and frozen release protection')
