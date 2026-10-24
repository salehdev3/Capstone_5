""
from pathlib import Path
import ast
import json
import re
import nbformat

ROOT = Path(__file__).resolve().parents[1]
NAME = 'Phase1_Final_Presentation'
intro = '''# الفرز الآلي لصور كاميرات الحياة البرية
## المرحلة الأولى: البيانات والتصنيف والتقييم

المشروع الخامس — AI251P Applied Computer Vision

يدرس المشروع تصنيف الإطارات إلى فارغة أو محتوية على حيوان، وتصنيف الأنواع
باستخدام القصاصات البشرية من CCT20. نقارن CNN من الصفر مع ResNet-50
بتجميد الشبكة الأساسية وتدريبها الكامل، مع فصل مواقع الكاميرات بين المجموعات.

يتكون العرض من أربعة أقسام: البيانات، CNN، التعلم بالنقل، والمقارنة والتقرير.
تعرض النتائج بعد تنفيذ الخلايا على البيانات الفعلية. تتطلب مراجعة التسميات
وملاحظات الفريق إدخالًا بشريًا، وتظل حالة التقرير غير مكتملة حتى توفر أدلتها.
التشغيل الكامل يحتاج GPU ومكتبات requirements.in وزوج torch/torchvision متوافق.
في Colab تثبت خلية البيئة المكتبات، ثم تنفذ الأقسام بالتسلسل في الجلسة نفسها.
ملف Python المصاحب يستخدم خلايا # %% ويمكن فتحه في محرر يدعم Python Interactive؛
التدقيق التفاعلي يتم في الدفتر، ولا يتجاوز تشغيل السكربت متطلبات التدقيق.
'''
cells = [nbformat.v4.new_markdown_cell(intro)]
cells.append(nbformat.v4.new_code_cell("import subprocess, sys\nsubprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ImageHash', 'pandas', 'scipy', 'matplotlib', 'Pillow', 'PyYAML', 'ipywidgets', 'scikit-learn', 'opencv-python-headless', 'tqdm', 'wandb', 'transformers==4.49.0'])"))
titles = ['البيانات والتدقيق — M1', 'الشبكة المدربة من الصفر — M2', 'التعلم بالنقل — M3', 'المقارنة والتحقق والتقرير']
for member, path in enumerate(sorted((ROOT/'notebooks').glob('Phase1_Member_*.ipynb')), 1):
    cells.append(nbformat.v4.new_markdown_cell(f'# {member}. {titles[member-1]}'))
    source = nbformat.read(path, 4)
    omitted = {0, 1, 2}
    omitted |= {35, 36} if member == 1 else ({4, 29, 30} if member in (2, 3) else {4})
    for index, original in enumerate(source.cells):
        if index in omitted:
            continue
        text = original.source
        if original.cell_type == 'markdown':
            
            text = text.splitlines()[0]
            text = re.sub(r' — .*Lab.*', '', text)
            text = text.replace('حفظ ملفات التسليم للعضوين 2 و3', 'تثبيت إصدار البيانات والإعداد المشترك')
            text = text.replace('المزامنة والتسليم', 'توثيق التجارب').replace('مخرج مطلوب', 'توثيق البيانات')
            cell = nbformat.v4.new_markdown_cell(text)
        else:
            text = text.replace('from google.colab import files\nfiles.download(archive)', "print('Submission archive:', archive)")
            if member == 4:
                text = re.sub(r"NAMES=\[.*?\]", "NAMES=['Phase1_Final_Presentation.ipynb']", text)
                text = text.replace('Run notebooks 1 → (2 and 3) → 4.', 'Run Phase1_Final_Presentation.ipynb in section order.')
            ast.parse(text)
            cell = nbformat.v4.new_code_cell(text)
        cell.metadata['source_section'] = {'notebook': path.name, 'cell': index}
        cells.append(cell)


payload = json.dumps([{'cell_type': c.cell_type, 'source': c.source, 'metadata': {}} for c in cells], ensure_ascii=False)
bootstrap = '''from pathlib import Path
import json
presentation_root = Path.cwd() if Path.cwd().name == 'CameraTrap_Phase1_Team' else Path.cwd()/'CameraTrap_Phase1_Team'
presentation_sources = presentation_root/'notebooks'
presentation_sources.mkdir(parents=True, exist_ok=True)
presentation_cells = json.loads(PAYLOAD)
for cell in presentation_cells:
    if cell['cell_type']=='code':
        cell.update(outputs=[], execution_count=None)
presentation_document = dict(nbformat=4, nbformat_minor=4, metadata={}, cells=presentation_cells)
presentation_path = presentation_sources/'Phase1_Final_Presentation.ipynb'
presentation_text = json.dumps(presentation_document, ensure_ascii=False, indent=1)
if presentation_path.exists():
    assert presentation_path.read_text() == presentation_text, 'Existing presentation source differs; use a fresh project directory.'
else:
    presentation_path.write_text(presentation_text)
'''.replace('PAYLOAD', repr(payload))
cells.insert(2, nbformat.v4.new_code_cell(bootstrap))
notebook = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name':'python3','display_name':'Python 3','language':'python'}})
nbformat.validate(notebook)
nbformat.write(notebook, ROOT/'notebooks'/f'{NAME}.ipynb')
export = ''
for cell in cells:
    if cell.cell_type == 'code':
        export += '\n' + cell.source + '\n'
ast.parse(export)
(ROOT/f'{NAME}.py').write_text(export)
print(f'Built {len(cells)} cells and Python export.')
