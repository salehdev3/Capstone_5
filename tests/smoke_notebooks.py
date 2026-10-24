""
from pathlib import Path
import ast, json, os, tempfile, subprocess, sys, hashlib
import nbformat
ROOT = Path(__file__).resolve().parents[1]
presentation = nbformat.read(ROOT/'notebooks/Phase1_Final_Presentation.ipynb', as_version=4)
def section(title):
    start = next(i for i,c in enumerate(presentation.cells) if c.cell_type == 'markdown' and title in ''.join(c.source))
    end = next((i for i in range(start+1,len(presentation.cells)) if presentation.cells[i].cell_type == 'markdown' and ''.join(presentation.cells[i].source).startswith('# Member')), len(presentation.cells))
    return [c for c in presentation.cells[start:end] if c.cell_type == 'code']
def virtual(cells, slots=None):
    out = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('') for _ in range((max(slots) + 1) if slots else len(cells))])
    for i, cell in enumerate(cells):
        out.cells[slots[i] if slots else i] = cell
    return out
data_cells = section('Member 1 — Data Foundations')
cnn_cells = section('Member 2 — CNN Classification')
transfer_cells = section('Member 3 — Transfer Learning')
report_cells = section('Member 4 — Comparison')
n1 = nbformat.v4.new_notebook(cells=data_cells)
n3 = nbformat.v4.new_notebook(cells=transfer_cells)
n2 = virtual([
    next(c for c in cnn_cells if 'def read_json' in c.source and 'ROOT=' in c.source),
    next(c for c in cnn_cells if 'FrameDataset' in c.source and 'class Preprocessor' in c.source),
    next(c for c in cnn_cells if 'def seed_everything' in c.source),
    next(c for c in cnn_cells if 'class ConvNN' in c.source),
    next(c for c in cnn_cells if 'def ece' in c.source),
    next(c for c in cnn_cells if 'def train_model' in c.source),
    next(c for c in cnn_cells if 'def evaluate_model' in c.source),
], [3,6,7,11,14,16,24])
n4 = nbformat.v4.new_notebook(cells=report_cells)
notebooks = {'Phase1_Member_1_Data.ipynb':n1,'Phase1_Member_2_CNN.ipynb':n2,'Phase1_Member_3_TransferLearning.ipynb':n3,'Phase1_Member_4_Report.ipynb':n4,'Phase1_Final_Presentation.ipynb':presentation}
for notebook in notebooks.values():
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code' and not cell.source.startswith('%'):
            ast.parse(cell.source)
print('PASS: unified notebook schema and section code cells')
n1=notebooks['Phase1_Member_1_Data.ipynb'];n2=notebooks['Phase1_Member_2_CNN.ipynb'];n3=notebooks['Phase1_Member_3_TransferLearning.ipynb']
shared=(ROOT/'preprocessing.py').read_text()
assert all(any(c.cell_type=='code' and c.source.startswith(shared) for c in n.cells) for n in [n1,n2,n3])
print('PASS: byte-identical preprocessing in Members 1, 2, 3')
with tempfile.TemporaryDirectory(prefix='phase1-smoke-') as tmp:
    os.chdir(tmp)
    ns={}
    for i in [3,6,7,11,14,16,24]:exec(n2.cells[i].source,ns)
    import torch,numpy as np,pandas as pd
    from PIL import Image
    torch.set_num_threads(2)
    data=Path('data');data.mkdir()
    rows=[]
    for split in ['train','val_select','val_calibrate','test']:
        for i in range(8):
            label=i%2;path=data/f'{split}-{i}.png'
            rng=np.random.default_rng(len(rows));pixels=rng.integers(0,50,(40,60,3),dtype=np.uint8)
            pixels[:,:,label]=pixels[:,:,label]+150
            Image.fromarray(pixels).save(path)
            rows.append(dict(id=f'{split}-{i}',path=str(path),camera=f'{split}-{i%2}',event=f'{split}-{i}',
              split=split,animal=label,labels=['fox'] if label else [],
              boxes=[{'species':['fox','deer'][label],'xyxy':[.1,.1,.9,.9]}]))
    ns['write_json'](data/'manifest.json',rows);ns['write_json'](data/'species.json',['fox','deer'])
    cfg=dict(project='phase1-synthetic-validation',profile='smoke',seed=42,data_dir=str(data),image_size=32,
      batch_size=4,epochs=2,patience=2,warmup_epochs=2,backbone='microsoft/resnet-50',optimizer='Adam',
      lr_scratch=.001,lr_frozen=.001,lr_full=.0001,weight_decay=0.,label_smoothing=.05,class_weighted_loss=True,
      target_animal_recall=.99,species_review_confidence=.6,bootstrap_repeats=20,
      preprocessing_sha256=hashlib.sha256(shared.encode()).hexdigest())
    ns.update(CFG=cfg,DATA=data,rows=rows,DATA_HASH=ns['digest'](data/'manifest.json'),DEVICE='cpu',WANDB_MODE='disabled',OUT=Path('runs/member2'))
    
    split_ns={'np':np}
    splitcell=next(c.source for c in n1.cells if c.cell_type=='code' and 'def assign_sites' in c.source)
    exec(splitcell[:splitcell.index('site_map=assign_sites')],split_ns)
    fixture=[dict(id=f'{cam}-{i}',camera=str(cam),event=f'{cam}-{i}',labels=['fox'] if i%2 else []) for cam in range(20) for i in range(20)]
    mapping=split_ns['assign_sites'](fixture,42)
    assert len(mapping)==20 and set(mapping.values())=={'train','val_select','val_calibrate','test'}
    for r in fixture:r['split']=mapping[r['camera']]
    split_ns['assert_disjoint'](fixture)
    print('PASS: camera-group stratification and disjointness')
    
    for task in ['triage','species']:
        result=ns['train_model'](task,'scratch')
        metric=ns['evaluate_model'](task,'scratch')
        assert metric['n']==8 and metric['unknown_count']==0
        history=pd.read_csv(ns['OUT']/task/'scratch'/'history.csv')
        assert np.allclose(history.learning_rate,[.0005,.001])
        assert result['context']['training_implementation']==2
        assert ns['train_model'](task,'scratch')==result
        assert ns['evaluate_model'](task,'scratch')==metric
        ns['show_results'](task,'scratch')
        folder=ns['OUT']/task/'scratch'
        assert all((folder/name).exists() for name in ['best.pt','policy.json','history.csv','test_probabilities.npz','learning_loss.png','learning_accuracy.png','reliability.png','confusion.png'])
    print('PASS: scratch training, calibration, evaluation, cached reuse and saved plots for both tasks')
    old=cfg['epochs'];cfg['epochs']=3
    try:ns['train_model']('triage','scratch')
    except AssertionError:pass
    else:raise AssertionError('Changed training configuration must reject cache')
    cfg['epochs']=old
    
    model=ns['make_model']('frozen',2,pretrained=False,size=32)
    ns['freeze_backbone'](model,True);model.eval();model.classifier.train()
    params=list(model.net.base_model.parameters());before=params[0].detach().clone()
    x=torch.randn(2,3,32,32);y=torch.tensor([0,1]);loss=torch.nn.functional.cross_entropy(model(x),y);loss.backward()
    assert all(p.grad is None for p in params) and model.classifier[1].weight.grad is not None
    assert torch.equal(before,params[0])
    model.zero_grad(set_to_none=True);ns['freeze_backbone'](model,False)
    torch.nn.functional.cross_entropy(model(x),y).backward()
    assert params[0].grad is not None
    print('PASS: ResNet head replacement, frozen and full gradient flow (random weights)')
    
    n4=notebooks['Phase1_Member_4_Report.ipynb']
    probe_cell=next(c.source for c in n4.cells if c.cell_type=='code' and 'PROBE_SOURCE = ' in c.source)
    tree=ast.parse(probe_cell)
    assign=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROBE_SOURCE' for t in n.targets))
    probe=ast.literal_eval(assign.value)
    ckpt=ns['OUT']/'triage/scratch/best.pt'
    p=subprocess.run([sys.executable,'-c',probe,str(ckpt),rows[0]['path']],text=True,capture_output=True,timeout=120)
    assert p.returncode==0,p.stderr
    assert json.loads(p.stdout)['finite']
    print('PASS: saved checkpoint loads and predicts in a fresh CPU process')
    
    
    import shutil, types
    for r in rows:
        r['pixel_sha256']=hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()
        r['processed_file_md5']=hashlib.md5(Path(r['path']).read_bytes()).hexdigest()
    
    cfg['acceptance']={'animal_recall_min':.99,'empty_removal_min':.2,'species_macro_f1_min':.65,'species_ece_max':.1}
    ns['write_json'](data/'audit_report.json',{'status':'complete','unresolved_flagged_events':0})
    pd.DataFrame([{'reviewer':'','decision':'','notes':''}]).to_csv(data/'blur_review.csv',index=False)
    Path('runs/data').mkdir(parents=True,exist_ok=True)
    Path('runs/data/DATASET_CARD.md').write_text('Synthetic fixture only. Pending: human observations')
    for task in ['triage','species']:
        for regime in ['frozen','full']:
            dest=Path('runs/member3')/task/regime
            shutil.copytree(Path('runs/member2')/task/'scratch',dest)
    for member in [2,3]:
        out=Path(f'runs/member{member}')
        required=[p for p in out.rglob('*') if p.is_file()]
        ns['write_json'](f'handoff/member{member}_release.json',{'schema':2,'dataset_hash':ns['DATA_HASH'],
          'output_dir':str(out),'files':{str(p):ns['digest'](p) for p in required}})
    for directory in ['review','submission','notebooks']:Path(directory).mkdir(exist_ok=True)
    shutil.copy2(ROOT/'notebooks/Phase1_Final_Presentation.ipynb',Path('notebooks/Phase1_Final_Presentation.ipynb'))
    ns['release']={'dataset_card':'runs/data/DATASET_CARD.md'}
    qa_cell=next(c.source for c in n4.cells if c.cell_type=='code' and 'checks=[]' in c.source)
    exec(qa_cell,ns)
    assert len(ns['comparison'])==6 and all(c['passed'] for c in ns['checks'])
    
    for start in ['import matplotlib.pyplot as plt','audit_path=','links_path=','gaps=[]','def table(df)']:
        cell=next(c.source for c in n4.cells if c.cell_type=='code' and c.source.startswith(start))
        if start=='gaps=[]':ns['fresh']=[]
        exec(cell,ns)
    assert ns['STATUS']=='INCOMPLETE'
    report=Path('submission/REPORT_DRAFT.md').read_text()
    assert '\n\n' in report and '| task | regime |' in report
    assert ns['gaps']
    
    colab=types.ModuleType('google.colab');colab.files=types.SimpleNamespace(download=lambda path:None)
    sys.modules['google.colab']=colab
    package=next(c.source for c in n4.cells if c.cell_type=='code' and c.source.startswith('import shutil'))
    exec(package,ns)
    repo=Path('submission/repository')
    assert not list(repo.rglob('*.pt')) and not list(repo.rglob('*.jpg'))
    assert len((repo/'.gitignore').read_text().splitlines())>5
    assert (repo/'acceptance.csv').exists()
    print('PASS: six-row report fixture, pending-review gates, Markdown and private-file exclusion')
print('All smoke checks passed. Synthetic fixture results are discarded.')
