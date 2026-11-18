"""Retrain a separately versioned Phase-1-style frozen species classifier for Phase 2."""
from pathlib import Path
from collections import defaultdict
import argparse
import hashlib
import importlib.util
import io
import json
import math
import random
import shutil
import time
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from scipy.optimize import minimize_scalar

MANIFEST_SHA='d3bfb6ba0c9d31412f9d603a6f63ae7c4890f458c5d2aa2a87d7653bb882bb53'
PREPROCESSING_SHA='87e3366dd7946e566c3a9238095748eb698ed0b57e2a9a5699d49817c440a5eb'
CONFIG=dict(version='species_recovery_v1',task='species',regime='frozen',backbone='microsoft/resnet-50',
    seed=42,image_size=224,batch_size=32,epochs=12,patience=4,optimizer='Adam',lr_frozen=.001,
    weight_decay=0.,label_smoothing=0.,class_weighted_loss=False,workers=0,
    preprocessing_sha256=PREPROCESSING_SHA,manifest_sha256=MANIFEST_SHA,
    training_time_limit=None,species_review_confidence=.6,
    scope='New Phase-2 deployment classifier; not the original Phase-1 checkpoint or a claim of identical results.')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def save_torch(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');torch.save(value,temp);temp.replace(path)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def metrics(logits,labels,classes,temperature=1.):
    z=torch.as_tensor(logits,dtype=torch.float64)/temperature
    labels=np.asarray(labels,dtype=int)
    probability=z.softmax(-1).numpy();pred=probability.argmax(1)
    scores=[];per_class=[]
    for k,name in enumerate(classes):
        tp=int(((pred==k)&(labels==k)).sum());fp=int(((pred==k)&(labels!=k)).sum());fn=int(((pred!=k)&(labels==k)).sum())
        precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1);f1=2*tp/max(2*tp+fp+fn,1)
        scores.append(f1);per_class.append(dict(species=name,precision=precision,recall=recall,f1=f1,support=int((labels==k).sum())))
    conf=probability.max(1);correct=pred==labels;bins=np.minimum((conf*15).astype(int),14)
    ece=sum(float((bins==i).mean())*abs(float(correct[bins==i].mean())-float(conf[bins==i].mean())) for i in range(15) if np.any(bins==i))
    return dict(n=len(labels),accuracy=float(correct.mean()),macro_f1_all_training_classes=float(np.mean(scores)),
        nll=float(nn.functional.cross_entropy(z,torch.as_tensor(labels))),ece=ece,
        top3_accuracy=float(np.mean([y in row for y,row in zip(labels,np.argsort(probability,axis=1)[:,-min(3,len(classes)):])])),per_class=per_class)


def fit_temperature(logits,labels):
    z=np.asarray(logits,dtype=np.float64);y=np.asarray(labels,dtype=int)
    def objective(log_t):
        a=z/math.exp(float(log_t));a=a-a.max(axis=1,keepdims=True)
        return float(np.mean(np.log(np.exp(a).sum(1))-a[np.arange(len(y)),y]))
    result=minimize_scalar(objective,bounds=(math.log(.05),math.log(20.)),method='bounded')
    if not result.success:raise RuntimeError('Temperature calibration failed')
    return float(math.exp(result.x))


def prepare(root):
    root=Path(root);source=root/'phase1_source'
    manifest=source/'data/cct20_team_v2/manifest.json';vocabulary=source/'data/cct20_team_v2/species.json'
    if not manifest.is_file() or not vocabulary.is_file():
        raise FileNotFoundError('Install the provided frozen metadata ZIP first; no new split is allowed.')
    assert digest(manifest)==MANIFEST_SHA,'Frozen Phase-1 manifest changed'
    rows=json.loads(manifest.read_text());classes=json.loads(vocabulary.read_text())
    assert isinstance(classes,list) and len(classes)==14 and len(set(classes))==14
    assert len({r['id'] for r in rows})==len(rows)
    for key in ('camera','event','pixel_sha256'):
        groups=defaultdict(set)
        for r in rows:
            assert r.get(key);groups[r[key]].add(r['split'])
        assert all(len(s)==1 for s in groups.values()),f'Cross-split leakage: {key}'
    run=root/'phase2_results/species_recovery_v1';run.mkdir(parents=True,exist_ok=True)
    config_path=run/'config.json'
    if config_path.exists():assert json.loads(config_path.read_text())==CONFIG,'Use a new version for changed settings'
    else:save_json(config_path,CONFIG)
    preprocessing_path=source/'preprocessing.py'
    assert preprocessing_path.is_file() and digest(preprocessing_path)==PREPROCESSING_SHA
    spec=importlib.util.spec_from_file_location('species_recovery_preprocessing',preprocessing_path)
    preprocessing=importlib.util.module_from_spec(spec);spec.loader.exec_module(preprocessing)
    samples={split:[] for split in ('train','val_select','val_calibrate','test')}
    for row in rows:
        for i,box in enumerate(row['boxes']):
            species=box['species'];coords=box['xyxy']
            assert len(coords)==4 and all(math.isfinite(x) for x in coords)
            assert 0<=coords[0]<coords[2]<=1 and 0<=coords[1]<coords[3]<=1
            record=dict(row,path=str(root/'images'/(row['id']+'.jpg')),crop=coords,
                label=classes.index(species) if species in classes else -1,sample_id=f"{row['id']}_{i}",species=species)
            samples[row['split']].append(record)
    train_vocabulary={r['species'] for r in samples['train']}
    assert train_vocabulary==set(classes),'Training vocabulary differs'
    known={s:[r for r in v if r['label']>=0] for s,v in samples.items()}
    save_json(run/'data_scope.json',dict(manifest_sha256=MANIFEST_SHA,
        counts={s:dict(known=len(known[s]),unknown=len(samples[s])-len(known[s])) for s in samples},
        policy='Frozen camera/event split; unknown species excluded from closed-set scores and reported separately. Test is not used in training, selection or calibration.'))
    return run,rows,classes,samples,known,preprocessing


def verify_images(samples,run,tag):
    unique={r['id']:r for r in samples};audit=[]
    for number,r in enumerate(unique.values(),1):
        p=Path(r['path'])
        if not p.is_file():raise FileNotFoundError('Missing original image: '+str(p)+'. Restore the original Phase-2 images before training.')
        raw=p.read_bytes();md5=hashlib.md5(raw).hexdigest()
        pixel_match=None
        if md5!=r['processed_file_md5']:
            with Image.open(io.BytesIO(raw)) as im:
                rgb=im.convert('RGB')
                pixel_match=rgb.size==(r['width'],r['height']) and hashlib.sha256(rgb.tobytes()).hexdigest()==r['pixel_sha256']
            if not pixel_match:raise ValueError('Image pixels differ from frozen Phase-1 release: '+r['id'])
        audit.append(dict(id=r['id'],sha256=hashlib.sha256(raw).hexdigest(),phase1_file_match=md5==r['processed_file_md5'],phase1_pixels_match=True))
        if number%500==0:print('Verified images:',number,'/',len(unique),flush=True)
    save_json(run/(tag+'_image_hashes.json'),audit)


def make_model(count,cache=None,pretrained=True):
    from transformers import ResNetConfig,ResNetForImageClassification
    class TransferNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net=(ResNetForImageClassification.from_pretrained('microsoft/resnet-50',cache_dir=str(cache)) if pretrained
                else ResNetForImageClassification(ResNetConfig()))
            self.net.classifier=nn.Sequential(nn.Flatten(start_dim=1),nn.Linear(self.net.config.hidden_sizes[-1],count))
            self.net.config.num_labels=count
        def forward(self,x):return self.net(x).logits
        @property
        def classifier(self):return self.net.classifier
    model=TransferNet()
    for p in model.net.base_model.parameters():p.requires_grad=False
    for p in model.classifier.parameters():p.requires_grad=True
    return model


def predict(model,loader,device):
    model.eval();zs=[];ys=[]
    with torch.inference_mode():
        for x,y,_ in loader:
            zs.append(model(x.to(device)).float().cpu().numpy());ys.extend(y.tolist())
    if not zs:raise ValueError('No known-class samples in split')
    return np.concatenate(zs),np.asarray(ys,dtype=np.int64)


def restore_rng(saved,generator):
    random.setstate(saved['python_rng'])
    state=saved['numpy_rng'];np.random.set_state((state[0],np.asarray(state[1],dtype=np.uint32),state[2],state[3],state[4]))
    torch.set_rng_state(saved['torch_rng']);generator.set_state(saved['loader_rng'])
    if torch.cuda.is_available() and saved.get('cuda_rng'):torch.cuda.set_rng_state_all(saved['cuda_rng'])


def capture_rng(generator):
    state=np.random.get_state()
    return dict(python_rng=random.getstate(),numpy_rng=(state[0],state[1].tolist(),state[2],state[3],state[4]),
        torch_rng=torch.get_rng_state(),loader_rng=generator.get_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def install(root,run):
    versioned=Path(root)/'models/species_recovery_v1';versioned.mkdir(parents=True,exist_ok=True)
    for name in ('best.pt','policy.json','summary.json'):
        source=run/name;target=versioned/name
        if target.exists():assert digest(target)==digest(source),'Existing versioned model differs; preserve it and use a new recovery version'
        else:
            temp=target.with_suffix(target.suffix+'.tmp');shutil.copyfile(source,temp);temp.replace(target)
    alias=Path(root)/'models/species';alias.mkdir(parents=True,exist_ok=True)
    # Never replace an existing original checkpoint or its calibration policy.
    for name in ('best.pt','policy.json'):
        target=alias/name
        if not target.exists():
            temp=target.with_suffix(target.suffix+'.tmp');shutil.copyfile(versioned/name,temp);temp.replace(target)
        elif digest(target)!=digest(versioned/name):
            print('Preserved existing classifier file:',target,flush=True)
    save_json(versioned/'installation.json',dict(checkpoint_sha256=digest(versioned/'best.pt'),policy_sha256=digest(versioned/'policy.json'),
        note='New retrained recovery version. Phase-1 report, old metrics and all detector/segmenter weights are unchanged.'))
    print('SPECIES_PT =',repr(str(versioned/'best.pt')),flush=True)
    print('SPECIES_POLICY =',repr(str(versioned/'policy.json')),flush=True)


def train(root):
    root=Path(root);run,rows,classes,samples,known,pre=prepare(root)
    if (run/'summary.json').is_file():
        summary=json.loads((run/'summary.json').read_text())
        assert summary['checkpoint_sha256']==digest(run/'best.pt')
        assert summary['policy_sha256']==digest(run/'policy.json')
        install(root,run);print('Reusing completed recovery:',summary,flush=True);return
    verify_images(known['train']+known['val_select']+known['val_calibrate'],run,'training_selection_calibration')
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4);seed_all(CONFIG['seed'])
    save_json(run/'environment.json',dict(torch=str(torch.__version__),numpy=np.__version__,device=device,
        cuda=torch.version.cuda,amp=device=='cuda',comparison='Same core Phase-1 recipe; full 12-epoch allowance rather than its 900-second cutoff, zero workers for resumability. Hardware and libraries may differ.'))
    print('Device:',device,'| train:',len(known['train']),'| validation:',len(known['val_select']),'| calibration:',len(known['val_calibrate']),flush=True)
    print('Frozen ResNet-50, 224 px, Adam lr=0.001, up to 12 epochs. CPU execution can take substantially longer than GPU.',flush=True)
    resume=run/'last.pt'
    model=make_model(len(classes),root/'model_cache/huggingface',pretrained=not resume.exists()).to(device)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=CONFIG['lr_frozen'],weight_decay=0.)
    amp=device=='cuda';scaler=torch.amp.GradScaler('cuda',enabled=amp)
    generator=torch.Generator().manual_seed(CONFIG['seed'])
    data=DataLoader(pre.FrameDataset(known['train'],224,train=True),batch_size=32,shuffle=True,num_workers=0,generator=generator,pin_memory=amp)
    validation=DataLoader(pre.FrameDataset(known['val_select'],224),batch_size=32,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(42))
    history=[];best=-1.;stale=0;start_epoch=0
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=True)
        assert saved['config']==CONFIG and saved['classes']==classes
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer'])
        if amp and saved['scaler']:scaler.load_state_dict(saved['scaler'])
        history=saved['history'];best=saved['best'];stale=saved['stale'];start_epoch=saved['epoch']
        restore_rng(saved,generator);print('Resuming after completed epoch',start_epoch,flush=True)
    started=time.perf_counter()
    for epoch in range(start_epoch,CONFIG['epochs']):
        if stale>=CONFIG['patience']:break
        model.eval();model.classifier.train();total=correct=0;loss_sum=0.
        for batch,(x,y,_) in enumerate(data,1):
            x=x.to(device);y=y.to(device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp):
                logits=model(x);loss=nn.functional.cross_entropy(logits,y)
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            total+=len(y);correct+=int((logits.argmax(1)==y).sum());loss_sum+=float(loss.detach())*len(y)
            if batch%20==0:print(f'Epoch {epoch+1}/12 | batch {batch}/{len(data)}',flush=True)
        z,y=predict(model,validation,device);score=metrics(z,y,classes)
        row=dict(epoch=epoch+1,learning_rate=.001,train_loss=loss_sum/total,train_accuracy=correct/total,
            val_loss=score['nll'],val_accuracy=score['accuracy'],val_macro_f1=score['macro_f1_all_training_classes'],train_samples_seen=total,elapsed_seconds=time.perf_counter()-started)
        history.append(row);print(row,flush=True);pd.DataFrame(history).to_csv(run/'history.csv',index=False)
        if score['macro_f1_all_training_classes']>best:
            best=score['macro_f1_all_training_classes'];stale=0
            save_torch(run/'best.pt',dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                task='species',regime='frozen',classes=classes,image_size=224,epoch=epoch+1,config=CONFIG,
                source='Phase-2 species_recovery_v1 retraining; not the original Phase-1 checkpoint'))
        else:stale+=1
        save_torch(resume,dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),epoch=epoch+1,config=CONFIG,classes=classes,history=history,best=best,stale=stale,**capture_rng(generator)))
        print('Saved best.pt and resumable last.pt to Drive.',flush=True)
    saved=torch.load(run/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(saved['state_dict'])
    calibration=DataLoader(pre.FrameDataset(known['val_calibrate'],224),batch_size=32,shuffle=False,num_workers=0)
    z,y=predict(model,calibration,device);temperature=fit_temperature(z,y)
    policy=dict(temperature=temperature,calibration_split='val_calibrate',species_review_confidence=.6,
        checkpoint_sha256=digest(run/'best.pt'),version='species_recovery_v1',test_used_for_calibration=False)
    save_json(run/'policy.json',policy)
    before=metrics(z,y,classes);after=metrics(z,y,classes,temperature)
    save_json(run/'calibration_metrics.json',dict(before=before,after=after,temperature_bounds=[.05,20.]))
    # Fresh model reload validates the actual deployment architecture and saved logits.
    probe=make_model(len(classes),pretrained=False).to(device);probe.load_state_dict(saved['state_dict']);probe.eval();model.eval()
    x=next(iter(calibration))[0][:2].to(device)
    with torch.inference_mode():torch.testing.assert_close(model(x),probe(x))
    save_json(run/'checkpoint_reload.json',dict(passed=True,samples=len(x),checkpoint_sha256=digest(run/'best.pt')))
    summary=dict(version='species_recovery_v1',task='species',regime='frozen',best_epoch=saved['epoch'],classes=classes,
        selection_score=best,selection_split='val_select',train_n=len(known['train']),val_select_n=len(known['val_select']),val_calibrate_n=len(known['val_calibrate']),
        checkpoint_sha256=digest(run/'best.pt'),policy_sha256=digest(run/'policy.json'),temperature=temperature,
        calibration_ece_before=before['ece'],calibration_ece_after=after['ece'],test_evaluated=False,
        scope='New Phase-2 recovery deployment model; no claim of matching Phase-1 weights or metrics.')
    save_json(run/'summary.json',summary);install(root,run);print('TRAINING AND CALIBRATION COMPLETE',summary,flush=True)


def evaluate_test(root):
    root=Path(root);run,rows,classes,samples,known,pre=prepare(root)
    summary=json.loads((run/'summary.json').read_text());policy=json.loads((run/'policy.json').read_text())
    assert digest(run/'best.pt')==summary['checkpoint_sha256']==policy['checkpoint_sha256']
    assert digest(run/'policy.json')==summary['policy_sha256']
    eval_root=run/'historical_test'
    if (eval_root/'metrics.json').exists():
        previous=json.loads((eval_root/'metrics.json').read_text())
        assert previous['checkpoint_sha256']==summary['checkpoint_sha256'] and previous['policy_sha256']==summary['policy_sha256']
        print('Reusing fixed-policy historical test:',previous,flush=True);return
    eval_root.mkdir(exist_ok=True)
    verify_images(known['test'],eval_root,'test')
    save_json(eval_root/'protocol.json',dict(checkpoint_sha256=digest(run/'best.pt'),policy_sha256=digest(run/'policy.json'),
        manifest_sha256=MANIFEST_SHA,n_known=len(known['test']),n_unknown=len(samples['test'])-len(known['test']),independent_test=False,
        scope='Historical Phase-1 test split, already reported previously. No model, threshold or temperature selection here.'))
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    model=make_model(len(classes),pretrained=False).to(device)
    checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(checkpoint['state_dict']);model.eval()
    loader=DataLoader(pre.FrameDataset(known['test'],224),batch_size=32,shuffle=False,num_workers=0)
    z,y=predict(model,loader,device);results=metrics(z,y,classes,policy['temperature'])
    probability=torch.as_tensor(z/policy['temperature']).softmax(1).numpy();pred=probability.argmax(1)
    records=[dict(sample_id=r['sample_id'],id=r['id'],camera=r['camera'],event=r['event'],truth=classes[int(t)],predicted=classes[int(k)],confidence=float(p.max())) for r,t,k,p in zip(known['test'],y,pred,probability)]
    pd.DataFrame(records).to_csv(eval_root/'predictions.csv',index=False)
    np.savez_compressed(eval_root/'logits.npz',logits=z,y=y)
    results.update(split='historical_test',independent_test=False,checkpoint_sha256=digest(run/'best.pt'),policy_sha256=digest(run/'policy.json'),unknown_count=len(samples['test'])-len(known['test']))
    save_json(eval_root/'metrics.json',results)
    assert digest(run/'best.pt')==summary['checkpoint_sha256'] and digest(run/'policy.json')==summary['policy_sha256']
    print('HISTORICAL TEST COMPLETE',results,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--stage',choices=['train','test'],required=True)
    args=parser.parse_args();(train if args.stage=='train' else evaluate_test)(args.root)
