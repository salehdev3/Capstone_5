from pathlib import Path
import subprocess, sys, hashlib, json, urllib.request, importlib.util

ROOT = Path(globals().get('ROOT', '/content/drive/MyDrive/CameraTrap_Phase2'))
OUT = Path(globals().get('OUT', ROOT/'phase2_results'))
RELEASE = ROOT/'releases/Segmentation_M4_Guided_Recovery_v2'
INTEGRATION_OUT = OUT/'member4_recovery_v2'
INTEGRATION_OUT.mkdir(parents=True, exist_ok=True)
SPECIES_PT = Path(globals().get('SPECIES_PT', ROOT/'models/species/best.pt'))
SPECIES_POLICY = Path(globals().get('SPECIES_POLICY', ROOT/'models/species/policy.json'))

# Look only in known Phase-1 run folders; never substitute detection weights.
if not SPECIES_PT.is_file():
    original_root = ROOT.parent/'CameraTrap_Phase1_Team_Mixed_v2'
    for regime in ['frozen','full']:
        candidate = original_root/'runs/member3/species'/regime/'best.pt'
        if candidate.is_file():
            SPECIES_PT = candidate
            if (candidate.parent/'policy.json').is_file():
                SPECIES_POLICY = candidate.parent/'policy.json'
            break
sha256 = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
KNOWN_PHASE1_CHECKPOINTS = {'644b0d485b7fdc3888f86fa729476bbd9a02252527e448f6aa25b10fcaebc780': {'regime': 'frozen', 'policy_sha256': 'cd9ded359a2e5a0b3c5c584ff6f471c2ce7dabc72996b61bbcac16b8f557b7a7'}, '79d7510746a92f505857dc98ee7301ffc0e9c416ac6ea51c72d88d7685f430ae': {'regime': 'full', 'policy_sha256': '52fa37d0214164385a253c73e1ddf6bda0851f79b0c8978cde55637186281ae2'}}
if SPECIES_PT.is_file() and not SPECIES_POLICY.is_file():
    matching = KNOWN_PHASE1_CHECKPOINTS.get(sha256(SPECIES_PT))
    if matching:
        SPECIES_POLICY.parent.mkdir(parents=True, exist_ok=True)
        temp = SPECIES_POLICY.with_suffix('.download')
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/salehdev3/Capstone_5/0893b6f/results/species/'+matching['regime']+'/policy.json', temp)
        assert sha256(temp) == matching['policy_sha256'], 'Classifier policy checksum mismatch'
        temp.replace(SPECIES_POLICY)
print('Species checkpoint:', SPECIES_PT)
print('Species policy:', SPECIES_POLICY)
if not SPECIES_PT.is_file() or not SPECIES_POLICY.is_file():
    status = dict(ready=False, missing_files=[str(p) for p in [SPECIES_PT,SPECIES_POLICY] if not p.is_file()],
                  reason='Trained Phase-1 species weights and matching calibration are required; no species predictions have been produced.')
    (INTEGRATION_OUT/'integration_readiness.json').write_text(json.dumps(status, indent=2))
    raise FileNotFoundError('Missing trained species classifier. Restore best.pt and its matching policy.json in ROOT/models/species/, or set SPECIES_PT and SPECIES_POLICY to their actual Drive paths before rerunning. Detection best.pt cannot replace species best.pt. Missing: '+', '.join(status['missing_files']))

subprocess.run([sys.executable,'-m','pip','install','-q',
                'ultralytics==8.3.221','transformers==4.49.0'], check=True)
MODULE = INTEGRATION_OUT/'member4_integration_v2.py'
urllib.request.urlretrieve(
    'https://raw.githubusercontent.com/salehdev3/Capstone_5/phase-2/tools/member4_integration_v2.py', MODULE)
assert sha256(MODULE) == 'c6e24e8c3c5e29fd796e338e612b74745e04403e6b64feacaceef67eeaaf0e4d', "Integration module checksum mismatch"
spec = importlib.util.spec_from_file_location('member4_integrated', MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
pipeline = module.IntegratedPipeline(RELEASE, SPECIES_PT, SPECIES_POLICY, INTEGRATION_OUT)
process_frame = pipeline.process_frame
process_batch = pipeline.process_batch
classify_crop = pipeline.classify_crop
runtime_detector = pipeline.detector
classifier = pipeline.classifier
classes = pipeline.classes
species_policy = pipeline.species_policy
(OUT/'integration_provenance.json').write_text(
    (INTEGRATION_OUT/'integration_provenance.json').read_text())
print('Integrated pipeline initialized: PyTorch M4 + Phase-1 species + Recovery v2 masks.')
print('Next: HTTP batch service and live request. No inference metrics are claimed by initialization.')
