"""Export and compare only the M4 detector pinned by the recovery-v2 release."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import torch
from ultralytics import YOLO
import onnxruntime as ort


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_parity(release, output_dir, threads=4):
    release, output_dir = Path(release), Path(output_dir)
    policy_path = release / 'refinement_policy.json'
    assert policy_path.is_file(), f'Missing installed recovery-v2 release: {policy_path}'
    policy = json.loads(policy_path.read_text())
    assert policy.get('version') == 'recovery_v2', 'Use the recovery-v2 release'
    weights = release / policy['detector_file']
    assert digest(weights) == policy['detector_sha256'], 'Detector checksum mismatch'
    output_dir.mkdir(parents=True, exist_ok=True)
    working_weights = output_dir / 'detector.pt'
    if working_weights.exists():
        assert digest(working_weights) == policy['detector_sha256'], 'Existing export checkpoint differs'
    else:
        shutil.copyfile(weights, working_weights)
    torch.set_num_threads(threads)
    detector = YOLO(str(working_weights))
    assert detector.task == 'detect' and detector.names == {0: 'animal'}
    detector.model.float().cpu().eval()
    size = int(policy['detector_input'])
    onnx_path = Path(detector.export(format='onnx', imgsz=size, dynamic=False,
        simplify=False, nms=False, opset=17, half=False, device='cpu'))
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=options,
        providers=['CPUExecutionProvider'])
    detector.model.float().cpu().eval()
    rng = np.random.default_rng(42)
    cases = []
    for name in ('zeros', 'constant', 'seeded_random'):
        shape = (1, 3, size, size)
        x = (np.zeros(shape, np.float32) if name == 'zeros' else
             np.full(shape, .5, np.float32) if name == 'constant' else
             rng.random(shape, dtype=np.float32))
        with torch.inference_mode():
            native = detector.model(torch.from_numpy(x))
            native = (native[0] if isinstance(native, (tuple, list)) else native).cpu().numpy()
        exported = session.run(None, {session.get_inputs()[0].name: x})[0]
        assert native.shape == exported.shape, f'Shape mismatch: {name}'
        assert np.isfinite(native).all() and np.isfinite(exported).all()
        np.testing.assert_allclose(native, exported, rtol=1e-3, atol=1e-3)
        cases.append(dict(input=name, shape=list(native.shape),
            max_abs_error=float(np.abs(native-exported).max()), passed=True))
    report = dict(scope='M4 detector raw forward outputs only; no NMS, classification, M5 or SAM parity claim',
        source_detector_sha256=policy['detector_sha256'], segmentation_policy_sha256=digest(policy_path),
        onnx_sha256=digest(onnx_path), onnx_path=str(onnx_path),
        imgsz=size, precision='float32', opset=17, dynamic=False, nms=False,
        providers=session.get_providers(), torch_threads=threads, ort_threads=threads,
        rtol=1e-3, atol=1e-3, passed=True, cases=cases,
        max_abs_error=max(c['max_abs_error'] for c in cases))
    (output_dir/'onnx_parity.json').write_text(json.dumps(report, indent=2))
    print('Detector ONNX parity: PASS')
    print('Saved:', output_dir/'onnx_parity.json')
    print('Maximum absolute error:', report['max_abs_error'])
    return detector, onnx_path, session, x, report
