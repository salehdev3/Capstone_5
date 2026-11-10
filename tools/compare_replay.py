"""Compare original and independently rerun result trees without training or altering them."""
import argparse
import csv
import hashlib
import json
import math
from datetime import date
from pathlib import Path


def compare(original, replay, reviewer, run_url, replay_date, output, tolerance=1e-6):
    original, replay, output = map(Path, (original, replay, output))
    if original.resolve() == replay.resolve():
        raise ValueError('Original and replay must be different result trees')
    if not reviewer.strip() or not run_url.startswith('https://wandb.ai/'):
        raise ValueError('Provide the actual reviewer and W&B replay URL')
    date.fromisoformat(replay_date)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Tolerance must be finite and nonnegative')
    rows, sources = [], []
    for task in ('triage', 'species'):
        for regime in ('scratch', 'frozen', 'full'):
            paths = [base / task / regime / 'test_metrics.json' for base in (original, replay)]
            a, b = [json.loads(p.read_text()) for p in paths]
            # Check semantic compatibility, never compare numbers from a different dataset/config.
            for field in ('context', 'task', 'regime', 'split', 'image_size', 'n'):
                if field not in a or field not in b or a[field] != b[field]:
                    raise ValueError(f'{task}/{regime}: incompatible {field}')
            metrics = ['accuracy', 'macro_f1_all_training_classes', 'macro_f1_supported_classes', 'ece']
            if task == 'triage':
                metrics += ['operating_point.animal_recall', 'operating_point.empty_removal_rate', 'policy.threshold']
            for metric in metrics:
                x, y = a, b
                for key in metric.split('.'):
                    x, y = x[key], y[key]
                x, y = float(x), float(y)
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError(f'Non-finite metric: {metric}')
                rows.append(dict(task=task, regime=regime, metric=metric, original=x,
                                 replay=y, delta=y-x, matches=abs(x-y)<=tolerance))
            sources.append(dict(task=task, regime=regime,
                                original_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),
                                replay_sha256=hashlib.sha256(paths[1].read_bytes()).hexdigest()))
    output.mkdir(parents=True, exist_ok=True)
    with (output/'replay_comparison.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    evidence=dict(reviewer=reviewer,date=replay_date,run_url=run_url,tolerance=tolerance,
                  all_metrics_match=all(r['matches'] for r in rows),sources=sources,
                  scope='Numerical comparison of supplied result trees. Operator identity and fresh training require the linked run records.')
    (output/'replay_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return evidence

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('original','replay','reviewer','run-url','replay-date','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--tolerance',type=float,default=1e-6)
    args=vars(p.parse_args())
    result=compare(**args)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['all_metrics_match'] else 1)
