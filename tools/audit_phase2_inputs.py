"""Read-only audit of a recovered Phase 1 workspace; no training or pickle loading."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(workspace):
    workspace = workspace.resolve()
    checks = []

    def check(name, passed, detail):
        checks.append(dict(name=name, passed=bool(passed), detail=detail))

    def resolve(relative):
        relative = Path(relative)
        # Local team images were recorded with this original Colab prefix.
        # Rebase only that known prefix; preserve the frozen manifest bytes.
        colab_root = Path('/content/CameraTrap_Phase1_Team_Mixed_v2')
        if relative.is_absolute() and relative.is_relative_to(colab_root):
            relative = relative.relative_to(colab_root)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Expected workspace-relative path')
        result = (workspace / relative).resolve()
        if not result.is_relative_to(workspace):
            raise ValueError('Path escapes workspace')
        return result

    summary = json.loads((REPO / 'results/species/frozen/summary.json').read_text())
    expected_manifest = summary['context']['manifest_sha256']
    release_path = workspace / 'handoff/dataset_release.json'
    check('dataset_release_present', release_path.is_file(), 'handoff/dataset_release.json')
    rows = []
    if release_path.is_file():
        release = json.loads(release_path.read_text())
        check('release_contract', release.get('schema') == 2 and release.get('audit_complete') is True,
              'schema=2 and audit_complete=true required')
        check('release_files_nonempty', bool(release.get('files')), 'Frozen file hashes required')
        for relative, expected in release.get('files', {}).items():
            path = resolve(relative)
            check('release_hash:' + relative, path.is_file() and sha256(path) == expected,
                  'SHA-256 must match frozen release')
        manifest = resolve(release['data_dir']) / 'manifest.json'
        check('manifest_matches_phase1', manifest.is_file() and sha256(manifest) == expected_manifest,
              expected_manifest)
        if manifest.is_file():
            rows = json.loads(manifest.read_text())
            check('manifest_nonempty', bool(rows), 'Frozen image records required')
            check('unique_ids', len({r['id'] for r in rows}) == len(rows), len(rows))
            check('expected_splits', {r['split'] for r in rows} == {'train', 'val_select', 'val_calibrate', 'test'},
                  dict(collections.Counter(r['split'] for r in rows)))
            for field in ('camera', 'event', 'pixel_sha256'):
                groups = collections.defaultdict(set)
                for row in rows:
                    if row.get(field):
                        groups[row[field]].add(row['split'])
                complete = all(row.get(field) for row in rows)
                check(field + '_disjoint', complete and all(len(s) == 1 for s in groups.values()),
                      'Required on every row; no cross-split overlap')
            missing = [r['id'] for r in rows if not resolve(r['path']).is_file()]
            check('images_present', not missing, dict(missing_count=len(missing), example_ids=missing[:10]))
            mismatched = []
            for row in rows:
                path = resolve(row['path'])
                if path.is_file():
                    actual = hashlib.md5(path.read_bytes()).hexdigest()
                    if actual != row.get('processed_file_md5'):
                        mismatched.append(row['id'])
            check('present_image_hashes', not mismatched,
                  dict(mismatch_count=len(mismatched), example_ids=mismatched[:10]))
    for task in ('triage', 'species'):
        saved = json.loads((REPO / f'results/{task}/frozen/summary.json').read_text())
        relative = f'runs/member3/{task}/frozen/best.pt'
        checkpoint = workspace / relative
        check(task + '_checkpoint_hash', checkpoint.is_file() and sha256(checkpoint) == saved['checkpoint_sha256'],
              dict(path=relative, expected_sha256=saved['checkpoint_sha256']))
        policy = workspace / f'runs/member3/{task}/frozen/policy.json'
        expected = json.loads((REPO / f'results/{task}/frozen/policy.json').read_text())
        check(task + '_policy', policy.is_file() and json.loads(policy.read_text()) == expected,
              'Must match recorded Phase 1 calibration policy')
    check('preprocessing_hash', sha256(REPO / 'preprocessing.py') == summary['context']['preprocessing_sha256'],
          summary['context']['preprocessing_sha256'])
    candidates = collections.Counter()
    for row in rows:
        # Incomplete positives must never become empty detection targets.
        kind = ('empty' if not row['animal'] else
                'needs_box_review' if row.get('incomplete_boxes', True) or not row.get('boxes') else
                'boxed_positive')
        candidates[row['split'] + ':' + kind] += 1
    return dict(workspace=str(workspace), ready_for_next_step=all(c['passed'] for c in checks),
                checks=checks, detection_candidate_counts=dict(candidates),
                note='This checks release identity and file presence, not image decoding or visual annotation correctness.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True,
                        help='Recovered CameraTrap_Phase1_Team_Mixed_v2 folder')
    parser.add_argument('--output', type=Path, help='Optional JSON audit report')
    args = parser.parse_args()
    report = audit(args.workspace)
    content = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + '\n')
    print(content)
    raise SystemExit(0 if report['ready_for_next_step'] else 1)


if __name__ == '__main__':
    main()
