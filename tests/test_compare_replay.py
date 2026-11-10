"""Reject incomparable reruns; expose numeric differences without claiming reproduction."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('compare_replay',Path(__file__).resolve().parents[1]/'tools/compare_replay.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class ReplayTests(unittest.TestCase):
    def test_comparison_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'a';b=root/'b'
            for base in (a,b):
                for task in ('triage','species'):
                    for regime in ('scratch','frozen','full'):
                        path=base/task/regime/'test_metrics.json';path.parent.mkdir(parents=True)
                        path.write_text(json.dumps(dict(context={'manifest_sha256':'fixture'},task=task,regime=regime,
                            split='test',image_size=32,n=8,accuracy=.75,macro_f1_all_training_classes=.7,
                            macro_f1_supported_classes=.7,ece=.1,
                            operating_point={'animal_recall':1.,'empty_removal_rate':.2},policy={'threshold':.4})))
            def run():return module.compare(a,b,'Synthetic tester','https://wandb.ai/fixture/project/runs/test','2026-10-28',root/'out')
            self.assertTrue(run()['all_metrics_match'])
            path=b/'species/frozen/test_metrics.json';data=json.loads(path.read_text());data['accuracy']=.5;path.write_text(json.dumps(data))
            self.assertFalse(run()['all_metrics_match'])
            data['context']['manifest_sha256']='different';path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):run()
            with self.assertRaises(ValueError):module.compare(a,a,'Tester','https://wandb.ai/x','2026-10-28',root/'out')

if __name__=='__main__':unittest.main()
