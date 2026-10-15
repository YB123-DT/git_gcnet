"""Guard train-only budget decisions and exact pilot reuse, with no GPU work."""
import json
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import dispatch
from runtime import atomic_json

class BudgetTests(unittest.TestCase):
    def fixture(self,root,b_better):
        tasks=[]
        for mr in (.3,.6):
            for epochs,stage in ((100,50),(200,100)):
                cfg=dict(model='MoMKE',model_version='causal',dataset='MELD',seed=66,missing_rate=mr,epochs=epochs,stage_epoch=stage)
                run=root/f'pilot_{mr}_{epochs}';run.mkdir()
                history=[dict(epoch=ep,train_loss=(.8 if b_better else 1.05) if epochs==200 else 1.,acc=0. if epochs==200 else 100.) for ep in range(stage+1,epochs+1)]
                (run/'history.jsonl').write_text('\n'.join(json.dumps(x) for x in history))
                tasks.append(dict(id=str(run),phase='budget_pilot',config=cfg,run_dir=str(run)))
        for version in ('causal','original'):
            for mr in (.3,.6):
                cfg=dict(model='MoMKE',model_version=version,dataset='MELD',seed=66,missing_rate=mr,epochs=200,stage_epoch=100)
                config_path=root/f'{version}_{mr}.json';atomic_json(config_path,cfg)
                tasks.append(dict(id=f'{version}_{mr}',phase='formal',config=cfg,config_path=str(config_path),run_dir=str(root/f'formal_{version}_{mr}')))
        atomic_json(root/'experiment_registry.json',dict(tasks=tasks))
        gates={f'MoMKE_{v}_MELD':dict(ready=False,smoke_passed=True) for v in ('causal','original')}
        gates['MoMKE_causal_MELD_pilot']=dict(ready=True)
        return tasks,gates

    def test_both_budget_choices_ignore_opposite_test_rankings_and_reuse_pilots(self):
        for better,expected in ((True,200),(False,100)):
            with self.subTest(better=better),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);tasks,gates=self.fixture(root,better)
                with patch.object(dispatch,'ROOT',root),patch.object(dispatch,'valid_result',return_value=True):
                    self.assertTrue(dispatch.budget_decision(tasks,gates))
                note=json.loads((root/'audit/momke_budget_decision.json').read_text())
                self.assertEqual(note['selected_epochs'],expected)
                actual=json.loads((root/'experiment_registry.json').read_text())['tasks']
                formal=[t for t in actual if t['phase']=='formal']
                self.assertTrue(all(t['config']['epochs']==expected for t in formal))
                for task in formal:
                    self.assertEqual(json.loads(Path(task['config_path']).read_text()),task['config'])
                    self.assertEqual('reused_from_budget_pilot' in task,task['config']['model_version']=='causal')
                self.assertTrue(all(gates[f'MoMKE_{v}_MELD']['ready'] for v in ('causal','original')))

    def test_unfinished_pilots_keep_formal_budget_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks,gates=self.fixture(Path(tmp),True)
            with patch.object(dispatch,'ROOT',Path(tmp)),patch.object(dispatch,'valid_result',return_value=False):
                self.assertFalse(dispatch.budget_decision(tasks,gates))
            self.assertFalse(gates['MoMKE_causal_MELD']['ready'])
            self.assertFalse((Path(tmp)/'audit/momke_budget_decision.json').exists())

if __name__=='__main__':unittest.main()
