import json,tempfile,unittest
from pathlib import Path
from training.experiment_plan import Plan
from training.lr_schedule import DEFAULTS
class PlanV2Test(unittest.TestCase):
    def test_groot_v1_starts_without_baseline_scores_and_no_old_control_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Plan.__new__(Plan);p.folder=Path(tmp);(p.folder/'configs').mkdir();p.save=lambda:None;p.event=lambda *a,**k:None
            base=json.loads(Path('training/config.json').read_text());base.update(learning_rate=1e-5,action_learning_rate=1e-4,lr_schedule='fixed')
            p.state=dict(base=base,plan_revision=2,jobs={},stages={},evaluations={},encoders={f'{s}/{v}':{} for s in ('S','B') for v in ('rae-1in1','rae-4in1')},status='Running')
            p.ensure_stage('baseline',[p.config(model=m,tactile=False) for m in ('starvla','starvla_pi')])
            p.ensure_stage('baseline_v2',[dict(p.config(model=m,tactile=False),**DEFAULTS,head_action_queries_only=m!='starvla_groot_v1') for m in ('starvla_groot_v1','starvla_pi')])
            def complete(stage):
                for i,key in enumerate(p.state['stages'][stage]):
                    j=p.state['jobs'][key];j.update(status='TrainingSucceeded',checkpoint='/'+key+'.pt')
                    if p.score(key) is not None:continue
                    e=p.evaluation(key);e.update(status='Completed',result=dict(tasks={str(t):dict(valid=50,successes=dict(vlm_B=20,vlm_C=30,vlm_D=40).get(stage,10)+i,complete=True) for t in range(8)}))
            p.advance();complete('vlm_A');p.advance();self.assertIn('vlm_B',p.state['stages'])
            self.assertTrue(all(p.score(k) is None for k in p.state['stages']['baseline_v2']))
            complete('baseline_v2');p.advance()
            self.assertEqual(len(p.state['stages']['vlm_B']),4)
            for key in p.state['stages']['vlm_B']:
                j=p.state['jobs'][key];self.assertEqual(j['model'],'starvla_groot_v1');self.assertEqual(j['config']['learning_rate'],2.5e-5)
                self.assertEqual(j['source_template'],str(p.folder/'frozen_source_v2'))
                self.assertNotIn(key,p.state['stages']['vlm_A'])
            for stage in ('vlm_B','vlm_C','vlm_D'):
                complete(stage);p.advance()
            self.assertEqual(p.state['jobs'][p.state['stages']['vlm_E'][0]]['model'],'starvla_pi')
            complete('vlm_E');complete('head');p.advance()
            self.assertEqual(len(p.state['jobs']),25)
            final=[e for e in p.state['evaluations'].values() if e['kind']=='final'];self.assertEqual(len(final),6)
            self.assertTrue(all(e['job'] not in p.state['stages']['baseline'] for e in final))
if __name__=='__main__':unittest.main()
