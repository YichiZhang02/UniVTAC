import json,tempfile,unittest
from pathlib import Path
from training.experiment_plan import Plan
from training.eval_results import load,append


class PlanTest(unittest.TestCase):
    def test_all_stages_reuse_to_22_and_final_six(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan=Plan.__new__(Plan);plan.folder=Path(tmp);(plan.folder/'configs').mkdir()
            base=json.loads(Path('training/config.json').read_text())
            base.update(encoder='rae',rae_size='S',rae_variant='rae-1in1',tactile_frames=1,tactile_tokens='pool3x3',encoder_checkpoint=None)
            plan.state=dict(base=base,jobs={},evaluations={},stages={},encoders={f'{s}/{v}':{} for s in ('S','B') for v in ('rae-1in1','rae-4in1')},status='Running')
            plan.save=lambda:None;plan.event=lambda *args,**kwargs:None
            plan.ensure_stage('baseline',[plan.config(model=m,tactile=False) for m in ('starvla','starvla_pi')])
            tasks=('grasp_classify','insert_HDMI','insert_hole','insert_tube','lift_bottle','lift_can','pull_out_key','put_bottle_in_shelf')
            def complete(stage):
                ids=plan.state['stages'][stage]
                for i,key in enumerate(ids):
                    job=plan.state['jobs'][key];job.update(status='TrainingSucceeded',checkpoint='/tmp/'+key+'.pt')
                    e=plan.evaluation(key)
                    if e['status']=='Completed':continue
                    successes=10+i
                    # Ensure the newly created configuration beats the reused reference.
                    if stage=='vlm_B':successes=30+i
                    elif stage=='vlm_C':successes=35+i
                    elif stage=='vlm_D':successes=43+i
                    elif stage in ('head','vlm_E'):successes=40+i
                    e.update(status='Completed',result=dict(tasks={t:dict(successes=successes,valid=50,complete=True) for t in tasks}))
            complete('baseline');plan.advance()
            for stage in ('vlm_A','vlm_B','vlm_C','vlm_D','vlm_E','head'):
                self.assertIn(stage,plan.state['stages']);complete(stage);plan.advance()
            self.assertEqual(len(plan.state['jobs']),22)
            final=[e for e in plan.state['evaluations'].values() if e['kind']=='final']
            self.assertEqual(len(final),6)
            for e in final:e['status']='Completed'
            plan.advance();self.assertEqual(plan.state['status'],'Completed')
            selected=plan.state['winners']['vlm_A']
            # Later reuse/tie rounds must not retroactively change the selected stage.
            self.assertEqual(plan.winner('vlm_A'),selected)

    def test_error_exclusion_resume_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'result.json';context=dict(task='lift_can',requested=2)
            result=load(p,context);append(p,result,1,'error');append(p,result,2,'success')
            resumed=load(p,context);self.assertEqual(resumed['valid'],1)
            append(p,resumed,3,'failed');self.assertTrue(resumed['complete']);self.assertEqual(resumed['successes'],1)
            with self.assertRaises(ValueError):load(p,dict(context,requested=3))
            with self.assertRaises(ValueError):append(p,resumed,3,'success')


if __name__=='__main__':unittest.main()
