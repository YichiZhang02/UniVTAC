import json,tempfile,unittest
from pathlib import Path
from training.experiment_plan import Plan,LR_DEFAULTS
class PiCGridTest(unittest.TestCase):
    def test_six_new_runs_and_two_existing_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Plan.__new__(Plan);p.folder=Path(tmp);(p.folder/'configs').mkdir();p.save=lambda:None;p.event=lambda *a,**k:None
            c=json.loads(Path('training/config.json').read_text());c.update(model='starvla_pi',tactile=False)
            p.state=dict(base=c,plan_revision=2,independent_vlm_selection=True,independent_vlm_a_models=['pi'],independent_vlm_b_models=['pi'],independent_vlm_c_models=['pi'],pi_c_freeze_grid=[False,True],jobs={},stages={},evaluations={},encoders={f'{s}/{v}':{} for s in ('S','B') for v in ('rae-1in1','rae-4in1')})
            p.ensure_stage('baseline_v2',[c]);p.advance()
            a=p.state['stages']['vlm_A_pi'];chosen=next(k for k in a if p.state['jobs'][k]['config']['tactile_wrapper']=='tactile' and p.state['jobs'][k]['config']['tactile_rope']=='1d')
            p.state['winners']={'vlm_A_pi':chosen};p.advance();b=p.state['stages']['vlm_B_pi'];p.state['winners']['vlm_B_pi']=chosen
            before=set(p.state['jobs']);p.advance();ids=p.state['stages']['vlm_C_pi']
            self.assertEqual(len(ids),8);self.assertEqual(len(set(ids)-before),6)
            controls=[k for k in ids if p.state['jobs'][k]['config']['rae_size']=='S' and p.state['jobs'][k]['config']['rae_variant']=='rae-1in1']
            self.assertEqual(len(controls),2);self.assertTrue(all(k in b for k in controls))
            p.advance();self.assertEqual(len(set(p.state['jobs'])-before),6)
if __name__=='__main__':unittest.main()
