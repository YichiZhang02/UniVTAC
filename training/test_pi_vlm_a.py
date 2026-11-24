import itertools
import tempfile
import unittest
from pathlib import Path

from training.experiment_plan import Plan


class PIVLMAPlanTest(unittest.TestCase):
    def test_nine_independent_candidates_and_idempotent_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Plan.__new__(Plan)
            p.folder=Path(tmp)
            (p.folder/'configs').mkdir()
            p.save=lambda:None
            p.event=lambda *args,**kwargs:None
            p.state=dict(plan_revision=2,independent_vlm_selection=True,
                         independent_vlm_a_models=['pi'],encoders={'S/rae-1in1':{}},
                         stages={'vlm_B':['existing_groot_B']},evaluations={},
                         jobs={'pi_baseline':dict(stage='baseline_v2',model='starvla_pi',signature='baseline',
                             config=dict(model='starvla_pi',tactile=False,steps=50000,chunk=50)),
                               'existing_groot_B':dict(stage='vlm_B',model='starvla_groot_v1',
                             status='Submitted',signature='existing',config={})})
            p.advance()
            ids=p.state['stages']['vlm_A_pi']
            self.assertEqual(len(ids),9)
            configs=[p.state['jobs'][key]['config'] for key in ids]
            self.assertEqual({(c['tactile_wrapper'],c['tactile_rope']) for c in configs},
                             set(itertools.product(('none','vision','tactile'),('none','1d','3d'))))
            for key,cfg in zip(ids,configs):
                self.assertEqual(cfg['model'],'starvla_pi')
                self.assertEqual(cfg['lr_schedule'],'cosine')
                self.assertEqual(cfg['learning_rate'],2.5e-5)
                self.assertTrue(cfg['tactile'])
                self.assertTrue(cfg['head_action_queries_only'])
                self.assertEqual(cfg['encoder_checkpoint'],str(p.encoder_path('S','rae-1in1')))
                self.assertEqual(p.state['jobs'][key]['source_template'],str(p.folder/'frozen_source_v2'))
                self.assertNotIn('resume_checkpoint',p.state['jobs'][key])
            p.advance()
            self.assertEqual(p.state['stages']['vlm_A_pi'],ids)
            self.assertEqual(len(p.state['jobs']),11)
            self.assertEqual(p.state['jobs']['existing_groot_B']['status'],'Submitted')
            self.assertNotIn('vlm_C',p.state['stages'])
            self.assertNotIn('vlm_B_pi',p.state['stages'])
            # Enabling B reuses the selected A checkpoint and creates only three jobs.
            p.state['winners']={'vlm_A_pi':ids[7]}
            p.state['independent_vlm_b_models']=['pi']
            p.advance()
            b_ids=p.state['stages']['vlm_B_pi']
            self.assertEqual(len(b_ids),4)
            self.assertIn(ids[7],b_ids)
            self.assertEqual(len(p.state['jobs']),14)
            for key in b_ids:
                cfg=p.state['jobs'][key]['config']
                self.assertEqual((cfg['tactile_wrapper'],cfg['tactile_rope']),('tactile','1d'))
                self.assertEqual(cfg['model'],'starvla_pi')
            p.advance()
            self.assertEqual(len(p.state['jobs']),14)
            self.assertNotIn('vlm_C_pi',p.state['stages'])
            p.state['winners']['vlm_B_pi']=b_ids[1]
            p.state['independent_vlm_c_models']=['pi']
            p.advance()
            self.assertNotIn('vlm_C_pi',p.state['stages'])
            p.state['encoders']={f'{size}/{variant}':{} for size,variant in
                itertools.product(('S','B'),('rae-1in1','rae-4in1'))}
            p.advance()
            c_ids=p.state['stages']['vlm_C_pi']
            self.assertEqual(len(c_ids),4)
            self.assertIn(b_ids[1],c_ids)
            self.assertEqual(len(p.state['jobs']),17)
            for key in c_ids:
                cfg=p.state['jobs'][key]['config']
                self.assertTrue(cfg['freeze_tactile_encoder'])
                self.assertEqual(cfg['tactile_tokens'],'pool3x3')
                self.assertEqual(cfg['tactile_frames'],1 if cfg['rae_variant']=='rae-1in1' else 4)
                self.assertEqual(cfg['encoder_checkpoint'],str(p.encoder_path(cfg['rae_size'],cfg['rae_variant'])))
            p.advance()
            self.assertEqual(len(p.state['jobs']),17)
            self.assertNotIn('vlm_D_pi',p.state['stages'])
            p.state['winners']['vlm_C_pi']=c_ids[2]
            p.state['independent_vlm_d_models']=['pi']
            p.advance()
            d_ids=p.state['stages']['vlm_D_pi']
            self.assertEqual(len(d_ids),3)
            self.assertIn(c_ids[2],d_ids)
            self.assertEqual(len(p.state['jobs']),19)
            self.assertEqual({p.state['jobs'][k]['config']['vlm_attention_mask'] for k in d_ids},
                             {'causal','modality','bidirectional'})
            reference=p.state['jobs'][c_ids[2]]['config']
            for key in d_ids:
                cfg=p.state['jobs'][key]['config']
                self.assertEqual({k:v for k,v in cfg.items() if k!='vlm_attention_mask'},
                                 {k:v for k,v in reference.items() if k!='vlm_attention_mask'})
            p.advance()
            self.assertEqual(len(p.state['jobs']),19)
            self.assertNotIn('head',p.state['stages'])


if __name__=='__main__':unittest.main()
