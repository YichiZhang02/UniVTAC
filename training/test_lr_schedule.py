import copy,unittest
from training.lr_schedule import DEFAULTS,factor,build
class ScheduleTest(unittest.TestCase):
    def test_boundaries_and_floor(self):
        cfg=dict(DEFAULTS,steps=50000)
        self.assertAlmostEqual(factor(0,cfg),1/1001)
        self.assertGreater(factor(999,cfg),factor(10,cfg))
        self.assertGreater(factor(1000,cfg),factor(15000,cfg))
        self.assertAlmostEqual(factor(30000,cfg),.1)
        self.assertAlmostEqual(factor(50000,cfg),.1)
    def test_optimizer_and_scheduler_resume(self):
        import torch
        cfg=dict(DEFAULTS,steps=50000)
        a=torch.nn.Parameter(torch.ones(1));opt=torch.optim.AdamW([a],lr=cfg['learning_rate']);sch=build(opt,cfg)
        for _ in range(12):a.grad=torch.ones_like(a);opt.step();sch.step()
        opt_state=copy.deepcopy(opt.state_dict());schedule_state=copy.deepcopy(sch.state_dict())
        b=torch.nn.Parameter(a.detach().clone());other=torch.optim.AdamW([b],lr=cfg['learning_rate']);other_sch=build(other,cfg)
        other.load_state_dict(opt_state);other_sch.load_state_dict(schedule_state)
        for _ in range(5):
            for param,optimizer,scheduler in [(a,opt,sch),(b,other,other_sch)]:
                param.grad=torch.ones_like(param);optimizer.step();scheduler.step()
            self.assertTrue(torch.equal(a,b));self.assertEqual(opt.param_groups[0]['lr'],other.param_groups[0]['lr'])
if __name__=='__main__':unittest.main()
