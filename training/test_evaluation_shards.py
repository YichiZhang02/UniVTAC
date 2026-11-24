import json,tempfile,unittest
from pathlib import Path
from training.experiment_plan import Plan
from training.eval_config import SETTINGS

class ShardTest(unittest.TestCase):
    def test_wait_for_all_tasks_then_macro_average(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Plan.__new__(Plan);p.save=lambda:None;p.event=lambda *a,**k:None
            shards={}
            for index,task in enumerate(SETTINGS['task_step_limits']):
                out=Path(tmp)/task;out.mkdir()
                result=dict(complete=True,valid=50,successes=index,context=dict(checkpoint='/model40.pt',step_lim=SETTINGS['task_step_limits'][task]))
                (out/'status.json').write_text(json.dumps(dict(status='Completed',tasks={task:result})))
                shards[task]=dict(status='Failed' if index==0 else 'Running',output=str(out))
            last=list(shards)[-1];(Path(shards[last]['output'])/'status.json').unlink()
            e=dict(status='Sharded',id='eval',job='job',output=tmp,shards=shards,checkpoint='/model40.pt',checkpoint_step=40000)
            p.state=dict(jobs={'job':dict(checkpoint='/model.pt')},evaluations={'eval':e})
            p.poll_shards();self.assertEqual(e['status'],'Sharded')
            shards[last].update(status='Completed',result=dict(complete=True,valid=50,successes=7,context=dict(checkpoint='/model40.pt')))
            p.poll_shards();self.assertEqual(e['status'],'Completed')
            self.assertEqual(e['result']['checkpoint'],'/model40.pt');self.assertEqual(e['result']['checkpoint_step'],40000)
            self.assertAlmostEqual(e['result']['macro_success_rate'],sum(range(8))/400)
if __name__=='__main__':unittest.main()
