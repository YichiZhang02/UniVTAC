import copy
import json
import tempfile
import unittest
from pathlib import Path
from training.eval_results import validated_complete
from training.eval_config import SETTINGS
from training.experiment_plan import Plan


class ShutdownRecoveryTest(unittest.TestCase):
    def result(self,task):
        return dict(context=dict(task=task,checkpoint='/model.pt',requested=50,
                    start_seed=1000000,step_lim=SETTINGS['task_step_limits'][task]),
                    rollouts=[dict(seed=1000000+i,status='success' if i<30 else 'failed') for i in range(50)],
                    valid=50,successes=30,complete=True)

    def test_reject_bad_records(self):
        r=self.result('lift_can');context=copy.deepcopy(r['context'])
        self.assertTrue(validated_complete(r,context))
        for change in ('duplicate','counter','checkpoint','incomplete'):
            bad=copy.deepcopy(r)
            if change=='duplicate':bad['rollouts'][1]['seed']=1000000
            if change=='counter':bad['successes']=31
            if change=='checkpoint':bad['context']['checkpoint']='/other.pt'
            if change=='incomplete':bad['rollouts'].pop()
            self.assertFalse(validated_complete(bad,context),change)

    def test_recover_two_queued_shards_then_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Plan.__new__(Plan);p.save=lambda:None;p.event=lambda *a,**kw:None
            shards={}
            for task in SETTINGS['task_step_limits']:
                out=Path(tmp)/task;out.mkdir();r=self.result(task)
                (out/f'{task}.result.json').write_text(json.dumps(r))
                failed=task in ('insert_HDMI','put_bottle_in_shelf')
                (out/'status.json').write_text(json.dumps(dict(status='Failed' if failed else 'Completed',exit_code=1 if failed else 0,tasks={task:r})))
                (out/f'{task}.log').write_text('Final Result: 30/50(60.00%) success.')
                shards[task]=dict(status='Planned' if failed else 'Running',output=str(out))
            e=dict(id='eval',job='job',status='Sharded',episodes=50,seed=1000000,output=tmp,shards=shards)
            p.state=dict(jobs={'job':dict(checkpoint='/model.pt')},evaluations={'eval':e})
            # Complete records without the simulator's final summary are insufficient.
            marker=Path(shards['insert_HDMI']['output'])/'insert_HDMI.log';marker.write_text('unfinished')
            p.poll_shards();self.assertEqual(e['status'],'Sharded')
            marker.write_text('Final Result: 30/50(60.00%) success.')
            p.poll_shards();self.assertEqual(e['status'],'Completed')
            self.assertAlmostEqual(e['result']['macro_success_rate'],.6)
            self.assertEqual(shards['insert_HDMI']['recovery']['worker_exit_code'],1)


if __name__=='__main__':unittest.main()
