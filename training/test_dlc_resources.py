import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from training.dlc_resources import POOLS,capacity,preferred
from training.experiment_plan import Plan

def quota(total,allocated,submitted):
    return dict(Status='Ready',QuotaName='test',QuotaDetails=dict(
        AllocatableQuota=dict(GPU=str(total),CPU='1000',Memory='10000Gi'),
        AllocatedQuota=dict(GPU=str(allocated),CPU='100',Memory='100Gi'),
        SelfSubmittedQuota=dict(GPU=str(submitted),CPU='100',Memory='100Gi')))

class ResourceTest(unittest.TestCase):
    def test_preference_queued_requests_and_memory(self):
        a,b=[x[1] for x in POOLS]
        pools={a:quota(192,185,229),b:quota(48,33,33)}
        self.assertEqual(capacity(pools[a]),0)
        self.assertEqual(capacity(pools[b]),3)
        self.assertEqual(preferred(pools),b)
        pools[a]=quota(192,180,180);self.assertEqual(preferred(pools),a)
        pools[a]['QuotaDetails']['AllocatableQuota']['Memory']='200Gi'
        self.assertEqual(preferred(pools),b)

    def test_transfer_waits_for_confirmed_stop_and_preserves_running(self):
        with tempfile.TemporaryDirectory() as t:
            p=Plan.__new__(Plan);p.folder=Path(t);p.save=lambda:None;p.event=lambda *a,**k:None
            a,b=[x[1] for x in POOLS]
            out=Path(t);(out/'submission.json').write_text(json.dumps(['dlc','--resource_id',a,'--name','original']))
            job=dict(status='Submitted',cloud_status='Queuing',resource_id=a,job_id='dlcqueued',output=t,name='original')
            running=dict(job,status='Submitted',cloud_status='Running',job_id='dlcrunning')
            p.state=dict(jobs=dict(queued=job,running=running));statuses={'dlcqueued':'Queuing','dlcrunning':'Running'}
            p.cloud_status=lambda j:statuses[j['job_id']]
            stopped=[];p.stop_queued=lambda j:stopped.append(j['job_id'])
            with patch('training.experiment_plan.resource_snapshot',return_value={a:quota(192,192,196),b:quota(48,44,44)}):
                p.rebalance_resources();self.assertEqual(stopped,['dlcqueued'])
                self.assertEqual(job['status'],'Relocating');self.assertEqual(job['job_id'],'dlcqueued')
                statuses['dlcqueued']='Stopping';p.rebalance_resources();self.assertEqual(job['status'],'Relocating')
                statuses['dlcqueued']='Stopped';p.rebalance_resources()
                self.assertEqual(job['status'],'Prepared');self.assertNotIn('job_id',job)
                self.assertEqual(job['resource_id'],b);self.assertEqual(job['job_history'][0]['job_id'],'dlcqueued')
                self.assertTrue((out/'dlcqueued.submission.json').exists())
                self.assertEqual(running['job_id'],'dlcrunning');self.assertEqual(running['status'],'Submitted')
if __name__=='__main__':unittest.main()
