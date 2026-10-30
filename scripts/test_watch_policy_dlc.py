"""Exercise local completion gating and DLC deduplication without real submissions."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.test_encoder_starvla_monitor import FakeDLC
from scripts.watch_policy_dlc import LocalPolicyMonitor, local_status


class LocalPolicyMonitorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.remote = FakeDLC()
        self.statuses = {'Sparsh_mae': 'Succeeded', 'RAE': 'Succeeded', 'SD-VAE': 'Running'}
        records = []
        for method in self.statuses:
            source = self.folder / f'{method}.pth'
            source.write_bytes(b'checkpoint')
            records.append(dict(method=method, name='test_'+method, encoder_job_id='local:'+method,
                source_checkpoint=str(source), frozen_checkpoint=str(self.folder/'frozen'/f'{method}.pth'),
                output_dir=str(self.folder/'runs'/method), phase='WaitingEncoder',
                environment=dict(ENCODER_INIT='official', ENCODER_METHOD=method)))
        state = dict(status='Active', interval_seconds=600, source_root=str(self.folder/'source'), records=records, history=[])
        (self.folder/'state.json').write_text(json.dumps(state))
        self.monitor = LocalPolicyMonitor(self.folder, remote_call=self.remote, validator=lambda *args: {},
            readiness=lambda r: (self.statuses[r['method']], {}))

    def tearDown(self):
        self.temp.cleanup()

    def test_ready_jobs_first_then_sd_after_completion(self):
        first = self.monitor.tick()
        self.assertEqual(self.remote.submissions, 2)
        self.assertEqual(first['records'][2]['phase'], 'WaitingEncoder')
        self.monitor.tick()
        self.assertEqual(self.remote.submissions, 2)
        self.statuses['SD-VAE'] = 'Succeeded'
        final = self.monitor.tick()
        self.assertEqual(final['status'], 'Complete')
        self.assertEqual(self.remote.submissions, 3)
        self.monitor.tick()
        self.assertEqual(self.remote.submissions, 3)
        self.assertIn(str(self.folder/'source/training/train_starvla_groot.sh'), final['records'][2]['command'])

    def test_timeout_after_acceptance_reconciles_without_duplicate(self):
        self.remote.timeout_after_accept = True
        self.monitor.tick()
        self.monitor.tick()
        self.assertEqual(self.remote.submissions, 2)

    def test_failed_local_encoder_never_submits(self):
        self.statuses = {method: 'Failed' for method in self.statuses}
        self.monitor.tick()
        self.assertEqual(self.remote.submissions, 0)

    def test_missing_launcher_is_not_completion(self):
        (self.folder/'launch.json').write_text(json.dumps({'launcher_pid': 99999999}))
        status, _ = local_status(dict(local_encoder_run=str(self.folder)))
        self.assertEqual(status, 'Failed')
        (self.folder/'exit_code').write_text('0\n')
        self.assertEqual(local_status(dict(local_encoder_run=str(self.folder)))[0], 'Succeeded')


if __name__ == '__main__':
    unittest.main()
