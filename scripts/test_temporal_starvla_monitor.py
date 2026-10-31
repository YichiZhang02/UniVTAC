"""Per-encoder readiness and real-validation gating without paid submissions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import signal

from scripts.test_encoder_starvla_monitor import FakeDLC
from scripts.watch_temporal_starvla import TemporalPolicyMonitor, parse_blocks, temporal_status, watch_lease
from scripts.watch_encoder_starvla import atomic_json

ROOT = Path(__file__).resolve().parents[1]


class TemporalMonitorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.remote = FakeDLC()
        self.validated = []
        self.jobs, self.records = [], []
        for method, environment in parse_blocks(ROOT / '2.sh').items():
            output = self.folder / 'encoders' / method
            output.mkdir(parents=True)
            (output / 'encoder.pth').write_bytes(b'test checkpoint')
            self.jobs.append(dict(method=method, output_dir=str(output), state='running'))
            self.records.append(dict(method=method, name='test_'+method, phase='WaitingEncoder',
                encoder_job_id='local:'+method, encoder_status_file=str(self.folder / 'encoders.json'),
                source_checkpoint=str(output / 'encoder.pth'),
                frozen_checkpoint=str(self.folder / 'frozen' / method / 'encoder.pth'),
                output_dir=str(self.folder / 'runs' / method), environment=environment))
        self.save_jobs()
        atomic_json(self.folder / 'state.json', dict(status='Active', interval_seconds=3600,
            source_root=str(self.folder / 'source'), history=[], records=self.records))
        self.monitor = TemporalPolicyMonitor(self.folder, remote_call=self.remote,
            validator=lambda *args: {'temporal_backend': 'conv3d'}, local_validator=self.validate)

    def tearDown(self):
        self.temp.cleanup()

    def save_jobs(self):
        atomic_json(self.folder / 'encoders.json', dict(jobs=self.jobs))

    def complete(self, method):
        next(job for job in self.jobs if job['method'] == method).update(state='complete', returncode=0)
        self.save_jobs()

    def validate(self, folder, state, record):
        self.validated.append(record['method'])
        return dict(passed=True, method=record['method'], conv3d_gradient_finite=True,
                    conv3d_gradient_nonzero=True)

    def test_existing_partial_checkpoint_is_not_completion(self):
        self.assertEqual(temporal_status(self.records[0])[0], 'Running')
        self.monitor.tick()
        self.assertEqual(self.remote.submissions, 0)
        self.assertEqual(self.validated, [])

    def test_one_encoder_validates_and_submits_without_waiting_for_others(self):
        self.complete('RVAE4Frame')
        state = self.monitor.tick()
        self.assertEqual(self.validated, ['RVAE4Frame'])
        self.assertEqual(self.remote.submissions, 1)
        self.assertEqual(sum(r['phase'] == 'Submitted' for r in state['records']), 1)
        self.assertEqual(sum(r['phase'] == 'WaitingEncoder' for r in state['records']), 7)
        self.monitor.tick()
        self.assertEqual(self.remote.submissions, 1)
        self.assertEqual(self.validated, ['RVAE4Frame'])

    def test_local_failure_prevents_submission_then_retries(self):
        self.complete('SD-VAE4Frame')
        def fail(*args):
            raise RuntimeError('Invalid Conv3d gradient')
        self.monitor.local_validator = fail
        state = self.monitor.tick()
        record = next(r for r in state['records'] if r['method'] == 'SD-VAE4Frame')
        self.assertEqual(record['phase'], 'LocalValidationFailed')
        self.assertEqual(self.remote.submissions, 0)
        self.monitor.local_validator = self.validate
        self.monitor.tick()
        self.assertEqual(self.remote.submissions, 1)

    def test_waiting_for_gpu_is_not_a_validation_pass(self):
        self.complete('ResNet4Frame')
        self.monitor.local_validator = lambda *args: None
        state = self.monitor.tick()
        self.assertEqual(state['records'][0]['phase'], 'WaitingGPU')
        self.assertEqual(self.remote.submissions, 0)

    def test_response_lost_after_acceptance_reconciles_without_revalidation(self):
        self.complete('ResNet4Frame')
        self.remote.timeout_after_accept = True
        self.monitor.tick()
        self.monitor.tick()
        self.assertEqual(self.remote.submissions, 1)
        self.assertEqual(self.validated, ['ResNet4Frame'])

    def test_failed_encoder_and_wrong_checkpoint_cannot_submit(self):
        self.jobs[0].update(state='failed', returncode=1)
        self.save_jobs()
        state = self.monitor.tick()
        self.assertEqual(state['records'][0]['phase'], 'EncoderFailed')
        self.complete('ResNet4Frame')
        self.monitor.validate = lambda *args: (_ for _ in ()).throw(ValueError('Wrong checkpoint'))
        self.monitor.tick()
        self.assertEqual(self.remote.submissions, 0)
        self.assertEqual(self.validated, [])

    def test_all_eight_complete_and_no_duplicate_after_restart(self):
        for job in self.jobs:
            self.complete(job['method'])
        state = self.monitor.tick()
        self.assertEqual(state['status'], 'Complete')
        restarted = TemporalPolicyMonitor(self.folder, remote_call=self.remote,
            validator=lambda *args: {}, local_validator=self.validate)
        restarted.tick()
        self.assertEqual(self.remote.submissions, 8)
        self.assertEqual(len(self.validated), 8)

    def test_watchdog_resumes_only_the_original_scheduler(self):
        lease = self.folder / 'lease.json'
        atomic_json(lease, dict(monitor_pid=10, monitor_key='monitor',
                               scheduler_pid=20, scheduler_key='original', expires=0))
        module = 'scripts.watch_temporal_starvla'
        with patch(module+'.process_key', side_effect=lambda pid: 'original' if pid == 20 else None), \
                patch(module+'.os.kill') as kill:
            watch_lease(lease)
            kill.assert_called_once_with(20, signal.SIGCONT)
        with patch(module+'.process_key', return_value='reused'), patch(module+'.os.kill') as kill:
            watch_lease(lease)
            kill.assert_not_called()


if __name__ == '__main__':
    unittest.main()
