"""Adapted encoder completion and local-test gating without real DLC jobs."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.test_encoder_starvla_monitor import FakeDLC
from scripts.watch_adapted_starvla import AdaptedPolicyMonitor, encoder_status, parse_blocks, validate_checkpoint
from scripts.watch_encoder_starvla import atomic_json

ROOT = Path(__file__).resolve().parents[1]


class AdaptedMonitorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.remote = FakeDLC()
        self.validated = []
        self.rows, self.records = [], []
        for method, environment in parse_blocks(ROOT / '3.sh').items():
            output = self.folder / 'encoders' / method
            output.mkdir(parents=True)
            (output / 'encoder.pth').write_bytes(b'test weights')
            self.rows.append(dict(method=method, output_dir=str(output), initialization='adapted',
                                  status='Prepared', command='test adapted training'))
            self.records.append(dict(method=method, name='policy_'+method, phase='WaitingEncoder',
                encoder_job_id='pending:'+method, encoder_ledger=str(self.folder / 'encoders.json'),
                source_checkpoint=str(output / 'encoder.pth'),
                frozen_checkpoint=str(self.folder / 'frozen' / method / 'encoder.pth'),
                output_dir=str(self.folder / 'runs' / method), environment=environment))
        self.save_rows()
        atomic_json(self.folder / 'state.json', dict(status='Active', interval_seconds=3600,
            source_root=str(self.folder / 'source'), records=self.records, history=[]))
        self.monitor = AdaptedPolicyMonitor(self.folder, remote_call=self.remote,
            validator=lambda *args: {'initialization': 'adapted'}, local_validator=self.validate)

    def tearDown(self):
        self.temp.cleanup()

    def save_rows(self):
        atomic_json(self.folder / 'encoders.json', self.rows)

    def complete(self, method):
        row = next(row for row in self.rows if row['method'] == method)
        row.update(job_id='encoder-'+method, status='Submitted')
        self.remote.encoder_status[row['job_id']] = 'Succeeded'
        self.save_rows()

    def validate(self, folder, state, record):
        self.validated.append(record['method'])
        return dict(passed=True, method=record['method'], encoder_gradient_finite=True,
                    encoder_gradient_nonzero=True)

    def test_prepared_encoder_and_partial_checkpoint_do_not_submit(self):
        self.monitor.tick()
        self.assertEqual(self.remote.submissions, 0)
        self.assertEqual(self.validated, [])

    def test_one_completed_encoder_triggers_only_its_policy(self):
        self.complete('Sparsh_mae')
        state = self.monitor.tick()
        self.assertEqual(self.remote.submissions, 1)
        self.assertEqual(self.validated, ['Sparsh_mae'])
        self.assertEqual(sum(r['phase'] == 'WaitingEncoder' for r in state['records']), 4)
        command = state['records'][0]['command']
        self.assertIn('ENCODER_PRETRAIN_INIT=adapted', command)
        self.assertIn('TACTILE_TYPE=pool3x3', command)
        self.monitor.tick()
        self.assertEqual(self.remote.submissions, 1)

    def test_failed_encoder_and_invalid_initialization_are_blocked(self):
        self.complete('Sparsh_DINOv2')
        self.remote.encoder_status['encoder-Sparsh_DINOv2'] = 'Stopped'
        self.monitor.tick()
        self.assertEqual(self.remote.submissions, 0)
        self.rows[0]['initialization'] = 'scratch'
        self.save_rows()
        with self.assertRaisesRegex(ValueError, 'adapted'):
            encoder_status(self.records[0], self.remote)

    def test_local_test_failure_prevents_submission(self):
        self.complete('Anytouch2')
        def fail(*args):
            raise RuntimeError('Nonfinite local policy loss')
        self.monitor.local_validator = fail
        state = self.monitor.tick()
        self.assertEqual(self.remote.submissions, 0)
        self.assertEqual(state['records'][-1]['phase'], 'LocalValidationFailed')

    def test_accept_then_timeout_does_not_duplicate(self):
        self.complete('Sparsh_VJEPA')
        self.remote.timeout_after_accept = True
        self.monitor.tick()
        self.monitor.tick()
        self.assertEqual(self.remote.submissions, 1)
        self.assertEqual(self.validated, ['Sparsh_VJEPA'])

    def test_all_completed_exit_after_five_unique_submissions(self):
        for row in self.rows:
            self.complete(row['method'])
        state = self.monitor.tick()
        self.assertEqual(state['status'], 'Complete')
        self.assertEqual(self.remote.submissions, 5)
        self.monitor.tick()
        self.assertEqual(self.remote.submissions, 5)

    def test_anytouch2_rejects_the_old_training_objective(self):
        import torch
        checkpoint = self.folder / 'anytouch2.pth'
        metadata = dict(encoder={'weight': torch.zeros(1)}, method='Anytouch2', size='official',
            input_mode='depth_deform_nonlinear_radial', initialization='adapted', pretrained_mode='tactile',
            color_order='RGB', num_frames=4, frame_stride=2, image_size=[224, 224],
            epoch=1, step=1, val_loss=1., source_revision='official')
        torch.save(metadata, checkpoint)
        with self.assertRaisesRegex(ValueError, 'corrected video-MAE'):
            validate_checkpoint(checkpoint, {'method': 'Anytouch2'})
        metadata.update(training_objective='video_mae_masked_pixels_v1', mae_mask_ratio=.5,
                        mae_decoder_init='random')
        torch.save(metadata, checkpoint)
        self.assertEqual(validate_checkpoint(checkpoint, {'method': 'Anytouch2'})['mae_mask_ratio'], .5)

    def test_requested_quota_replaces_default_resource(self):
        path = self.folder / 'state.json'
        state = json.loads(path.read_text())
        state['resource_id'] = 'quota4wxjihmxlih'
        atomic_json(path, state)
        calls = []
        def capture(arguments):
            calls.append(arguments)
            return self.remote(arguments)
        self.monitor.real_remote = capture
        self.complete('Sparsh_mae')
        self.monitor.tick()
        submission = next(args for args in calls if args[0] == 'submit')
        self.assertEqual([arg for arg in submission if arg.startswith('--resource_id=')],
                         ['--resource_id=quota4wxjihmxlih'])


if __name__ == '__main__':
    unittest.main()
