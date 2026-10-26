"""No paid submissions: exercise completion gating and crash-safe deduplication."""
import json
from pathlib import Path
import shlex
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.watch_encoder_starvla import Monitor, parse_blocks


class FakeDLC:
    def __init__(self):
        self.encoder_status = {}
        self.jobs = {}
        self.submissions = 0
        self.timeout_after_accept = False

    def __call__(self, arguments):
        if arguments[0] == 'submit':
            self.submissions += 1
            name = next(x.split('=', 1)[1] for x in arguments if x.startswith('--name='))
            command = next(x.split('=', 1)[1] for x in arguments if x.startswith('--command='))
            job_id = f'dlctest{self.submissions:010d}'
            self.jobs[job_id] = dict(DisplayName=name, UserCommand=command, Status='Queuing')
            if self.timeout_after_accept:
                self.timeout_after_accept = False
                raise TimeoutError('Request accepted, response lost')
            return 0, 'Submitted '+job_id
        if '--display_name_regex' in arguments:
            return 0, ' '.join(self.jobs)
        job_id = arguments[2]
        if job_id in self.jobs:
            return 0, json.dumps(self.jobs[job_id])
        return 0, json.dumps(dict(Status=self.encoder_status.get(job_id, 'Running')))


class MonitorTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.remote = FakeDLC()
        blocks = parse_blocks(ROOT / '2.sh')
        source = self.folder / '2.sh'
        commands, ledger = [], []
        for method, env in blocks.items():
            output = self.folder / 'encoders' / method
            output.mkdir(parents=True)
            (output / 'encoder.pth').write_bytes(b'validated test weights')
            env['ENCODER_CKPT'] = str(output / 'encoder.pth')
            commands.append(shlex.join([*[f'{k}={v}' for k, v in env.items()],
                                        'bash', str(ROOT / 'training/train_starvla_groot.sh')]))
            ledger.append(dict(method=method, job_id='encoder-'+method, output_dir=str(output)))
        source.write_text('\n\n'.join(commands)+'\n')
        self.ledger = self.folder / 'encoders.json'
        self.ledger.write_text(json.dumps(ledger))
        self.monitor = Monitor(self.folder / 'watch', self.remote,
                               lambda path, record: dict(initialization='scratch', epoch=3))
        self.monitor.initialize(source, self.ledger, 3600)

    def tearDown(self):
        self.temporary.cleanup()

    def test_running_or_failed_encoders_are_not_submitted(self):
        self.remote.encoder_status['encoder-MAE'] = 'Failed'
        state = self.monitor.tick()
        self.assertEqual(self.remote.submissions, 0)
        self.assertEqual(state['records'][0]['phase'], 'EncoderFailed')
        self.assertTrue(state['next_check_at'])

    def test_successful_early_stop_submits_only_once(self):
        self.remote.encoder_status['encoder-I-JEPA'] = 'Succeeded'
        state = self.monitor.tick()
        self.monitor.tick()
        self.assertEqual(self.remote.submissions, 1)
        record = next(r for r in state['records'] if r['method'] == 'I-JEPA')
        self.assertEqual(record['phase'], 'Submitted')
        self.assertIn('NUM_PROCESSES=4', record['command'])
        self.assertIn('ENCODER_PRETRAIN_INIT=scratch', record['command'])
        self.assertEqual(Path(record['frozen_checkpoint']).read_bytes(), b'validated test weights')
        original = Path(record['source_checkpoint'])
        temporary = original.with_suffix('.new')
        temporary.write_bytes(b'different weights')
        temporary.replace(original)
        self.assertEqual(Path(record['frozen_checkpoint']).read_bytes(), b'validated test weights')

    def test_accept_then_timeout_is_reconciled_without_resubmission(self):
        self.remote.encoder_status['encoder-MAE'] = 'Succeeded'
        self.remote.timeout_after_accept = True
        first = self.monitor.tick()
        self.assertEqual(first['records'][0]['phase'], 'Submitting')
        second = self.monitor.tick()
        self.assertEqual(second['records'][0]['phase'], 'Submitted')
        self.assertEqual(self.remote.submissions, 1)

    def test_invalid_checkpoint_cannot_trigger_a_submission(self):
        self.remote.encoder_status['encoder-MAE'] = 'Succeeded'
        def invalid(path, record):
            raise ValueError('Wrong initialization')
        self.monitor.validate = invalid
        state = self.monitor.tick()
        self.assertEqual(self.remote.submissions, 0)
        self.assertIn('Wrong initialization', state['records'][0]['last_error'])

    def test_all_submitted_finishes_monitor(self):
        for method in ('MAE', 'DINOv2', 'I-JEPA', 'V-JEPA', 'Wan2.2-VAE'):
            self.remote.encoder_status['encoder-'+method] = 'Succeeded'
        state = self.monitor.tick()
        self.assertEqual(state['status'], 'Complete')
        self.assertIsNone(state['next_check_at'])
        self.assertEqual(self.remote.submissions, 5)
        self.monitor.tick()
        self.assertEqual(self.remote.submissions, 5)

    def test_reinitialize_preserves_job_identity_and_progress(self):
        first = json.loads(self.monitor.path.read_text())
        second = self.monitor.initialize(self.folder / '2.sh', self.ledger, 3600)
        self.assertEqual(first['monitor_id'], second['monitor_id'])
        self.assertEqual([r['name'] for r in first['records']], [r['name'] for r in second['records']])


if __name__ == '__main__':
    unittest.main()
