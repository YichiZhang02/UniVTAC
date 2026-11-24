import os
import sys
import tempfile
import unittest
from pathlib import Path
from training.remote_evaluation import run_with_watchdog


class WatchdogTest(unittest.TestCase):
    def test_preserves_exit_status(self):
        code,error=run_with_watchdog([sys.executable,'-c','raise SystemExit(3)'],
                                    cwd='.',env=os.environ,result_files=[],poll_interval=.01)
        self.assertEqual(code,3)
        self.assertIsNone(error)

    def test_stalled_process_group_is_killed(self):
        with tempfile.TemporaryDirectory() as tmp:
            child_pid=Path(tmp)/'child.pid'
            source="import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)"
            code,error=run_with_watchdog([sys.executable,'-c',source,str(child_pid)],
                cwd='.',env=os.environ,result_files=[],poll_interval=.02,stall_timeout=.3,stop_grace=.1)
            self.assertEqual(code,124)
            self.assertIn('No saved rollout progress',error)
            # A killed child may briefly remain a zombie until init reaps it.
            status=Path('/proc')/child_pid.read_text()/'stat'
            self.assertTrue(not status.exists() or status.read_text().split()[2]=='Z')


if __name__=='__main__':unittest.main()
