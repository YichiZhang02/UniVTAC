import subprocess,sys,unittest
class CleanupTest(unittest.TestCase):
    def test_hung_cleanup_exits_with_original_status(self):
        for code in (0,1):
            script=f'''from training.eval_cleanup import close_bounded
import time
class Hung:
 def close(self):time.sleep(20)
close_bounded([Hung()],{code},timeout=.1)
'''
            result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,timeout=3)
            self.assertEqual(result.returncode,code);self.assertIn('cleanup timed out',result.stderr)
if __name__=='__main__':unittest.main()
