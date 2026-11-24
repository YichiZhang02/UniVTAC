"""Bound simulator cleanup so failed rollouts cannot leave a live evaluation forever."""
import os,sys,threading,traceback

def close_bounded(resources,exit_code,timeout=20):
    def expired():
        print('Evaluation cleanup timed out; exiting with code '+str(exit_code),file=sys.stderr,flush=True)
        os._exit(exit_code)
    timer=threading.Timer(timeout,expired);timer.daemon=True;timer.start()
    try:
        for resource in resources:
            if resource is not None:
                try:resource.close()
                except Exception:traceback.print_exc()
    finally:timer.cancel()
