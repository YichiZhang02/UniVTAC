"""Local-only policy RPC, keeping simulator and VLM dependencies in separate environments."""
import argparse
import torch
import zmq
from training.inference import ChunkPolicy

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--port',type=int,default=5557)
    a=p.parse_args();model=ChunkPolicy(a.checkpoint)
    ctx=zmq.Context();sock=ctx.socket(zmq.REP);sock.bind(f'tcp://127.0.0.1:{a.port}')
    print('Policy server ready',flush=True)
    try:
        while True:
            request=sock.recv_pyobj()
            if request['op']=='metadata':result=dict(config=model.cfg,stats=model.stats)
            elif request['op']=='predict':result=model.predict(request['batch']).cpu()
            else:raise ValueError(request['op'])
            sock.send_pyobj(result)
    finally:sock.close();ctx.term()
if __name__=='__main__':main()
