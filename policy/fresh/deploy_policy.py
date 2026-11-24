"""Shared Isaac51 rollout adapter for the fresh ACT / Qwen-GR00T checkpoints."""
from collections import deque
import os
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'policy/starvla_groot/vendor'))
from training.data import image_tensor
from training.inference import ChunkPolicy
from encoder.tactile import from_observation
from policy._base_policy import BasePolicy

def cpu_array(value):
    return value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else value

class Policy(BasePolicy):
    def __init__(self,args):
        checkpoint=args.get('checkpoint') or os.environ.get('CHECKPOINT')
        if not checkpoint:raise ValueError('Set CHECKPOINT to a fresh training checkpoint')
        self.model=RemotePolicy(os.environ['POLICY_SERVER']) if os.environ.get('POLICY_SERVER') else ChunkPolicy(checkpoint)
        self.cfg=self.model.cfg
        self.reset()
    def reset(self):
        size=self.cfg['tactile_frames']*self.cfg['tactile_stride']
        self.history=deque(maxlen=size);self.actions=deque()
    def encode_obs(self,obs):
        dim=len(self.model.stats['state_mean'])
        q=torch.as_tensor(cpu_array(obs['embodiment']['joint'])[:dim],dtype=torch.float32)[None]
        images=torch.stack([image_tensor(cpu_array(obs['observation'][c]['rgb'])) for c in ('head','wrist')])[None]
        tactile=torch.empty(1,0)
        if self.cfg['tactile']:
            frame=[]
            for side in ('left','right'):
                key=next(k for k in (f'{side}_tactile',f'{side}_gsmini') if k in obs['tactile'])
                data=obs['tactile'][key]
                if self.cfg.get('tactile_input_mode','marker_rgb')=='depth_deform_radial':
                    frame.append(from_observation(data,method=self.cfg.get('tactile_interpolation','tps_bilinear'),
                                                  grid=self.cfg.get('tactile_interpolation_grid',28)))
                else:
                    frame.append(image_tensor(cpu_array(data['rgb_marker'])))
            self.history.append(torch.stack(frame))
            frames=list(self.history)
            while len(frames)<self.history.maxlen:frames.insert(0,frames[0])
            tactile=torch.stack([frames[-1-i*self.cfg['tactile_stride']] for i in reversed(range(self.cfg['tactile_frames']))],1)[None]
        return dict(state=q,image=images,tactile=tactile)
    def eval(self,task,observation):
        batch=self.encode_obs(observation)
        if not self.actions:
            batch['lang']=[task.instruction]
            self.actions.extend(self.model.predict(batch)[0])
        action=self.actions.popleft().to(task.device).float()
        task.take_action(action,action_type='qpos')

class RemotePolicy:
    def __init__(self,address):
        import zmq
        self.context=zmq.Context();self.socket=self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO,600000);self.socket.setsockopt(zmq.SNDTIMEO,600000)
        self.socket.connect(address);self.socket.send_pyobj({'op':'metadata'})
        metadata=self.socket.recv_pyobj();self.cfg=metadata['config'];self.stats=metadata['stats']
    def predict(self,batch):
        self.socket.send_pyobj({'op':'predict','batch':batch});return self.socket.recv_pyobj()
    def close(self):
        self.socket.close(linger=0);self.context.term()
