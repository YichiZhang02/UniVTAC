"""Predict 50, execute 30; restore arm targets while keeping gripper absolute."""
import torch
from training.models import ACT,StarVLA
from policy.starvla_pi import StarVLAPI
from policy.starvla_groot_v1 import StarVLAGrootV1
from training.actions import ACTION_CONTRACT, decode_actions

class ChunkPolicy:
    def __init__(self,checkpoint,device='cuda'):
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.cfg=payload['config'];self.stats=payload['stats'];self.device=device
        self.model=({'act': ACT, 'starvla': StarVLA, 'starvla_pi': StarVLAPI, 'starvla_groot_v1': StarVLAGrootV1}[self.cfg['model']])(self.cfg,len(self.stats['state_mean']))
        self.model.load_state_dict(payload['model'],strict=True);self.model.to(device).eval()
    @torch.inference_mode()
    def predict(self,batch):
        batch={k:v.to(self.device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        anchor=batch['state'].clone()
        stats={k:torch.as_tensor(v,device=self.device,dtype=anchor.dtype) for k,v in self.stats.items()}
        batch['state']=(anchor-stats['state_mean'])/stats['state_std']
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if self.cfg['model']=='act':
                normalized=self.model.model(batch['state'],self.model.vision(batch['image']),batch['tactile'],None)[0]
            else:
                normalized=self.model.predict_action(batch)
        actions=normalized.float()*stats['action_std']+stats['action_mean']
        contract=self.cfg.get('action_contract')
        if contract == ACTION_CONTRACT:
            absolute=decode_actions(anchor, actions)
        elif contract is None:
            # Historical checkpoints used relative values for all eight coordinates.
            absolute=actions+anchor[:,None]
        else:
            raise ValueError(f'Unsupported action contract: {contract}')
        return absolute[:,:self.cfg['execute']]
