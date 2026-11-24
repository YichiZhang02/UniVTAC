import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]

class Tactile(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.kind=cfg['encoder'];self.mode=cfg['tactile_tokens']
        if self.kind=='rae':
            from encoder.rae import Encoder
            self.encoder=Encoder(cfg['tactile_frames'],cfg.get('rae_variant'),cfg.get('rae_size','S'));self.dim=self.encoder.embed_dim
            if self.encoder.frames != cfg['tactile_frames']:
                raise ValueError('RAE variant and tactile_frames disagree')
        elif self.kind=='univtac':
            from torchvision.models import resnet18
            self.encoder=resnet18(weights=None,num_classes=512);self.dim=512
        else: raise ValueError(self.kind)
        checkpoint=cfg['encoder_checkpoint'] or (cfg.get('univtac_checkpoint') if self.kind=='univtac' else None)
        if checkpoint:
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if self.kind=='rae' and 'encoder_spec' in state:
                spec=state['encoder_spec']
                if spec.get('size','S') != self.encoder.size:
                    raise ValueError('Pretrained RAE size does not match policy config')
                if spec['variant'] != self.encoder.variant:
                    raise ValueError('Pretrained RAE variant does not match policy config')
                if spec['input_mode'] != cfg.get('tactile_input_mode','marker_rgb'):
                    raise ValueError('Pretrained RAE input representation does not match policy config')
                if self.encoder.frames > 1 and spec['stride'] != cfg['tactile_stride']:
                    raise ValueError('Pretrained RAE frame stride does not match policy config')
                if spec.get('representation') is not None:
                    from encoder.tactile import representation_spec
                    expected=representation_spec(cfg.get('tactile_interpolation','tps_bilinear'),cfg.get('tactile_interpolation_grid',28))
                    if spec['representation'] != expected:
                        raise ValueError('Pretrained RAE interpolation specification does not match policy config')
            state=state.get('encoder',state.get('model',state))
            if self.kind=='univtac' and any(k.startswith('backbone.') for k in state):
                state={k[len('backbone.'):]:v for k,v in state.items() if k.startswith('backbone.')}
            self.encoder.load_state_dict(state,strict=True)
        from encoder.token_pooling import Learned3x3
        self.downsample = Learned3x3(self.dim) if self.mode == 'learned3x3' else None
        self.freeze_encoder = bool(cfg.get('freeze_tactile_encoder',False))
        if self.freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def token_layout(self, x):
        sensors = x.shape[1]
        frames = self.encoder.latent_frames if self.kind == 'rae' else 1
        side = 1 if self.mode == 'cls' else 3 if self.mode in ('pool3x3','learned3x3') else 14 if self.kind == 'rae' else 7
        return sensors, (1 if self.mode == 'cls' else frames), side, side

    def forward(self,x):
        b,s,t,c,h,w=x.shape
        if self.kind=='rae':
            if self.mode != 'learned3x3':
                return self.encoder.tokens(x.flatten(0,1),self.mode).reshape(b,-1,self.dim)
            frames = self.encoder.latent_frames
            patches = self.encoder.tokens(x.flatten(0,1),'full')
            features = patches.reshape(b*s*frames,14,14,self.dim).permute(0,3,1,2)
            return self.downsample(features).flatten(2).transpose(1,2).reshape(b,-1,self.dim)
        x=x[:,:,-1].flatten(0,1)
        # Native UniVTAC ResNet trunk; expose its spatial features for token selection.
        e=self.encoder
        x=e.maxpool(e.relu(e.bn1(e.conv1(x))))
        for layer in (e.layer1,e.layer2,e.layer3,e.layer4): x=layer(x)
        if self.mode=='cls': x=F.adaptive_avg_pool2d(x,1)
        elif self.mode=='pool3x3': x=F.adaptive_avg_pool2d(x,3)
        elif self.mode=='learned3x3': x=self.downsample(x)
        elif self.mode!='full': raise ValueError(self.mode)
        return x.flatten(2).transpose(1,2).reshape(b,-1,512)

class ACT(nn.Module):
    def __init__(self,cfg,dim):
        super().__init__()
        import yaml
        sys.path.insert(0,str(ROOT/'policy/ACT'))
        from policy.ACT.detr.models.detr_vae import build
        conf=yaml.safe_load((ROOT/'policy/ACT/train_config_vision_all.yml').read_text())
        conf.update(state_dim=dim,chunk_size=cfg['chunk'],camera_names=['cam_head','cam_wrist'],
                    tactile_names=[],tactile_ckpt=None)
        # Build directly: the upstream policy constructor creates its own CUDA optimizer.
        self.model=build(SimpleNamespace(**conf))
        weights=torch.load(cfg['vision_checkpoint'],map_location='cpu',weights_only=True)
        self.model.backbones[0][0].body.load_state_dict({k:v for k,v in weights.items() if not k.startswith('fc.')},strict=True)
        self.tactile=Tactile(cfg) if cfg['tactile'] else None
        if self.tactile:
            self.model.tactile_names=['left','right']
            self.model.tactile_input_proj=nn.Conv2d(self.tactile.dim,conf['hidden_dim'],1)
            # Use a shared token backbone via the native ACT feature interface.
            self.model.backbones[1]=ACTTactile(self.tactile,conf['hidden_dim'])
        self.kl_weight=10.
    def forward(self,batch):
        from policy.ACT.act_policy import kl_divergence
        images=self.vision(batch['image'])
        pred,_,(mu,logvar)=self.model(batch['state'],images,batch['tactile'],None,
                                   batch['action'],batch['is_pad'])
        valid=(~batch['is_pad']).unsqueeze(-1)
        l1=((pred-batch['action']).abs()*valid).sum()/(valid.sum()*pred.shape[-1])
        return l1+self.kl_weight*kl_divergence(mu,logvar)[0][0]

    @staticmethod
    def vision(images):
        mean=images.new_tensor([.485,.456,.406])[None,None,:,None,None]
        std=images.new_tensor([.229,.224,.225])[None,None,:,None,None]
        return (images-mean)/std

class ACTTactile(nn.Module):
    def __init__(self,tactile,hidden):
        super().__init__();self.tactile=tactile;self.hidden=hidden
    def forward(self,x):
        z=self.tactile(x[:,None]).transpose(1,2).unsqueeze(-2)
        return [z],[torch.zeros((len(x),self.hidden,1,z.shape[-1]),device=x.device,dtype=x.dtype)]

class StarVLA(nn.Module):
    def __init__(self,cfg,dim):
        super().__init__()
        from omegaconf import OmegaConf
        from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T
        conf=OmegaConf.create({'framework':{'name':'QwenGR00T','qwenvl':{
            'base_vlm':cfg['base_vlm'],'attn_implementation':'sdpa'},'action_model':{
            'action_dim':dim,'state_dim':dim,'action_horizon':cfg['chunk'],
            'hidden_size':cfg['action_hidden_dim'],'action_hidden_dim':cfg['action_hidden_dim'],
            'expert_hidden_size':cfg.get('action_attention_dim'),
            'num_target_vision_tokens':cfg.get('action_future_tokens',32),
            'num_inference_timesteps':cfg.get('num_inference_timesteps',10),
            'diffusion_model_cfg':{'action_query_count':cfg['chunk'] if cfg.get('head_action_queries_only',False) else None},
            'repeated_diffusion_steps':cfg.get('flow_repeats',8)}},'datasets':{'vla_data':{}},'trainer':{}})
        self.native=Qwen_GR00T(conf)
        self.native.qwen_vl_interface.model.gradient_checkpointing_disable()
        self.native.qwen_vl_interface.model.config.use_cache=False
        vlm_config = self.native.qwen_vl_interface.model.config
        getattr(vlm_config, 'text_config', vlm_config).use_cache = False
        self.tactile=Tactile(cfg) if cfg['tactile'] else None
        hidden=self.native.qwen_vl_interface.model.config.hidden_size
        self.project=nn.Linear(self.tactile.dim,hidden) if self.tactile else None
        from training.tactile_vlm import TactileVLM, validate_config
        validate_config(cfg)
        self.fusion_mode=cfg.get('tactile_fusion','vlm')
        inject = self.tactile is not None and self.fusion_mode != 'head'
        self.vlm_tactile=TactileVLM(cfg if inject else dict(cfg,tactile_wrapper='none'),hidden) \
            if inject or cfg.get('vlm_attention_mask','causal') != 'causal' else None

    def encode(self,batch):
        from PIL import Image
        images=[[Image.fromarray((x.permute(1,2,0).cpu().numpy()*255).astype('uint8'))
                 for x in sample] for sample in batch['image']]
        vlm=self.native.qwen_vl_interface
        if self.vlm_tactile is not None and self.tactile is not None and self.fusion_mode != 'head':
            tactile=self.project(self.tactile(batch['tactile']))
            layers,packed=self.vlm_tactile(vlm,images,batch['lang'],tactile,
                                          self.tactile.token_layout(batch['tactile']))
            return layers[-1],packed.valid
        if self.vlm_tactile is not None:
            layers,packed=self.vlm_tactile(vlm,images,batch['lang'])
            hidden,mask=layers[-1],packed.valid
        else:
            inputs=vlm.build_qwenvl_inputs(images,batch['lang'])
            outputs=vlm(**inputs,output_hidden_states=True,return_dict=True,use_cache=False)
            hidden=outputs.hidden_states[-1]
            mask=inputs['attention_mask'].bool()
        if self.tactile:
            tokens=self.project(self.tactile(batch['tactile'])).to(hidden.dtype)
            hidden=torch.cat((hidden,tokens),1)
            mask=torch.cat((mask,torch.ones(tokens.shape[:2],device=mask.device,dtype=torch.bool)),1)
        return hidden,mask

    def forward(self,batch):
        hidden,mask=self.encode(batch)
        # Repeat-end targets are deliberate; padded actions use terminal hold targets.
        repeats=int(self.native.config.framework.action_model.repeated_diffusion_steps)
        return self.native.action_model(hidden.repeat(repeats,1,1),batch['action'].to(hidden.dtype).repeat(repeats,1,1),
            batch['state'][:,None].to(hidden.dtype).repeat(repeats,1,1),encoder_attention_mask=mask.repeat(repeats,1))

    @torch.inference_mode()
    def predict_action(self,batch):
        hidden,mask=self.encode(batch)
        return self.native.action_model.predict_action(hidden,batch['state'][:,None].to(hidden.dtype),
                                                      encoder_attention_mask=mask)
