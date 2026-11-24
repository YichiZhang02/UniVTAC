"""Uniform StarVLA LR schedule, matching UniVTAC1 warmup/cosine defaults."""
import math

DEFAULTS=dict(learning_rate=2.5e-5,action_learning_rate=2.5e-5,
              lr_schedule='cosine',scheduler_warmup_steps=1000,
              scheduler_decay_steps=30000,scheduler_decay_lr=2.5e-6)

def factor(step,cfg):
    warm=int(cfg.get('scheduler_warmup_steps',1000));decay=int(cfg.get('scheduler_decay_steps',30000))
    peak=float(cfg['learning_rate']);floor=float(cfg.get('scheduler_decay_lr',2.5e-6))
    if not (0<=warm<decay and 0<=floor<=peak and peak>0):raise ValueError('Invalid warmup/cosine parameters')
    total=int(cfg['steps'])
    if total<decay:warm=int(warm*total/decay);decay=total
    if step<warm:return 1/(warm+1)+(1-1/(warm+1))*step/warm
    alpha=floor/peak
    return alpha+(1-alpha)*.5*(1+math.cos(math.pi*min(step,decay)/decay))

def build(optimizer,cfg):
    if cfg.get('lr_schedule','cosine')=='fixed':return None
    if cfg.get('lr_schedule','cosine')!='cosine':raise ValueError('Unknown LR schedule')
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer,lambda step:factor(step,cfg))
