"""Restore saved model/AdamW state; restart the deterministic sampler at saved step."""
import random
import numpy as np
import torch

def restore(path,model,optimizer,cfg,stats):
    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    for key in ('model','chunk','global_batch','micro_batch','tactile','encoder','rae_variant','rae_size','tactile_tokens','tactile_fusion','tactile_wrapper','tactile_rope','vlm_attention_mask','world_size','learning_rate','action_learning_rate','lr_schedule','scheduler_warmup_steps','scheduler_decay_steps','scheduler_decay_lr'):
        if saved['config'].get(key)!=cfg.get(key):raise ValueError('Resume configuration mismatch: '+key)
    if saved['stats']!=stats:raise ValueError('Resume normalization statistics differ')
    model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
    return saved

def restore_rng(saved):
    torch.set_rng_state(saved['rng'])
    # Legacy checkpoints store rank-0 RNG only. All original ranks consumed the same
    # tensor shapes, but worker/Python/NumPy RNG histories were not checkpointed.
    if saved.get('cuda_rng'):torch.cuda.set_rng_state(saved['cuda_rng'][0])
