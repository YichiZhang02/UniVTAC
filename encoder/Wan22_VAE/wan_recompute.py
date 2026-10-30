"""Training-only Wan residual recomputation with isolated causal cache updates."""
import torch
from torch.utils.checkpoint import checkpoint
from diffusers.models.autoencoders.autoencoder_kl_wan import WanResidualBlock


class RecomputedWanResidualBlock(WanResidualBlock):
    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_idx is None:
            feat_idx = [0]
        if not self.training or not torch.is_grad_enabled():
            return super().forward(x, feat_cache, feat_idx)
        # Recomputation must see the original input cache, not the state left by
        # later video chunks. Use a fresh list and pointer on every invocation.
        start = feat_idx[0]
        # This block consumes exactly two causal convolution entries. Capturing
        # unrelated layers' caches would retain large tensors until backward.
        before = list(feat_cache[start:start+2]) if feat_cache is not None else None

        def recompute(value):
            cache = list(before) if before is not None else None
            index = [0]
            output = super(RecomputedWanResidualBlock, self).forward(value, cache, index)
            return output, cache, index[0]

        output, after, end = checkpoint(recompute, x, use_reentrant=False)
        if feat_cache is not None:
            feat_cache[start:start+end] = after
            feat_idx[0] = start + end
        return output


def enable_wan_recomputation(module):
    """Preserve parameter names and bind forward to each DataParallel replica."""
    for block in module.modules():
        if isinstance(block, WanResidualBlock):
            block.__class__ = RecomputedWanResidualBlock
