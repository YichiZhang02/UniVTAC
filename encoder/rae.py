"""RAE-1in1 / RAE-4in1 with learned absolute positions, no RoPE.

Single-frame processing is independent. The multi-frame variant jointly encodes
four frames and compresses time after all encoder Transformer blocks.
"""
import torch
from torch import nn
from torch.nn import functional as F

SIZES = {'S': (384, 6), 'B': (768, 12)}

VARIANTS = {'rae-1in1': (1, 1), 'rae-4in1': (4, 1)}

class Encoder(nn.Module):
    embed_dim = 384
    grid_size = 14
    def __init__(self, frames=4, variant=None, size='S'):
        super().__init__()
        if variant is None and frames not in (1,4):
            raise ValueError('RAE supports 1 or 4 frames')
        variant = variant or ('rae-1in1' if frames == 1 else 'rae-4in1')
        if variant not in VARIANTS:
            raise ValueError(f'Unknown RAE variant: {variant}')
        frames, latent_frames = VARIANTS[variant]
        if frames not in (1, 4):
            raise ValueError('RAE supports 1 or 4 frames')
        if size not in SIZES:raise ValueError(f'Unknown RAE size: {size}')
        self.size=size
        self.dim,heads = SIZES[size]
        self.embed_dim=self.dim
        dim=self.dim
        self.frames = frames
        self.variant = variant
        self.latent_frames = latent_frames
        self.patch = nn.Conv2d(3, dim, 16, 16)
        self.spatial_pe = nn.Parameter(torch.randn(1, 1, 196, dim) * .02)
        self.temporal_pe = nn.Parameter(torch.randn(1, frames, 1, dim) * .02)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * .02)
        self.cls_pe = nn.Parameter(torch.randn(1, 1, dim) * .02)
        self.blocks = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            dim, heads, dim*4, dropout=0., activation='gelu', batch_first=True,
            norm_first=True), 12, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.compress = nn.Conv1d(dim, dim, 4, stride=4) if frames == 4 else nn.Identity()
        for module in self.blocks.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight,std=.02)
                if module.bias is not None: nn.init.zeros_(module.bias)

    def encode(self, images):
        if images.ndim == 4:
            images = images[:, None]
        b, t, c, h, w = images.shape
        if (t,c,h,w) != (self.frames,3,224,224):
            raise ValueError(f'Expected B,{self.frames},3,224,224, got {images.shape}')
        x = self.patch(images.flatten(0,1)).flatten(2).transpose(1,2).reshape(b,t,196,self.dim)
        x = (x + self.spatial_pe + self.temporal_pe).flatten(1,2)
        x = self.norm(self.blocks(torch.cat(((self.cls+self.cls_pe).expand(b,-1,-1),x),1)))
        return x

    def latent(self, encoded):
        b = len(encoded); t = self.frames
        x = encoded
        patches = x[:,1:].reshape(b,t,196,self.dim).permute(0,2,3,1).reshape(b*196,self.dim,t)
        return self.compress(patches).reshape(b,196,self.dim,self.latent_frames).permute(0,3,1,2).contiguous()

    def tokens(self, images, mode='full'):
        encoded = self.encode(images)
        if mode == 'cls':
            return encoded[:,:1]
        patches = self.latent(encoded)
        if mode == 'pool3x3':
            b,t,_,_ = patches.shape
            return F.adaptive_avg_pool2d(patches.flatten(0,1).transpose(1,2).reshape(b*t,self.dim,14,14),3).flatten(2).transpose(1,2).reshape(b,t*9,self.dim)
        if mode != 'full':
            raise ValueError(mode)
        return patches.flatten(1,2)

    def forward(self, images):
        return self.tokens(images)

class RAE(nn.Module):
    def __init__(self, frames=4, variant=None, size='S'):
        super().__init__()
        self.encoder = Encoder(frames, variant, size)
        self.size=size
        self.dim=self.encoder.embed_dim
        self.frames = frames = self.encoder.frames
        self.variant = self.encoder.variant
        self.expand = nn.ConvTranspose1d(self.dim,self.dim,4,stride=4) if frames == 4 else nn.Identity()
        self.project = nn.Linear(self.dim,256)
        self.queries = nn.Parameter(torch.randn(1,frames*196,256)*.02)
        self.decoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            256,8,1024,dropout=0.,activation='gelu',batch_first=True,norm_first=True),4,
            enable_nested_tensor=False)
        self.pixels = nn.Linear(256,16*16*3)
    def forward(self, images):
        if images.ndim != 5:
            raise ValueError('Expected [B,T,3,224,224]')
        if self.frames == 1 and images.shape[1] > 1:
            # Equal four-frame reconstruction budget, independent frame models.
            b,t = images.shape[:2]
            return self.forward(images.flatten(0,1)[:,None]).reshape(b,t,3,224,224)
        z = self.encoder.tokens(images)
        b = len(z)
        z = z.reshape(b,self.encoder.latent_frames,196,self.dim).permute(0,2,3,1).reshape(b*196,self.dim,self.encoder.latent_frames)
        z = self.expand(z).reshape(b,196,self.dim,self.frames).permute(0,3,1,2).flatten(1,2)
        z = self.pixels(self.decoder(self.project(z)+self.queries))
        return z.reshape(b,self.frames,14,14,16,16,3).permute(0,1,6,2,4,3,5).reshape(b,self.frames,3,224,224)
