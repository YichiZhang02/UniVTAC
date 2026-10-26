"""UniVTAC SSL recipes on the official Sparsh backbones.

Uses the repository's four-frame contract, not an exact upstream reproduction.
MAE reconstructs 12-channel patches; JEPA predicts held-out spatial/tube tokens;
DINOv2 combines cross-view register distillation, masked patches and KoLeo.
Only ``encoder`` is exported; prediction heads and EMA teachers stay in last.pth.
"""
import copy

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as TF

from encoder.DINOv2.model import DINOHead, koleo
from encoder.vit_common import patchify


def prepared_frames(encoder, images):
    images = encoder.preprocess(images).flip(1)
    return images.transpose(1, 2) if encoder.method == 'V-JEPA' else images.flatten(1, 2)


def gather_tokens(tokens, indices):
    return tokens.gather(1, indices[..., None].expand(-1, -1, tokens.shape[-1]))


class SparshTraining(nn.Module):
    recipe = 'univtac_sparsh_ssl_v1'

    def __init__(self, encoder, mask_ratio=.75):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        method, dim = encoder.method, encoder.embed_dim
        if method == 'MAE':
            from tactile_ssl.probe.reconstruction import MaskDecoderViT
            self.decoder = MaskDecoderViT(img_size=encoder.hw, in_chans=12,
                input_embed_dim=dim, embed_dim=192, depth=4, num_heads=6,
                pos_embed_fn='sinusoidal', patch_size=16)
        else:
            self.teacher = copy.deepcopy(encoder.model).requires_grad_(False).eval()
            if method == 'DINOv2':
                self.cls_head = DINOHead(dim)
                self.patch_head = DINOHead(dim)
                self.teacher_cls_head = copy.deepcopy(self.cls_head).requires_grad_(False)
                self.teacher_patch_head = copy.deepcopy(self.patch_head).requires_grad_(False)
                self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
                nn.init.normal_(self.mask_token, std=.02)
                self.register_buffer('cls_center', torch.zeros(1, 1024))
                self.register_buffer('patch_center', torch.zeros(1, 1, 1024))
            else:
                from tactile_ssl.model.vision_transformer import vit_predictor
                self.predictor = vit_predictor(input_dim=dim, img_size=encoder.hw,
                    embed_dim=384, depth=6, num_heads=6, pos_embed_fn='sinusoidal',
                    num_frames=4 if method == 'V-JEPA' else 1, tubelet_size=2)
                grid = encoder.hw[0]//16
                starts = torch.cartesian_prod(torch.arange(grid-5), torch.arange(grid-5))
                disjoint = ((starts[:, None, 0]-starts[None, :, 0]).abs() >= 6
                            ) | ((starts[:, None, 1]-starts[None, :, 1]).abs() >= 6)
                self.register_buffer('block_starts', starts, persistent=False)
                self.register_buffer('block_pairs', disjoint.nonzero(), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        for name in ('teacher', 'teacher_cls_head', 'teacher_patch_head'):
            if hasattr(self, name):
                getattr(self, name).eval()
        return self

    @torch.no_grad()
    def update_teacher(self, momentum=.996):
        if not hasattr(self, 'teacher'):
            return
        pairs = [(self.encoder.model, self.teacher)]
        if self.encoder.method == 'DINOv2':
            pairs += [(self.cls_head, self.teacher_cls_head),
                      (self.patch_head, self.teacher_patch_head)]
        for student, teacher in pairs:
            for source, target in zip(student.parameters(), teacher.parameters()):
                target.lerp_(source, 1-momentum)

    def forward(self, images):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=images.is_cuda):
            x = prepared_frames(self.encoder, images)
            if self.encoder.method == 'MAE':
                loss, metrics = self.mae(x)
            elif self.encoder.method == 'DINOv2':
                loss, metrics = self.dino(x)
            else:
                loss, metrics = self.jepa(x)
        return loss, dict(metrics, _sample_count=loss.new_tensor(len(images)))

    def mae(self, x):
        batch, count = len(x), self.encoder.model.patch_embed.num_patches
        order = torch.rand(batch, count, device=x.device).argsort(1)
        keep = max(1, int(count*(1-self.mask_ratio)))
        visible = order[:, :keep]
        tokens = self.encoder.model(x, [visible])
        prediction = self.decoder(tokens, x.shape, order.argsort(1))
        target = patchify(x)
        error = (prediction.float()-target.float()).square().mean(-1)
        masked = torch.ones_like(error).scatter_(1, visible, 0)
        loss = (error*masked).sum()/masked.sum()
        return loss, {'masked_reconstruction': loss.detach()}

    def block_masks(self, x):
        # Two disjoint 6x6 blocks; video masks cover both temporal tube slices.
        batch, grid = len(x), self.encoder.hw[0]//16
        time = 2 if self.encoder.method == 'V-JEPA' else 1
        row, col = torch.meshgrid(torch.arange(grid, device=x.device),
                                 torch.arange(grid, device=x.device), indexing='ij')
        pairs = self.block_pairs[torch.randint(len(self.block_pairs), (batch,), device=x.device)]
        starts = self.block_starts[pairs]
        top, left = starts[:, :, 0, None, None], starts[:, :, 1, None, None]
        masks = ((row >= top) & (row < top+6) & (col >= left) & (col < left+6)).any(1)
        masks = masks.flatten(1).repeat(1, time)
        target = torch.stack([m.nonzero().flatten() for m in masks])
        context = torch.stack([(~m).nonzero().flatten() for m in masks])
        return context, target

    def jepa(self, x):
        context, target_ids = self.block_masks(x)
        features = self.encoder.model(x, [context])
        prediction = self.predictor(features, [context], [target_ids])
        with torch.no_grad():
            target = self.teacher(x)
            target = gather_tokens(F.layer_norm(target.float(), (target.shape[-1],)), target_ids)
        loss = F.smooth_l1_loss(prediction.float(), target)
        # Track both image and spatial variation to expose collapsed teachers.
        return loss, {'masked_latent': loss.detach(),
                      'student_feature_std': features.float().std(0, unbiased=False).mean().detach(),
                      'teacher_feature_std': target.std(0, unbiased=False).mean().detach()}

    def view(self, x, local=False):
        # Shared geometry and intensity for all four frames in a sensor clip.
        result = []
        for sample in x:
            i, j, h, w = RandomResizedCrop.get_params(sample,
                (.4, .7) if local else (.8, 1.), (.85, 1.15))
            image = TF.resized_crop(sample, i, j, h, w, list(self.encoder.hw))
            result.append((image*torch.empty((), device=x.device).uniform_(.9, 1.1)).clamp(0, 1))
        return torch.stack(result)

    def masked_features(self, x, mask):
        model = self.encoder.model
        pos = model.interpolate_pos_encoding(x.shape, x.dtype, device=x.device)
        tokens = model.patch_embed(x)
        tokens = torch.where(mask[..., None], self.mask_token.to(tokens.dtype), tokens) + pos
        tokens = torch.cat((model.register_tokens.expand(len(x), -1, -1), tokens), 1)
        for block in model.blocks:
            tokens = block(tokens)
        tokens = model.norm(tokens)
        return tokens[:, 0], tokens[:, model.num_register_tokens:]

    def dino(self, x):
        views = [self.view(x), self.view(x), self.view(x, local=True)]
        count = self.encoder.model.patch_embed.num_patches
        masks = [torch.rand(len(x), count, device=x.device) < self.mask_ratio for _ in range(2)]
        # Always supervise at least one patch, also in tiny smoke batches.
        for mask in masks:
            mask[:, 0] = True
        student = [self.masked_features(view, mask) for view, mask in zip(views, masks)]
        local = self.encoder.model.forward_features(views[2])['x_norm_regtokens'][:, 0]
        student_cls = [self.cls_head(z[0]).float()/.1 for z in student] + [self.cls_head(local).float()/.1]
        student_patch = [self.patch_head(z[1]).float()/.1 for z in student]
        with torch.no_grad():
            teacher = [self.teacher.forward_features(view) for view in views[:2]]
            cls = [self.teacher_cls_head(z['x_norm_regtokens'][:, 0]).float() for z in teacher]
            patches = [self.teacher_patch_head(z['x_norm_patchtokens']).float() for z in teacher]
            targets_cls = [((z-self.cls_center)/.04).softmax(-1) for z in cls]
            targets_patch = [((z-self.patch_center)/.04).softmax(-1) for z in patches]
            if self.training:
                means = [torch.stack(cls).mean((0, 1))[None],
                         torch.stack(patches).mean((0, 1, 2))[None, None]]
                if dist.is_initialized():
                    for mean in means:
                        dist.all_reduce(mean)
                        mean.div_(dist.get_world_size())
                self.cls_center.lerp_(means[0], .1)
                self.patch_center.lerp_(means[1], .1)
        cls_loss = torch.stack([-(t*s.log_softmax(-1)).sum(-1).mean()
            for i, t in enumerate(targets_cls) for j, s in enumerate(student_cls) if i != j]).mean()
        patch_loss = torch.stack([-(t[m]*s[m].log_softmax(-1)).sum(-1).mean()
            for t, s, m in zip(targets_patch, student_patch, masks)]).mean()
        regularizer = torch.stack([koleo(z[0]) for z in student]).mean()
        loss = cls_loss + patch_loss + .1*regularizer
        return loss, {'cls_distillation': cls_loss.detach(), 'patch_distillation': patch_loss.detach(),
                      'koleo': regularizer.detach(),
                      'student_feature_std': student[0][0].float().std(0, unbiased=False).mean().detach()}
