"""DINOv2-style global CLS and masked patch self-distillation."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from vit_encoder import TactileViT
from vit_common import augment


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=1024):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, 2048), nn.GELU(),
                                 nn.Linear(2048, 2048), nn.GELU(),
                                 nn.Linear(2048, 256))
        self.prototypes = nn.Linear(256, out_dim, bias=False)

    def forward(self, features):
        bottleneck = F.normalize(self.mlp(features), dim=-1)
        weights = F.normalize(self.prototypes.weight, dim=-1)
        return F.linear(bottleneck, weights)


def koleo(features):
    features = F.normalize(features.float(), dim=-1)
    if len(features) < 2:
        return features.new_zeros(())
    distances = torch.cdist(features, features).masked_fill(
        torch.eye(len(features), device=features.device, dtype=torch.bool), float('inf'))
    return -distances.min(dim=1).values.clamp_min(1e-6).log().mean()


@torch.no_grad()
def sinkhorn_targets(logits, temperature, iterations=None):
    """Balance sharp teacher logits without losing prototypes to exp underflow."""
    log_assignments = (logits.float() / temperature).T
    prototypes, samples = log_assignments.shape
    if iterations is None:
        iterations = 30 if samples < prototypes else 3
    log_assignments -= torch.logsumexp(log_assignments.flatten(), dim=0)
    for _ in range(iterations):
        log_assignments -= torch.logsumexp(log_assignments, dim=1, keepdim=True)
        log_assignments -= math.log(prototypes)
        log_assignments -= torch.logsumexp(log_assignments, dim=0, keepdim=True)
        log_assignments -= math.log(samples)
    return (log_assignments + math.log(samples)).T.exp()


class DINOv2(nn.Module):
    def __init__(self, size, in_chans=3, student_temperature=.1,
                 teacher_temperature=.02, mask_ratio=.5, patch_weight=1.,
                 koleo_weight=.1, head_koleo_weight=.1, centering='ema',
                 num_prototypes=1024):
        super().__init__()
        self.encoder = TactileViT(size, in_chans=in_chans)
        dim = self.encoder.embed_dim
        self.cls_head = DINOHead(dim, num_prototypes)
        self.patch_head = DINOHead(dim, num_prototypes)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=.02)
        self.teacher = copy.deepcopy(self.encoder)
        self.teacher_cls_head = copy.deepcopy(self.cls_head)
        self.teacher_patch_head = copy.deepcopy(self.patch_head)
        for module in (self.teacher, self.teacher_cls_head, self.teacher_patch_head):
            for param in module.parameters():
                param.requires_grad_(False)
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.mask_ratio = mask_ratio
        self.patch_weight = patch_weight
        self.koleo_weight = koleo_weight
        self.head_koleo_weight = head_koleo_weight
        if centering not in ('ema', 'sinkhorn'):
            raise ValueError('centering must be ema or sinkhorn')
        self.centering = centering
        self.register_buffer('cls_center', torch.zeros(1, num_prototypes))
        self.register_buffer('patch_center', torch.zeros(1, 1, num_prototypes))

    def forward(self, images):
        views = [augment(images), augment(images), augment(images, local=True)]
        masks = [torch.rand(images.shape[0], 196, device=images.device) < self.mask_ratio
                 for _ in range(2)]
        globals_student = [self.encoder.tokens(view, patch_mask=mask, mask_token=self.mask_token)
                           for view, mask in zip(views[:2], masks)]
        local_student = self.encoder.tokens(views[2])
        all_student = globals_student + [local_student]
        cls_student = [self.cls_head(tokens[:, 0]) / self.student_temperature
                       for tokens in all_student]
        patch_student = [self.patch_head(tokens[:, 1:]) / self.student_temperature
                         for tokens in globals_student]
        with torch.no_grad():
            teacher_tokens = [self.teacher.tokens(view) for view in views[:2]]
            cls_logits = [self.teacher_cls_head(tokens[:, 0]) for tokens in teacher_tokens]
            patch_logits = [self.teacher_patch_head(tokens[:, 1:]) for tokens in teacher_tokens]
            if self.centering == 'sinkhorn':
                cls_targets = sinkhorn_targets(
                    torch.cat(cls_logits), self.teacher_temperature).chunk(2)
                patch_shape = patch_logits[0].shape
                patch_targets = sinkhorn_targets(
                    torch.cat(patch_logits).reshape(-1, patch_shape[-1]),
                    self.teacher_temperature).reshape(2, *patch_shape).unbind(0)
            else:
                cls_targets = [F.softmax((logits - self.cls_center) / self.teacher_temperature, dim=-1)
                               for logits in cls_logits]
                patch_targets = [F.softmax((logits - self.patch_center) / self.teacher_temperature, dim=-1)
                                 for logits in patch_logits]
            if self.training and self.centering == 'ema':
                self.cls_center.mul_(.9).add_(torch.stack(cls_logits).mean((0, 1))[None], alpha=.1)
                self.patch_center.mul_(.9).add_(torch.stack(patch_logits).mean((0, 1, 2))[None, None], alpha=.1)
        cls_terms = [-(target * F.log_softmax(student, dim=-1)).sum(-1).mean()
                     for ti, target in enumerate(cls_targets)
                     for si, student in enumerate(cls_student) if ti != si]
        cls_loss = torch.stack(cls_terms).mean()
        patch_terms = [-(target[mask] * F.log_softmax(student[mask], dim=-1)).sum(-1).mean()
                       for target, student, mask in zip(patch_targets, patch_student, masks)]
        patch_loss = torch.stack(patch_terms).mean()
        koleo_loss = torch.stack([koleo(tokens[:, 0]) for tokens in globals_student]).mean()
        head_koleo = torch.stack([koleo(logits) for logits in cls_student[:2]]).mean()
        loss = (cls_loss + self.patch_weight * patch_loss
                + self.koleo_weight * koleo_loss + self.head_koleo_weight * head_koleo)
        cls_entropy = torch.stack([-(target * target.clamp_min(1e-8).log()).sum(-1).mean()
                                   for target in cls_targets]).mean()
        cls_usage = torch.cat(cls_targets, dim=0).mean(dim=0)
        cls_usage_entropy = -(cls_usage * cls_usage.clamp_min(1e-8).log()).sum()
        patch_entropy = torch.stack([-(target * target.clamp_min(1e-8).log()).sum(-1).mean()
                                     for target in patch_targets]).mean()
        return loss, {'cls_distillation': cls_loss.detach(),
                      'patch_distillation': patch_loss.detach(),
                      'koleo': koleo_loss.detach(),
                      'head_koleo': head_koleo.detach(),
                      'cls_feature_std': globals_student[0][:, 0].float().std(dim=0).mean().detach(),
                      'cls_head_std': cls_student[0].float().std(dim=0).mean().detach(),
                      'cls_target_entropy': cls_entropy.detach(),
                      'cls_usage_entropy': cls_usage_entropy.detach(),
                      'patch_target_entropy': patch_entropy.detach()}

    @torch.no_grad()
    def update_teacher(self, momentum=.996):
        for source, target in ((self.encoder, self.teacher),
                               (self.cls_head, self.teacher_cls_head),
                               (self.patch_head, self.teacher_patch_head)):
            for student, teacher in zip(source.parameters(), target.parameters()):
                teacher.lerp_(student, 1 - momentum)
