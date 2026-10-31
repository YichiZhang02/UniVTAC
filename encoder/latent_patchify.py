"""Losslessly pack spatial latent patches into token channels, preserving time."""


def patchify_latents(latent, patch_size=2):
    """Convert [B,C,H,W] or [B,C,T,H,W] to time-major [B,N,C*p*p]."""
    if latent.ndim == 4:
        latent = latent.unsqueeze(2)
    if latent.ndim != 5 or patch_size not in (1, 2):
        raise ValueError('Expected a 4D/5D latent and patch_size 1 or 2')
    b, c, t, h, w = latent.shape
    if h % patch_size or w % patch_size:
        raise ValueError('Latent spatial dimensions must be divisible by patch_size')
    return (latent.reshape(b, c, t, h // patch_size, patch_size, w // patch_size, patch_size)
            .permute(0, 2, 3, 5, 1, 4, 6)
            .reshape(b, t * (h // patch_size) * (w // patch_size), c * patch_size**2))
