import gc

import torch

from hanga.models.stylegan2_torch import Generator


def generate_latents(
    n_latents: int,
    ckpt: str,
    G_res: int,
    noconst: bool = False,
    latent_dim: int = 512,
    n_mlp: int = 8,
    channel_multiplier: int = 2,
) -> torch.tensor:
    """Generates random, mapped latents

    Args:
        n_latents: Number of mapped latents to generate
        ckpt: Generator checkpoint to use
        G_res: Generator's training resolution
        noconst: Whether the generator was trained without constant starting layer. Defaults to False.
        latent_dim: Size of generator's latent vectors. Defaults to 512.
        n_mlp: Number of layers in the generator's mapping network. Defaults to 8.
        channel_multiplier: Scaling multiplier for generator's channel depth. Defaults to 2.

    Returns: Set of mapped latents
    """
    generator = Generator(
        G_res,
        latent_dim,
        n_mlp,
        channel_multiplier=channel_multiplier,
        constant_input=not noconst,
        checkpoint=ckpt,
    ).cuda()
    zs = torch.randn((n_latents, latent_dim), device="cuda")
    latent_selection = generator(zs, map_latents=True).cpu()
    del generator, zs
    gc.collect()
    torch.cuda.empty_cache()
    return latent_selection
