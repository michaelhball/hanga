def chroma_weight_latents(chroma, latents):
    """Creates chromagram weighted latent sequence

    Args:
        chroma (th.tensor): Chromagram
        latents (th.tensor): Latents (must have same number as number of notes in chromagram)

    Returns:
        th.tensor: Chromagram weighted latent sequence
    """
    base_latents = (chroma[..., None, None] * latents[None, ...]).sum(1)
    return base_latents
