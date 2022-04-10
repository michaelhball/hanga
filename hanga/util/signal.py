"""
Signal processing utils taken from https://github.com/JCBrouwer/maua-stylegan2/
with minor modification. All credit goes to JCBrouwer.
"""
from typing import Optional

import librosa
import torch
import torch.nn.functional as F

# TODO: parameterise this non-globally
SMF = 1


def gaussian_filter(x: torch.tensor, sigma: float, causal: Optional[float] = None) -> torch.tensor:
    """Smooth tensors along time (first) axis with gaussian kernel.

    Args:
        x: Tensor to be smoothed
        sigma: Standard deviation for gaussian kernel (higher value gives smoother result)
        causal: Factor to multiply right side of gaussian kernel witorch. Lower value
            decreases effect of "future" values

    Returns: smoothed tensor
    """
    dim = len(x.shape)
    n_frames = x.shape[0]
    while len(x.shape) < 3:
        x = x[:, None]

    radius = min(int(sigma * 4 * SMF), 3 * len(x))
    channels = x.shape[1]

    kernel = torch.arange(-radius, radius + 1, dtype=torch.float32, device=x.device)
    kernel = torch.exp(-0.5 / sigma**2 * kernel**2)
    if causal is not None:
        kernel[radius + 1 :] *= 0 if not isinstance(causal, float) else causal
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, len(kernel)).repeat(channels, 1, 1)

    if dim == 4:
        t, c, h, w = x.shape
        x = x.view(t, c, h * w)
    x = x.transpose(0, 2)

    if radius > n_frames:  # prevent padding errors on short sequences
        x = F.pad(x, (n_frames, n_frames), mode="circular")
        print(
            f"WARNING: Gaussian filter radius ({int(sigma * 4 * SMF)}) is larger than number of "
            f"frames ({n_frames}).\n\t Filter size has been lowered to ({radius}). You might want "
            f"to consider lowering sigma ({sigma})."
        )
        x = F.pad(x, (radius - n_frames, radius - n_frames), mode="constant")
    else:
        x = F.pad(x, (radius, radius), mode="circular")

    x = F.conv1d(x, weight=kernel, groups=channels)

    x = x.transpose(0, 2)
    if dim == 4:
        x = x.view(t, c, h, w)

    if len(x.shape) > dim:
        x = x.squeeze()

    return x


def percentile(signal: torch.Tensor, p: int) -> torch.Tensor:
    """Calculate percentile of signal

    Args:
        signal: Signal to normalize
        p: [0-100]. Percentile to find

    Returns: percentile signal value
    """
    k = 1 + round(0.01 * float(p) * (signal.numel() - 1))
    return signal.view(-1).kthvalue(k).values.item()


def percentile_clip(signal: torch.Tensor, p: int) -> torch.Tensor:
    """Normalize signal between 0 and 1, clipping peak values above given percentile

    Args:
        signal: Signal to normalize
        p: [0-100]. Percentile to clip to

    Returns: clipped signal
    """
    locs = torch.arange(0, signal.shape[0])
    peaks = torch.ones(signal.shape, dtype=bool)
    main = signal.take(locs)

    plus = signal.take((locs + 1).clamp(0, signal.shape[0] - 1))
    minus = signal.take((locs - 1).clamp(0, signal.shape[0] - 1))
    peaks &= torch.gt(main, plus)
    peaks &= torch.gt(main, minus)

    signal = signal.clamp(0, percentile(signal[peaks], p))
    signal /= signal.max()

    return signal
