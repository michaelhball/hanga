import enum
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL


class ImageFormat(enum.Enum):
    BGR = enum.auto()
    RGB = enum.auto()


def imshow(img: np.ndarray):
    # TODO: make this better
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def imshow2(img1: np.ndarray, img2: np.ndarray):
    _, ax = plt.subplots(1, 2)
    ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.show()


def imshow3(img1: np.ndarray, img2: np.ndarray, img3: np.ndarray):
    _, ax = plt.subplots(1, 3)
    ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    ax[2].imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    plt.show()


# TODO: make a plot grid function


def np_to_pil(frame: np.ndarray) -> PIL.Image:
    color_frame = cv2.cvtColor(np.float32(frame), cv2.COLOR_BGR2RGB)
    return PIL.Image.fromarray(np.uint8(color_frame)).convert("RGB")


def mirror(img: np.ndarray, axis: int = 0, colour_format: ImageFormat = ImageFormat.BGR) -> np.ndarray:
    """Mirrors an image in either axis"""  # TODO: add a diagonal axis mirror
    assert axis in {0, 1}

    if colour_format == ImageFormat.BGR:
        idx = 1 if axis == 0 else 0
    elif colour_format == ImageFormat.RGB:
        idx = axis
    else:
        raise ValueError("`colour_format` must be one of the valid `Format` options")

    if idx == 0:
        return img[::-1, :, :]
    else:
        return img[:, ::-1, :]


def apply_brightness_contrast(input_img: np.ndarray, brightness: int = 0, contrast: int = 0) -> np.ndarray:
    """Util to modify an image's brightness and contrast, taken from
    https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
    brightness & contrast values range are in {-127, 127}.
    """
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def contrast_image(img, clip_limit=100, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def white_image(shape: Tuple[int, int, int]) -> np.ndarray:
    return np.ones(shape, dtype=np.float32)


def black_image(shape: Tuple[int, int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def image_grid(imgs: List[np.ndarray], rows: int, cols: int) -> PIL.Image:
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = PIL.Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid
