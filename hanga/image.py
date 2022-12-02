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


def cv2_clipped_zoom(img: np.ndarray, zoom_factor: float = 0) -> np.ndarray:
    """Center zoom in/out of the given image and returning an enlarged/shrinked view of
    the image without changing dimensions

    Taken from: (https://stackoverflow.com/a/48097478)

    Args:
        img: numpy array image
        zoom_factor : amount of zoom as a ratio [0 to Inf). Default 0.

    Returns:
        result: array of the same shape of the input img zoomed by the specified factor.
    """
    if zoom_factor == 0:
        return img

    height, width = img.shape[:2]  # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    print(width, height)
    print(new_width, new_height)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    # TODO: is this the midpoint that we can change ??
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1, x1, y2, x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]
    print((x1, y1), (x2, y2))

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) // 2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0, 0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode="constant")
    assert result.shape[0] == height and result.shape[1] == width
    return result
