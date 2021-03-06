import base64
from subprocess import Popen, PIPE
from typing import List

import cv2
import imageio
import numpy as np
from IPython.display import HTML
from tqdm.notebook import tqdm

from hanga import image as h_image


def get_frames_from_video_file(file_path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(file_path)
    frames = []
    is_read = True
    while is_read:
        is_read, frame = cap.read()
        if frame is not None:
            frames.append(frame)
    return frames


def interpolate(arrs: List[np.ndarray], steps: int) -> List[np.ndarray]:
    out = []
    for i in range(len(arrs) - 1):
        for index in range(steps):
            fraction = index / float(steps)
            out.append(arrs[i + 1] * fraction + arrs[i] * (1 - fraction))
    return out


def create_video(frames: List[np.ndarray], output_file_path: str, fps: int, convert: bool = True) -> None:
    with imageio.get_writer(output_file_path, mode="I", fps=fps) as writer:
        for image in tqdm(frames, desc="creating video"):
            image = image.astype(np.uint8)
            if convert:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            writer.append_data(np.array(image))


def create_video_pytti(frames: List[np.ndarray], fps: int, output_path: str):
    """TODO: Work out what the difference is between this an above (115mb vs 15mb for a 90s video)"""
    # fmt: off
    p = Popen([
        "ffmpeg", "-y", "-f", "image2pipe", "-vcodec", "png", "-r", str(fps), "-i", "-", "-vcodec",
        "libx264", "-r", str(fps), "-pix_fmt", "yuv420p", "-crf", "1", "-preset", "veryslow", output_path,
        ],
        stdin=PIPE,
    )
    for img in tqdm(frames):
        h_image.np_to_pil(img).save(p.stdin, "PNG")
    p.stdin.close()
    print("Encoding video...")
    p.wait()
    print("Video complete.")


def display_video(file_path: str) -> HTML:
    mp4 = open(file_path, "rb").read()
    data_uri = "data:video/mp4;base64," + base64.b64encode(mp4).decode()
    return HTML(
        """
      <video width=400 controls>
        <source src="%s" type="video/mp4">
      </video>
    """
        % data_uri
    )
