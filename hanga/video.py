import base64
from typing import List

import cv2
import imageio
import numpy as np
from IPython.display import HTML
from tqdm.notebook import tqdm


def interpolate(arrs: List[np.ndarray], steps: int) -> List[np.ndarray]:
    out = []
    for i in range(len(arrs) - 1):
        for index in range(steps):
            fraction = index / float(steps) 
            out.append(arrs[i + 1] * fraction + arrs[i] * (1 - fraction))
    return out


def create_video(frames: List[np.ndarray], output_file_path: str, fps: int, convert: bool=True) -> None:
    with imageio.get_writer(output_file_path, mode='I', fps=fps) as writer:
        for image in tqdm(frames, desc="creating video"):
            image = image.astype(np.uint8)
            if convert:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            writer.append_data(np.array(image))
            
            
def display_video(file_path: str):
    mp4 = open(file_path, 'rb').read()
    data_uri = "data:video/mp4;base64," + base64.b64encode(mp4).decode()
    return HTML("""
      <video width=400 controls>
        <source src="%s" type="video/mp4">
      </video>
    """ % data_uri)
