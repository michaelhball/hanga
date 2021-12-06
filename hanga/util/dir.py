import glob
import os
import re
from os.path import splitext as split_file
from typing import List


def get_image_paths(dirname: str, base_name: str) -> List[str]:
    all_frames = glob.glob(f"{dirname}/{base_name}_*.png")
    all_frames.sort(key=lambda s: int(split_file(s)[0].split("_")[-1]))
    return all_frames


def get_last_file(directory, pattern):
    def key(f):
        index = re.match(pattern, f).group("index")
        return 0 if index == "" else int(index)

    files = [f for f in os.listdir(directory) if re.match(pattern, f)]
    if len(files) == 0:
        return None, None
    files.sort(key=key)
    index = key(files[-1])
    return files[-1], index
