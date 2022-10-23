import pickle
import yaml
from pathlib import Path
from typing import Any
from yaml import Dumper, Loader

from cloudpathlib import CloudPath, GSPath
from hanga.util.typing import DictStrAny, JSON


def _is_gcs_path(file_path: str) -> bool:
    return file_path.startswith(GSPath.cloud_prefix)


def _is_cloud_path(file_path: str) -> bool:
    return _is_gcs_path(file_path)


def _is_local_path(file_path: str) -> bool:
    return not _is_cloud_path(file_path)


def file_exists(file_path: str) -> bool:
    """Checks whether a local or cloud `file_path` 'exists' (is a valid file or directory)."""
    if _is_cloud_path(file_path):
        return CloudPath(file_path).exists()
    else:
        return Path(file_path).exists()


def load_pickle(file_path: str, **kwargs) -> Any:
    """Load a pickled object from a given file_path, either local or cloud."""
    path = CloudPath(file_path) if _is_cloud_path(file_path) else Path(file_path)
    with path.open("rb") as f:
        return pickle.load(f, **kwargs)


def save_pickle(obj_to_pickle: Any, file_path: str, **kwargs) -> Any:
    """Saves an object as a pickle file to a given file_path, either local or cloud."""
    path = CloudPath(file_path) if _is_cloud_path(file_path) else Path(file_path)
    with path.open("wb") as f:
        pickle.dump(obj_to_pickle, f, **kwargs)


def load_yaml(file_path: str) -> JSON:
    """Loads a YAML file to JSON"""
    path = CloudPath(file_path) if _is_cloud_path(file_path) else Path(file_path)
    with path.open("r") as f:
        return yaml.load(stream=f, Loader=Loader)


def save_yaml(yaml_obj: DictStrAny, file_path: str) -> None:
    """Saves a YAML object to file"""
    path = CloudPath(file_path) if _is_cloud_path(file_path) else Path(file_path)
    with path.open("w") as f:
        yaml.dump(data=yaml_obj, stream=f, Dumper=Dumper, line_break="\n")
