import json
import pickle
from pathlib import Path
from typing import Union, Any


def load_config(path: Union[Path, str]):
    with open(Path(path), 'r') as f:
        return json.load(f)


def load_pkl(path: Union[str, Path]) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pkl(path: Union[str, Path], obj: Any):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
