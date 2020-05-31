import shutil
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, UnidentifiedImageError
from joblib import hash
from tqdm import tqdm


def ex2(input_dir: str, output_dir: str, logfile: str) -> int:
    _check_paths_params(input_dir, logfile, output_dir)

    files = sorted(Path(input_dir).rglob('*'), key=str)

    # ignore directories
    paths = [path for path in files if not path.is_dir()]
    return _process_files(paths, input_dir, output_dir, logfile)


def _check_paths_params(input_dir: str, logfile: str, output_dir: str):
    def _check_path(path: str):
        if not path:
            raise ValueError(f"Path must not be null or empty: {path}")

    _check_path(input_dir)
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input path '{input_dir}' does not exist!")
    if not input_path.is_dir():
        raise ValueError(f"Input path '{input_dir}' is not a directory!")

    _check_path(logfile)

    # optionally overwrite path and all subdirectories
    path = Path(output_dir)
    if path.exists() and _overwrite:
        shutil.rmtree(path, ignore_errors=True)

    path.mkdir(parents=True, exist_ok=True)


def _process_files(paths: List[Path], input_dir: str, output_dir: str, logfile: str) -> int:
    hashes = dict()

    with open(logfile, 'w') as log:
        def _write_to_log(path: Path, error_code: int):
            log.write(f'{path.relative_to(input_dir)};{error_code}\n')

        for path in tqdm(paths, desc='Checking images'):
            # check if extension (including leading '.') is valid
            if not path.suffix[1:].lower() in _file_types:
                _write_to_log(path, 1)
                continue
            if path.stat().st_size < _min_file_size:
                _write_to_log(path, 2)
                continue

            # file exists, is small enough and might be an image
            try:
                img = Image.open(str(path))
            except (FileNotFoundError, ValueError, UnidentifiedImageError) as e:
                _write_to_log(path, 3)
                if _verbose:
                    print(e)
                continue

            img = np.asarray(img)
            # check if image has more than one grayscale value

            # assuming we only have gray-scale images we just have to check if all values are equal
            # computationally much faster than np.var(img) == 0
            if np.all(img == img[0, 0]):
                _write_to_log(path, 4)
                continue

            # check if (W, H) big enough and image has exactly two dimensions
            if len(img.shape) != 2 or img.shape[0] < _w_min or img.shape[1] < _h_min:
                _write_to_log(path, 5)
                continue

            # file is valid in every way
            h = hash(img)
            if h in hashes:
                _write_to_log(path, 6)
                if _verbose:
                    print(f"'{path}' duplicate of {hashes[h]}")
                continue
            # new file found
            else:
                if _verbose:
                    print(f"New file: '{path}'")
                hashes[h] = (path, len(hashes) + 1)

    # batch copy all valid files
    for path, num in tqdm(hashes.values(), desc='Copying images'):
        shutil.copy(path, Path(output_dir, f'{num:06d}.jpg'))
    return len(hashes)


_min_file_size = 10_000
_file_types = {'jpg', 'jpeg'}
_w_min, _h_min = 100, 100
_overwrite, _verbose = True, False
