import argparse
import glob
import shutil
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, UnidentifiedImageError
from joblib import hash
from tqdm import tqdm


def ex2(input_dir: str, output_dir: str, logfile: str) -> int:
    _check_paths_params(input_dir, logfile, output_dir)

    files: List[str] = sorted(glob.iglob(f'{Path(input_dir)}/**', recursive=True))
    # ignore directories
    paths = [p for path in files if not (p := Path(path)).is_dir()]
    return _process_files(paths, input_dir, output_dir, logfile)


def _check_paths_params(input_dir: str, logfile: str, output_dir: str):
    def _check_path(path: str):
        if not path:
            raise ValueError(f"Path must not be null or empty: {path}")

    _check_path(input_dir)
    if not (input_path := Path(input_dir)).exists():
        raise ValueError(f"Input path '{input_dir}' does not exist!")
    if not input_path.is_dir():
        raise ValueError(f"Input path '{input_dir}' is not a directory!")

    _check_path(logfile)

    # optionally overwrite path and all subdirectories
    if (path := Path(output_dir)).exists() and _overwrite:
        shutil.rmtree(path, ignore_errors=True)

    path.mkdir(parents=True, exist_ok=True)


def _process_files(paths: List[Path], input_dir: str, output_dir: str, logfile: str) -> int:
    hashes = dict()

    with open(logfile, 'w') as log:
        def _write_to_log(path: Path, error_code: int):
            log.write(f'{path.relative_to(input_dir)};{error_code}\n')

        for path in tqdm(paths, desc='Checking images'):
            if not _valid_extension(path):
                _write_to_log(path, 1)
                continue
            if not path.stat().st_size >= _min_file_size:
                _write_to_log(path, 2)
                continue

            # file exists, is small enough and might be an image
            try:
                # experimentally faster than cv2.imread
                img = Image.open(str(path))
            except (FileNotFoundError, ValueError, UnidentifiedImageError) as e:
                _write_to_log(path, 3)
                continue

            img = np.asarray(img)
            # assuming we only have gray-scale images we just have to check if all values are equal
            # computationally much faster than np.var(img) == 0
            # if np.var(img) == 0:
            if np.all(img == img[0, 0]):
                _write_to_log(path, 4)
                continue

            # not too beautiful, but it does the trick
            if len(img.shape) != 2 or img.shape[0] < _w_min or img.shape[1] < _h_min:
                _write_to_log(path, 5)
                continue

            # file is valid in every way
            if (h := hash(img)) in hashes:
                _write_to_log(path, 6)
                if _verbose:
                    print(f"'{path}' duplicate of {hashes[h]}")
                continue

            # new file found
            else:
                if _verbose:
                    print(f"New file: '{path}'")
                hashes[h] = (path, len(hashes) + 1)

    # batch copy at the end
    for (path, num) in tqdm(hashes.values(), desc='Copying images'):
        shutil.copy(path, Path(output_dir, f'{num:06d}.jpg'))
    return len(hashes)


def _valid_extension(path: Path) -> bool:
    # match case insensitively
    file_name = str(path).lower()
    return any(file_name.endswith(ext.lower()) for ext in _file_types)


_min_file_size = 10_000
_file_types = {'jpg', 'jpeg'}
_w_min, _h_min = 100, 100
_overwrite, _verbose = True, False

if __name__ == '_main_':
    parser = argparse.ArgumentParser(prog='Programming in Python 2, Exercise 2')
    parser.add_argument('input_dir', type=str,
                        help='Relative or absolute path to input-directory. Path must point to a '
                             'directory. The directory must exist.')
    parser.add_argument('output_dir', type=str,
                        help='Relative or absolute path to output-directory. Path must point to a '
                             'directory. The directory does not have to exist and will be created ('
                             'contents will not be overwritten).')
    parser.add_argument('logfile', type=str,
                        help='Relative or absolute path to logfile. Path must point to a file. The '
                             'file does not have to exist and will be created (contents will be '
                             'overwritten)')
    parser.add_argument('-file_types', nargs='+', type=str, required=False, default=_file_types,
                        help=f'Allowed file extensions. Defaults to {_file_types}')
    parser.add_argument('-file_size', type=int, required=False, default=_min_file_size / 1_000,
                        help=f'Minimum file size in kB. Defaults to {_min_file_size / 1_000}.')
    parser.add_argument('-file_dimensions', type=str, required=False,
                        default=f'{_w_min}x{_h_min}',
                        help=f'Minimum dimensions for image (H, W). Defaults to "{_w_min}x{_h_min}".')
    parser.add_argument('--overwrite', const=True, action='store_const', default=_overwrite,
                        help='Overwrite output directory.')
    parser.add_argument('--verbose', const=True, action='store_const', default=_verbose,
                        help='Increase verbosity of output.')

    args = parser.parse_args()

    # positional arguments
    input_dir = args.input_dir
    output_dir = args.output_dir
    logfile = args.logfile

    # no duplicates necessary/wanted
    _file_types = set(args.file_types)

    # in kB
    _min_file_size = args.file_size * 1_000

    # convert to integer
    if args.file_dimensions.count('x') != 1:
        raise ValueError(f'Image must have two dimensions but had: {args.file_dimensions}')
    _w_min, _h_min = tuple(int(x) for x in args.file_dimensions.split('x'))

    _overwrite = args.overwrite
    _verbose = args.verbose

    print(ex2(input_dir, output_dir, logfile))
