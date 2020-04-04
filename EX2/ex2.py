import argparse
import glob
import shutil
from pathlib import Path
from typing import List

import cv2
import numpy as np


def clean_dataset(input_dir: str, output_dir: str, logfile: str) -> int:
    __check_paths_params(input_dir, logfile, output_dir)

    files: List[str] = sorted(glob.iglob(f'{Path(input_dir)}/**', recursive=True))
    return __process_files(files, input_dir, logfile)


def __check_paths_params(input_dir, logfile, output_dir, overwrite: bool = False):
    def __check_path(path: str):
        if not path:
            raise ValueError(f"Path must not be null or empty: {path}")

    __check_path(input_dir)
    if not (input_path := Path(input_dir)).exists():
        raise ValueError(f"Input path '{input_dir}' does not exist!")
    if not input_path.is_dir():
        raise ValueError(f"Input path '{input_dir}' is not a directory!")

    __check_path(logfile)

    # optionally overwrite path and all subdirectories
    if (path := Path(output_dir)).exists() and overwrite:
        shutil.rmtree(path, ignore_errors=True)

    path.mkdir(parents=True, exist_ok=True)


def __process_files(files: List[str], input_dir: str, logfile: str) -> int:
    def __write_to_log(path: Path, error_code: int):
        log.write(f'{path.relative_to(input_dir)};{error_code}\n')

    hashes = set()
    valid = 0
    with open(logfile, 'w') as log:
        for filename in files:
            if (path := Path(filename)).is_dir():
                # ignore directories
                continue
            if not __valid_extension(path):
                __write_to_log(path, 1)
                continue
            if not path.stat().st_size <= max_file_size:
                __write_to_log(path, 2)
                continue

            # file exists, is small enough and might be an image
            if (img := cv2.imread(filename)) is None:
                __write_to_log(path, 3)
                continue
            # TODO variance over only first two dims
            if np.var(img) == 0:
                __write_to_log(path, 4)
                continue
            if len(shape := img.shape) != len(max_dimensions) or shape > max_dimensions:
                __write_to_log(path, 5)
                continue

            # file is valid in every way
            if (h := hash(bytes(img))) in hashes:
                __write_to_log(path, 6)
                continue
            else:
                # new file found
                print(f'Found new file: {path}')
                hashes.add(h)
                valid += 1

                shutil.copy(path, Path('output', '%06d.jpg' % valid))
    return valid


def __valid_extension(path: Path) -> bool:
    # match case insensitively
    file_name = str(path).lower()
    for ext in file_types:
        # file name has to be longer than extension (x.jpg > jpg)
        if len(ext) < len(file_name) and file_name.endswith(ext.lower()):
            return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Programming in Python 2, Exercise 2')
    parser.add_argument('input_path', nargs=1, type=str,
                        help='Relative or absolute path to input-directory. Path must point to a '
                             'directory. The directory must exist.')
    parser.add_argument('output_path', nargs=1, type=str,
                        help='Relative or absolute path to output-directory. Path must point to a '
                             'directory. The directory does not have to exist and will be created ('
                             'contents will not be overwritten).')
    parser.add_argument('logfile', nargs=1, type=str,
                        help='Relative or absolute path to logfile. Path must point to a file. The '
                             'file does not have to exist and will be created (contents will be '
                             'overwritten)')
    parser.add_argument('-file_types', nargs='+', type=str, required=False, default=['jpg', 'jpeg'],
                        help=f'Allowed file extensions. Defaults to {["jpg", "jpeg"]}')
    parser.add_argument('-file_size', nargs=1, type=int, required=False, default=[10],
                        help='Maximum file size in kB. Defaults to 10.')
    parser.add_argument('-file_dimensions', nargs=1, type=str, required=False,
                        default=['100x100'],
                        help='Maximum dimensions for image (H, W[, D]). Defaults to "100x100".')
    args = parser.parse_args()

    # positional arguments
    input_path = args.input_path[0]
    output_path = args.output_path[0]
    logfile = args.logfile[0]

    # no duplicates necessary/wanted
    file_types = set(args.file_types)

    # in kB
    max_file_size = args.file_size[0] * 1000

    # convert to integer
    max_dimensions = tuple(int(x) for x in args.file_dimensions[0].split('x'))
    if not 2 <= len(max_dimensions) <= 3:
        raise ValueError(f'Image must have two or three dimensions but had: {args.file_dimensions}')

    clean_dataset(input_path, output_path, logfile)
