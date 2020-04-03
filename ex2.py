import argparse
import glob
import os
import shutil
from typing import List

import cv2
import numpy as np


def clean_dataset(input_dir: str, output_dir: str, logfile: str) -> int:
    __check_in_dir(input_dir)
    __create_output_dir(output_dir, overwrite=True)

    paths = __list_all_files_sorted(input_dir)
    return __process_files(paths, input_dir, logfile)


def __check_in_dir(input_dir: str):
    __check_path(input_dir)
    if not os.path.exists(input_dir):
        raise ValueError(f"Input path '{input_dir}' does not exist!")
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input path '{input_dir}' is not a directory!")


def __check_path(path: str):
    if not path:
        raise ValueError(f"Path must not be null or empty: {path}")


def __create_output_dir(path: str, overwrite: bool = False):
    __check_path(path)

    # optionally overwrite path and all subdirectories
    if os.path.exists(path) and overwrite:
        shutil.rmtree(path, ignore_errors=True)

    os.makedirs(path, exist_ok=True)


def __list_all_files_sorted(path: str) -> List[str]:
    # trimming away file separator
    if path.endswith(os.sep):
        path = path[:-1]
    # sort by natural order of strings
    return sorted(glob.glob(f'{path}/**', recursive=True))


def __process_files(paths: List[str], input_dir: str, logfile: str) -> int:
    def __write_to_log(path: str, error_code: int):
        log.write(f'{os.path.relpath(path, input_dir)};{error_code}\n')

    hashes = set()
    valid = 0
    with open(logfile, 'w') as log:
        for path in paths:
            if os.path.isdir(path):
                # ignore directories
                continue
            if not __valid_extension(path):
                __write_to_log(path, 1)
                continue
            if not __valid_file_size(path):
                __write_to_log(path, 2)
                continue

            # file exists, is small enough and might be an image
            if (img := cv2.imread(path)) is None:
                __write_to_log(path, 3)
                continue
            if np.var(img) == 0:
                __write_to_log(path, 4)
                continue
            if len(img.shape) != len(max_dimensions) or img.shape > max_dimensions:
                __write_to_log(path, 5)
                continue

            # file is valid in every way
            if (h := hash(bytes(img))) in hashes:
                __write_to_log(path, 6)
                continue
            else:
                # new file found
                # print(f'Found new file: {path}')
                hashes.add(h)
                valid += 1

                shutil.copy(path, ('output/%06d.jpg' % valid))
    return valid


def __valid_extension(path: str) -> bool:
    # length has to be at least 4 to even match
    # .jpg would be the shortest possible matching path
    return len(path) >= 4 and path[-3:].lower() == 'jpg' or path[-4:].lower() == 'jpeg'


def __valid_file_size(path: str) -> bool:
    return os.path.getsize(path) <= max_file_size


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

    input_path = args.input_path[0]
    output_path = args.output_path[0]
    logfile = args.logfile[0]

    file_types = args.file_types[0]

    # in kB
    max_file_size = args.file_size[0] * 1000

    # convert to integer
    max_dimensions = tuple(int(x) for x in args.file_dimensions[0].split('x'))
    if not 2 <= len(max_dimensions) <= 3:
        raise ValueError(f'Image must have two or three dimensions but had: {args.file_dimensions}')

    clean_dataset(input_path, output_path, logfile)
