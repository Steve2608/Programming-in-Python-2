import argparse
import glob
import shutil
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, UnidentifiedImageError
from joblib import hash
# to measure execution time
from tqdm import tqdm


def clean_dataset(input_dir: str, output_dir: str, logfile: str) -> int:
    __check_paths_params(input_dir, logfile, output_dir)

    files: List[str] = sorted(glob.iglob(f'{Path(input_dir)}/**', recursive=True))
    # ignore directories
    paths = list(
        filter(
            lambda path: not path.is_dir(),
            map(
                lambda filename: Path(filename),
                files
            )
        )
    )
    return __process_files(paths, input_dir, logfile)


def __check_paths_params(input_dir: str, logfile: str, output_dir: str):
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
    if (path := Path(output_dir)).exists() and __overwrite:
        shutil.rmtree(path, ignore_errors=True)

    path.mkdir(parents=True, exist_ok=True)


def __process_files(paths: List[Path], input_dir: str, logfile: str) -> int:
    def __write_to_log(path: Path, error_code: int):
        log.write(f'{path.relative_to(input_dir)};{error_code}\n')

    valid = 0
    hashes = dict()

    with open(logfile, 'w') as log:
        for path in tqdm(paths, desc='Checking images'):
            if not __valid_extension(path):
                __write_to_log(path, 1)
                continue
            if not path.stat().st_size >= __min_file_size:
                __write_to_log(path, 2)
                continue

            # file exists, is small enough and might be an image
            try:
                # experimentally faster than cv2.imread
                img = Image.open(str(path))
            except (FileNotFoundError, ValueError, UnidentifiedImageError) as e:
                __write_to_log(path, 3)
                continue

            img = np.asarray(img)
            # assuming we only have gray-scale images we just have to check if all values are equal
            # computationally much faster than np.var(img) == 0
            # if np.var(img) == 0:
            if np.all(img == img[0, 0]):
                __write_to_log(path, 4)
                continue

            # not too beautiful, but it does the trick
            if len(img.shape) != 2 or img.shape[0] < __w_min or img.shape[1] < __h_min:
                __write_to_log(path, 5)
                continue

            # file is valid in every way
            if (h := hash(img)) in hashes:
                __write_to_log(path, 6)
                if __verbose:
                    print(f"'{path}' duplicate of {hashes[h]}")
                continue

            # new file found
            else:
                if __verbose:
                    print(f"New file: '{path}'")
                valid += 1
                hashes[h] = (path, valid)

        # batch copy at the end
        for (path, num) in tqdm(hashes.values(), desc='Copying images'):
            shutil.copy(path, Path('output', f'{num:06d}.jpg'))
    return valid


def __valid_extension(path: Path) -> bool:
    # match case insensitively
    file_name = str(path).lower()
    for ext in __file_types:
        # file name has to be longer than extension (x.jpg > jpg)
        if len(ext) < len(file_name) and file_name.endswith(ext.lower()):
            return True
    return False


__min_file_size = 10_000
__file_types = ['jpg', 'jpeg']
__w_min, __h_min = 100, 100
__overwrite, __verbose = True, False

if __name__ == '__main__':
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
    parser.add_argument('-file_types', nargs='+', type=str, required=False, default=['jpg', 'jpeg'],
                        help=f'Allowed file extensions. Defaults to {["jpg", "jpeg"]}')
    parser.add_argument('-file_size', type=int, required=False, default=__min_file_size / 1_000,
                        help='Minimum file size in kB. Defaults to 10.')
    parser.add_argument('-file_dimensions', type=str, required=False,
                        default=f'{__w_min}x{__h_min}',
                        help='Minimum dimensions for image (H, W). Defaults to "100x100".')
    parser.add_argument('--overwrite', const=True, action='store_const', default=__overwrite,
                        help='Overwrite output directory.')
    parser.add_argument('--verbose', const=True, action='store_const', default=__verbose,
                        help='Increase verbosity of output.')

    args = parser.parse_args()

    # positional arguments
    input_dir = args.input_dir
    output_dir = args.output_dir
    logfile = args.logfile

    # no duplicates necessary/wanted
    __file_types = set(args.file_types)

    # in kB
    __min_file_size = args.file_size * 1000

    # convert to integer
    if args.file_dimensions.count('x') != 1:
        raise ValueError(f'Image must have two dimensions but had: {args.file_dimensions}')
    __w_min, __h_min = tuple(int(x) for x in args.file_dimensions.split('x'))

    __overwrite = args.overwrite
    __verbose = args.verbose

    print(clean_dataset(input_dir, output_dir, logfile))
