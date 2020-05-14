import argparse

from EX2.ex2 import ex2
from EX2.ex2 import _file_types, _min_file_size, _w_min, _h_min, _overwrite, _verbose

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
