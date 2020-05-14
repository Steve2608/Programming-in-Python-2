import argparse

import numpy as np

from EX3.ex3 import ImageNormalizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Programming in Python 2, Exercise 3')
    parser.add_argument('input_dir', type=str,
                        help='Relative or absolute path to input-directory. Path must point to a '
                             'directory. The directory must exist.')
    args = parser.parse_args()

    imag_norm = ImageNormalizer(args.input_dir)
    print(imag_norm)

    print('means/stds:', *imag_norm.stats, sep='\n', end='\n\n')

    print('sum(means):', sum([imag.mean() for imag in imag_norm.images]))
    print('mean(vars):', np.mean([imag.var() for imag in imag_norm.images]))
