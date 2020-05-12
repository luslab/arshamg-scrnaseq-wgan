#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

if __name__ == '__main__':
    """
    Main comment
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataPath', required=False,
                        help='Path to working folder')

    parser.add_argument(
        '--pbmc-convert', required=False,
        default=False, action='store_true',
        help='Pre convert PBMC base data')

    parsedArgs = parser.parse_args()