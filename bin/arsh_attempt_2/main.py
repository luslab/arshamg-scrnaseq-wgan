#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import logging
import datetime
import os

from lib.downloader import Downloader

if __name__ == '__main__':
    """
    Main comment
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--download_data', required=False, default=False, action='store_true')

    parsedArgs = parser.parse_args()

    # Source data downloader
    if parsedArgs.download_data:
        d = Downloader()
        d.downloadArshamData()
        sys.exit