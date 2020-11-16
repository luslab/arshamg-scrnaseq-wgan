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

from lib.preprocessor import Preprocessor

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_data', required=False, default=False, action='store_true')
    parser.add_argument('--preprocess', required=False, default=False, action='store_true')
    parser.add_argument('--real_analysis', required=False, default=False, action='store_true')
    parser.add_argument('--tf_records', required=False, default=False, action='store_true')
    parser.add_argument('--all_pre', required=False, default=False, action='store_true')
    parsedArgs = parser.parse_args()

    logger = logging.getLogger("arsh-gann")
    logger.info('Initializing')

    p = Preprocessor(logger)
    p.create_directories()

    # Source data downloader
    if parsedArgs.download_data:
        logger.info('Downloading arsham dataset...')
        p.downloadArshamData()
        sys.exit

    # Preprocessor
    if parsedArgs.preprocess:
        logger.info('Preprocessing arsham dataset')
        p.findTranscriptLengths()
        p.preprocessRnaData()
        p.annotateScData()
        p.preprocessScData()
        sys.exit

    # Real data analysis
    if parsedArgs.real_analysis:
        logger.info('Analysing real dataset')
        p.realDataAnalysis()
        sys.exit

    # Create TF records
    if parsedArgs.tf_records:
        p.createTfRecords()
        sys.exit

    if parsedArgs.all_pre:
        logger.info('Downloading arsham dataset...')
        p.downloadArshamData()
        logger.info('Preprocessing arsham dataset')
        p.findTranscriptLengths()
        p.preprocessRnaData()
        p.annotateScData()
        p.preprocessScData()
        logger.info('Analysing real dataset')
        p.realDataAnalysis()
        p.createTfRecords()
        sys.exit