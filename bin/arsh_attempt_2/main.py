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
from lib.nn import Net

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_data', required=False, default=False, action='store_true')
    parser.add_argument('--preprocess', required=False, default=False, action='store_true')
    parser.add_argument('--real_analysis', required=False, default=False, action='store_true')
    parser.add_argument('--tf_records', required=False, default=False, action='store_true')
    parser.add_argument('--all_pre', required=False, default=False, action='store_true')
    parser.add_argument('--train', required=False, default=False, action='store_true')
    parser.add_argument('--create_movie', required=False, default=False, action='store_true')

    parser.add_argument('--epochs', required=False, default=1)
    parser.add_argument('--write_freq', required=False, default=1)
    parser.add_argument('--output_dir', required=False, default='output')
    parser.add_argument('--data_dir', required=False, default='data')
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

    # All pre-processing 
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

    if parsedArgs.train:
        logger.info('Training nn...')
        number_epochs = int(parsedArgs.epochs)
        write_freq = int(parsedArgs.write_freq)
        output_dir = parsedArgs.output_dir
        data_dir = parsedArgs.data_dir

        n = Net(logger, number_epochs, write_freq, output_dir, data_dir)
        n.create_directories()
        n.train()
        sys.exit

    # if parsedArgs.movie:
    #     params_training_output = parsedArgs.training_output
    #     _create_movie_from_images()
    #     sys.exit