#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import logging
import os
import csv

import scanpy.api as sc

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

params_datapath = ''
logger = None

def pre_pbmc_convert():
    # Log
    logger = logging.getLogger("pbmc-conv")
    logger.info('Converting pbmc files...')

    # Setup paths
    data_file = "matrix.mtx"
    var_names_file = "genes.tsv"
    obs_names_file = "barcodes.tsv"
    output_h5ad_file = "68kPBMCs.h5ad"

    data_path = os.path.join(params_datapath,data_file)
    var_names_path = os.path.join(params_datapath,var_names_file)
    obs_names_path = os.path.join(params_datapath,obs_names_file)
    output_h5ad_path = os.path.join(params_datapath,output_h5ad_file)

    # Log
    logger.info(data_path)
    logger.info(var_names_path)
    logger.info(obs_names_path)
    logger.info(output_h5ad_path)

    # Load gene names
    with open(var_names_path, "r") as var_file:
        var_read = csv.reader(var_file, delimiter='\t')
        var_names = []
        for row in var_read:
            var_names.append(row[1])
    logger.info("Loaded " + str(len(var_names)) + " gene names")

    # Load UMIs
    with open(obs_names_path, "r") as obs_file:
        obs_read = csv.reader(obs_file, delimiter='\t')
        obs_names = []
        for row in obs_read:
            obs_names.append(row[0])
    logger.info("Loaded " + str(len(obs_names)) + " UMIs")

    # Load 10x data and convert
    andata = sc.read(data_path) 
    andata = andata.transpose()

    # Make var names unique
    andata.var_names = var_names
    andata.var_names_make_unique()
    andata.obs_names = obs_names
    andata.obs_names_make_unique()

    # Save output
    andata.write(filename=output_h5ad_path)

if __name__ == '__main__':
    """
    Main comment
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', required=False,
                        help='Path to working folder')

    parser.add_argument(
        '--pbmc_convert', required=False,
        default=False, action='store_true',
        help='Pre convert PBMC base data')

    parsedArgs = parser.parse_args()

    params_datapath = parsedArgs.data_path

    if parsedArgs.pbmc_convert:
        pre_pbmc_convert()
        sys.exit()