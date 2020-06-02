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
import random
import collections
from collections import Counter, namedtuple

import numpy as np
import pandas as pd
import scanpy.api as sc
import scipy.sparse as sp_sparse
from natsort import natsorted
import tensorflow as tf

sct = collections.namedtuple('sc', ('barcode', 'count_no', 'genes_no', 'dset', 'cluster'))
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

SEED = 0

params_datapath = ''
logger = None

params_pre_cluster_res = 0.15
params_pre_min_cells = 3
params_pre_min_genes = 10
params_pre_scale = 20000
params_pre_test_cells = 0
params_pre_valid_cells = 2000
params_train_split_files = 10

params_train_gene_count = 17789
params_train_dataset_buffer_size = 1000
params_train_dataset_batch_size = 32


def pre_pbmc_convert():
    # MTX files
    # https://math.nist.gov/MatrixMarket/formats.html
    # Format for sparse matrices that encode for the 
    # table coordinates of non zero data only (3 column file - x,y,value)
    # The tsv files encode the rows and columns (barcoded cells and matched genes)

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

    # Load gene names into array
    with open(var_names_path, "r") as var_file:
        var_read = csv.reader(var_file, delimiter='\t')
        var_names = []
        for row in var_read:
            var_names.append(row[1])
    logger.info("Loaded " + str(len(var_names)) + " gene names")

    # Load cell barcodes into array
    with open(obs_names_path, "r") as obs_file:
        obs_read = csv.reader(obs_file, delimiter='\t')
        obs_names = []
        for row in obs_read:
            obs_names.append(row[0])
    logger.info("Loaded " + str(len(obs_names)) + " barcodes")

    # Load count matrix data
    andata = sc.read(data_path) 
    andata = andata.transpose()

    # Make var names unique (appends numbers to duplicates)
    andata.var_names = var_names
    andata.var_names_make_unique()
    andata.obs_names = obs_names
    andata.obs_names_make_unique()

    # Save output
    andata.write(filename=output_h5ad_path)

def pre_pbmc_process():
    # Log
    logger = logging.getLogger("pbmc-process")
    logger.info('Preprocessing pbmc files...')

    # Setup paths
    h5ad_file = "68kPBMCs.h5ad"
    output_h5ad_file = "68kPBMCs_processed.h5ad"
    h5ad_path = os.path.join(params_datapath, h5ad_file)
    output_h5ad_path = os.path.join(params_datapath, output_h5ad_file)
   
    # Load single cell data
    sc_raw = sc.read(h5ad_path)

    # appends -1 -2... to the name of genes that already exist
    # this is already done in previous function
    sc_raw.var_names_make_unique()
    if sp_sparse.issparse(sc_raw.X):
        sc_raw.X = sc_raw.X.toarray()

    # Copy
    clustered = sc_raw.copy()

    # pre-processing
    sc.pp.recipe_zheng17(clustered)
    sc.tl.pca(clustered, n_comps=50)

    # clustering
    sc.pp.neighbors(clustered, n_pcs=50)
    sc.tl.louvain(clustered, resolution=params_pre_cluster_res)

    # add clusters to the raw data
    sc_raw.obs['cluster'] = clustered.obs['louvain']

    # adding clusters' ratios
    cells_per_cluster = Counter(sc_raw.obs['cluster'])
    clust_ratios = dict()
    for key, value in cells_per_cluster.items():
        clust_ratios[key] = value / sc_raw.shape[0]

    # Save
    clusters_ratios = clust_ratios
    clusters_no = len(cells_per_cluster)
    logger.info("Clustering of the raw data is done to %d clusters." % clusters_no)

    # Filter
    sc.pp.filter_cells(sc_raw, min_genes=params_pre_min_genes, copy=False)
    logger.info("Filtering of the raw data is done with minimum %d cells per gene." % params_pre_min_genes)

    sc.pp.filter_genes(sc_raw, min_cells=params_pre_min_cells, copy=False)
    logger.info("Filtering of the raw data is done with minimum %d genes per cell." % params_pre_min_cells)

    # Save
    cells_count = sc_raw.shape[0]
    genes_count = sc_raw.shape[1]
    logger.info("Cells number is %d , with %d genes per cell." % (cells_count, genes_count))

    # Scale
    sc.pp.normalize_per_cell(sc_raw, counts_per_cell_after=params_pre_scale)

    # Setup random
    random.seed(SEED)
    np.random.seed(SEED)

    # Calc how many validation cells there should be per cluster
    valid_cells_per_cluster = {
        key: int(value * params_pre_valid_cells)
        for key, value in clusters_ratios.items()}

    # Calc how many test cells there should be per cluster
    test_cells_per_cluster = {
        key: int(value * params_pre_test_cells)
        for key, value in clusters_ratios.items()}

    # Create a obs column for the split of the train/test valid data
    # Initialise to train and set it as a categorical column
    dataset = np.repeat('train', sc_raw.shape[0])
    unique_groups = np.asarray(['valid', 'test', 'train'])
    sc_raw.obs['split'] = pd.Categorical(values=dataset, categories=natsorted(unique_groups))

    # For each cluster 
    for key in valid_cells_per_cluster:
        # Get all the cells that match the cluster id
        indices = sc_raw.obs[sc_raw.obs['cluster'] == str(key)].index

        # Randomly assign a number of cells to the test and valid sets
        test_valid_indices = np.random.choice(
            indices, valid_cells_per_cluster[key] +
            test_cells_per_cluster[key], replace=False)

        test_indices = test_valid_indices[0:test_cells_per_cluster[key]]
        valid_indices = test_valid_indices[test_cells_per_cluster[key]:]

        # assign the test/training split
        for i in test_indices:
            #sc_raw.obs.set_value(i, 'split', 'test')
            sc_raw.obs.at[i, 'split'] = 'test'

        for i in valid_indices:
            #sc_raw.obs.set_value(i, 'split', 'valid')
            sc_raw.obs.at[i, 'split'] = 'valid'

    # Write final output
    sc_raw.write(output_h5ad_path)

def pre_pbmc_write_tf():
    # Log
    logger = logging.getLogger("pbmc-tf")
    logger.info('Preprocessing pbmc files to TF records...')

    h5ad_file = "68kPBMCs_processed.h5ad"
    h5ad_path = os.path.join(params_datapath, h5ad_file)
    train_filenames = [os.path.join(params_datapath, "tf", 'train-%s.tfrecords' % i)
                                for i in range(params_train_split_files)]
    valid_filename = os.path.join(params_datapath,"tf", 'validate.tfrecords')
    test_filename = os.path.join(params_datapath, "tf", 'test.tfrecords')

    # Load single cell data
    sc_raw = sc.read(h5ad_path)
    cell_count = sc_raw.shape[0]
    gene_count = sc_raw.shape[1]
    logger.info("Cells number is %d , with %d genes per cell." % (cell_count, gene_count))

    # Create TF Writers
    opt = tf.io.TFRecordOptions(compression_type='GZIP')
    valid = tf.io.TFRecordWriter(valid_filename, opt)
    test = tf.io.TFRecordWriter(test_filename, opt)
    train = [tf.io.TFRecordWriter(filename, opt) for filename in train_filenames]

    cat = sc_raw.obs['cluster'].cat.categories
    count = 0
    for line in sc_raw:
        # Doesnt stop for some reason
        if count == cell_count:
            break
        count += 1

        dset = line.obs['split'][0]

        # metadata from line
        scmd = sct(barcode=line.obs_names[0],
            count_no=int(np.sum(line.X)),
            genes_no=line.obs['n_genes'][0],
            dset=dset,
            cluster=line.obs['cluster'][0])

        sc_genes = line.X
        d = scmd

        flat = sc_genes.flatten()
        idx = np.nonzero(flat)[0]
        vals = flat[idx]

        feat_map = {}
        feat_map['indices'] = tf.train.Feature(int64_list=tf.train.Int64List(value=idx))
        feat_map['values'] = tf.train.Feature(float_list=tf.train.FloatList(value=vals))
        feat_map['barcode'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(d.barcode)]))
        feat_map['genes_no'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[d.genes_no]))
        feat_map['count_no'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[d.count_no]))

        # add hot encoding for classification problems
        #feat_map['cluster_1hot'] = tf.train.Feature(
        #    int64_list=tf.train.Int64List(value=[int(c == cat) for c in cat]))
        #feat_map['cluster_int'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(cat)]))

        example = tf.train.Example(features=tf.train.Features(feature=feat_map))
        example_str = example.SerializeToString()

        if d.dset == 'test':
                test.write(example_str)
        elif d.dset == 'train':
                train[random.randint(0, params_train_split_files - 1)].write(example_str)
        elif d.dset == 'valid':
                valid.write(example_str)
        else:
            raise ValueError("invalid dataset: %s" % d.dset)

    # Close the file writers
    test.close()
    valid.close()
    for f in train:
        f.close()

feature_map = {'scg': tf.io.SparseFeature(
        index_key='indices',
        value_key='values',
        dtype=tf.float32,
        size=params_train_gene_count)}

def _parse_example(example_str):
  example = tf.io.parse_single_example(example_str, feature_map)
  return tf.sparse.to_dense(example['scg'])


def train_pbmc():
    print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # find training and validation TF records
    tf_dir = os.path.join(params_datapath, 'tf')
    train_files = [os.path.join(tf_dir, f)
                   for f in os.listdir(tf_dir) if "train" in f]
    valid_files = [os.path.join(tf_dir, f)
                   for f in os.listdir(tf_dir) if "valid" in f]

    print("Found " + str(len(train_files)) + " train files")
    print("Found " + str(len(valid_files)) + " valid files")

    raw_dataset = tf.data.TFRecordDataset(train_files, compression_type='GZIP')

    dataset = raw_dataset.map(_parse_example) \
                         .shuffle(params_train_dataset_buffer_size) \
                         .batch(params_train_dataset_batch_size)
                         #.batch(params_train_dataset_batch_size, drop_remainder=True)




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

    parser.add_argument(
        '--pbmc_process', required=False,
        default=False, action='store_true',
        help='Preprocess PBMC base data')
    
    parser.add_argument(
        '--pbmc_tf', required=False,
        default=False, action='store_true',
        help='Write PBMC data to TF records')

    parser.add_argument(
        '--pbmc_train', required=False,
        default=False, action='store_true',
        help='Train on PBMC data')

    parsedArgs = parser.parse_args()

    params_datapath = parsedArgs.data_path

    if parsedArgs.pbmc_convert:
        pre_pbmc_convert()
        sys.exit

    if parsedArgs.pbmc_process:
        pre_pbmc_process()
        sys.exit

    if parsedArgs.pbmc_tf:
        pre_pbmc_write_tf()
        sys.exit

    if parsedArgs.pbmc_train:
        train_pbmc()
        sys.exit