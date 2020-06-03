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
from tensorflow import keras
from tensorflow.keras import layers

#sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_versions()
sc.settings.set_figure_params(dpi=80, facecolor='white')

sct = collections.namedtuple('sc', ('barcode', 'count_no', 'genes_no', 'dset', 'cluster'))
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

SEED = 0

params_datapath = ''
params_training_output = ''
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

params_train_number_epochs = 10
params_train_write_freq = 1

params_example_dataset_batch_size = 500

# NN shape
params_nn_latent_var_size = 100
params_nn_gen_l1_size = 600
params_nn_gen_l2_size = 600
params_nn_gen_l3_size = params_train_gene_count
params_nn_disc_in_size = params_train_gene_count
params_nn_disc_l1_size = 200
params_nn_disc_l2_size = 200

# NN params
params_nn_disc_dropout = 0.3
params_nn_learning_rate = 5e-5
params_nn_gp_lambda = 10

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

def _create_generator():
    model = tf.keras.Sequential()
    
    #L1
    model.add(layers.Dense(params_nn_gen_l1_size, use_bias=False, input_shape=(params_nn_latent_var_size,)))
    model.add(layers.LayerNormalization(epsilon=1e-6))
    model.add(layers.LeakyReLU())
    
    #L2
    model.add(layers.Dense(params_nn_gen_l2_size, use_bias=False))
    model.add(layers.LayerNormalization(epsilon=1e-6))
    model.add(layers.LeakyReLU())
    
    #L3
    model.add(layers.Dense(params_nn_gen_l3_size, use_bias=False))
    model.add(layers.LayerNormalization(epsilon=1e-6))
    model.add(layers.LeakyReLU())
    
    return model

def _create_discriminator():
    model = tf.keras.Sequential()
    
    #L1
    model.add(layers.Dense(params_nn_disc_l1_size, use_bias=False, input_shape=(params_nn_disc_in_size,)))
    model.add(layers.LayerNormalization(epsilon=1e-6))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(params_nn_disc_dropout))
    
    #L2
    model.add(layers.Dense(params_nn_disc_l2_size, use_bias=False))
    model.add(layers.LayerNormalization(epsilon=1e-6))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(params_nn_disc_dropout))
    
    #L3
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    
    return model

def _gen_noise(batch_size):
    noise = tf.random.normal([batch_size, params_nn_latent_var_size])
    return noise

def _generate_profile(gen, batch_size, scale_factor, label, gene_names, epoch):
    # Generate from latent variable space (Gaussian)
    noise = _gen_noise(batch_size)

    # Generate outputs
    generated_profile = gen(noise, training=False)

    # Create formatted dataframe from generator result
    df_gen_prof = pd.DataFrame(generated_profile.numpy()).T
    df_gen_prof = gene_names.join(df_gen_prof, lsuffix='', rsuffix='', how='inner')
    df_gen_prof.index = df_gen_prof.gene_name
    df_gen_prof = df_gen_prof.drop('gene_name', axis=1)
    df_gen_prof = df_gen_prof.add_prefix(label)

    # Scale back to expression data
    df_gen_prof = df_gen_prof * float(params_pre_scale)

    # Get limits
    gen_min = df_gen_prof.min().min()
    gen_max = df_gen_prof.max().max()

    print('Generated prof min/max ' + str(gen_min) + ' - ' + str(gen_max) + " at epoch " + str(epoch))
    
    return df_gen_prof

def _setup_tensorboard():
    # Define tensorboard metrics
    met_gen_loss = tf.keras.metrics.Mean('gen_loss', dtype=tf.float32)
    met_disc_loss = tf.keras.metrics.Mean('disc_loss', dtype=tf.float32)

    # Create log directories and tensor board summaries
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
   
    all_loss_log_path = os.path.join(params_training_output, "logs", "gradient_tape", current_time, "all_train")
    gen_loss_log_path = os.path.join(params_training_output, "logs", "gradient_tape", current_time, "gen_train")
    disc_loss_log_path = os.path.join(params_training_output, "logs", "gradient_tape", current_time, "disc_train")
    image_log_path = os.path.join(params_training_output, "logs", "images", current_time, "training")

    all_summary_writer = tf.summary.create_file_writer(all_loss_log_path)
    gen_summary_writer = tf.summary.create_file_writer(gen_loss_log_path)
    disc_summary_writer = tf.summary.create_file_writer(disc_loss_log_path)
    image_summary_writer = tf.summary.create_file_writer(image_log_path)

    return met_gen_loss, met_disc_loss, all_summary_writer, gen_summary_writer, disc_summary_writer, image_summary_writer

def _generate_example_profiles(generator, df_gene_names, sc_test, epoch):
    # Generate cell examples
    df_gen_prof = _generate_profile(generator, params_example_dataset_batch_size, params_pre_scale, 'gencell_ep' + str(epoch) + '_', df_gene_names, epoch)
    gen_prof_path = os.path.join(params_training_output, "gen_profiles","gen_prof_" + str(epoch) + ".csv")
    df_gen_prof.to_csv(gen_prof_path)

    # Load and format the generated cell profiles
    sc_gen = sc.read_csv(gen_prof_path, first_column_names=True)
    sc_gen = sc_gen.transpose()
    dataset_label = np.repeat('gen', sc_gen.shape[0])
    sc_gen.obs['dataset'] = dataset_label

    # Merge with raw
    sc_combined = sc_test.concatenate(sc_gen)

    # Create plot
    sc.pp.neighbors(sc_combined)
    sc.tl.umap(sc_combined)
    sc.pl.umap(sc_combined, save="_" + str(epoch) + ".png", show=False, color='dataset')

    # Load and record the image in a summary
    image = tf.io.read_file(os.path.join(params_training_output, "figures", "umap_" + str(epoch) + ".png"))
    image = tf.image.decode_png(image)
    image = np.reshape(image, (1, image.shape[0], image.shape[1], 4))

    return image

@tf.function
def _train_step(gen, disc, gen_opt, disc_opt, met_gen_loss, met_disc_loss, real_profiles):
    with tf.GradientTape(persistent=True) as tape:
        # Generate noise
        noise = _gen_noise(real_profiles.shape[0])

        # Generate some fake profiles using noise
        generated_profiles = gen(noise, training=True)
        
        # Pass the real profiles and the fake profiles through the disc
        real_output = disc(real_profiles, training=True)
        fake_output = disc(generated_profiles, training=True)

        # Calc loss
        dloss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
        gloss = -tf.reduce_mean(fake_output)

        # Calculate interpolated profile 
        shape = [tf.shape(real_profiles)[0]] + [1] * (real_profiles.shape.ndims - 1)
        epsilon = tf.random.uniform(shape=shape, minval=0., maxval=1.)
        interpolated_profiles = real_profiles + epsilon * (generated_profiles - real_profiles)

        # Run through disc with nested gradient tape
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(interpolated_profiles)
            d_interpolated = disc(interpolated_profiles, training=True)

        # Compute gradient penalty
        grad = tape2.gradient(d_interpolated, interpolated_profiles)
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        gradient_penalty = tf.reduce_mean((norm - 1.)**2)

        # Calculate adjusted loss
        gploss = dloss + (params_nn_gp_lambda * gradient_penalty)

    # Save the gradients
    gradients_of_generator = tape.gradient(gloss, gen.trainable_variables)
    gradients_of_discriminator = tape.gradient(gploss, disc.trainable_variables)

    # Apply the gradients
    gen_opt.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
    disc_opt.apply_gradients(zip(gradients_of_discriminator, disc.trainable_variables))

    met_disc_loss(gploss)
    met_gen_loss(gloss)

def train_pbmc():
    # Log
    logger = logging.getLogger("pbmc-train")
    logger.info('Training using PBMC data')

    logger.info("Data path: " + params_datapath)
    logger.info("Output path: " + params_training_output)
    logger.info("Epochs: " + str(params_train_number_epochs))
    logger.info("Write freq: " + str(params_train_write_freq))

    # Set image directory
    sc.settings.figdir = os.path.join(params_training_output, "figures")

    # Resolve paths
    h5ad_file = "68kPBMCs_processed.h5ad"
    h5ad_path = os.path.join(params_datapath, h5ad_file)

    # Load single cell data
    sc_raw = sc.read(h5ad_path)
    cell_count = sc_raw.shape[0]
    gene_count = sc_raw.shape[1]
    logger.info("Cells number is %d , with %d genes per cell." % (cell_count, gene_count))

    # Add dataset column
    dataset_label = np.repeat('pbmc', sc_raw.shape[0])
    sc_raw.obs['dataset'] = dataset_label

    # Get the validation data
    sc_test = sc_raw[sc_raw.obs['split'] == "valid", :]

    # Create gene name array
    df_gene_names = pd.DataFrame(sc_raw.var_names)
    df_gene_names.columns = ['gene_name']

    # Log TF computing resources
    print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # find training and validation TF records
    tf_dir = os.path.join(params_datapath, 'tf')
    train_files = [os.path.join(tf_dir, f)
                   for f in os.listdir(tf_dir) if "train" in f]
    valid_files = [os.path.join(tf_dir, f)
                   for f in os.listdir(tf_dir) if "valid" in f]

    logger.info("Found " + str(len(train_files)) + " train files")
    logger.info("Found " + str(len(valid_files)) + " valid files")

    # Load training set
    raw_dataset = tf.data.TFRecordDataset(train_files, compression_type='GZIP')

    # Create and shuffle dataset
    train_dataset = raw_dataset.map(_parse_example) \
                         .shuffle(params_train_dataset_buffer_size) \
                         .batch(params_train_dataset_batch_size)
                         #.batch(params_train_dataset_batch_size, drop_remainder=True)

    # Create generator and discriminator
    generator = _create_generator()
    discriminator = _create_discriminator()

    # Print summary
    #generator.summary()
    #discriminator.summary()

    # Create optimizers
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=params_nn_learning_rate, amsgrad=True)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=params_nn_learning_rate, amsgrad=True)

    # Create checkpoints
    checkpoint_prefix = os.path.join(params_training_output, "training_checkpoints", "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

    # Setup tensorboard
    met_gen_loss, met_disc_loss, all_summary_writer, gen_summary_writer, disc_summary_writer, image_summary_writer = _setup_tensorboard()

    # Training loop
    for epoch in range(params_train_number_epochs):
        logger.info("Epoch " + str(epoch))

        # Save checkpoints and gen example data
        if epoch % params_train_write_freq == 0:
            # Save checkpoint
            checkpoint.save(file_prefix = checkpoint_prefix)

            # Generate image set for epoch
            image = _generate_example_profiles(generator, df_gene_names, sc_test, epoch)
            with image_summary_writer.as_default():
                tf.summary.image("Generated profile UMAP", image, step=epoch)

        for data_batch in train_dataset:
            _train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, met_gen_loss, met_disc_loss, data_batch)

        with all_summary_writer.as_default():
            tf.summary.scalar('2_gen_loss', met_gen_loss.result(), step=epoch)
            tf.summary.scalar('3_disc_loss', met_disc_loss.result(), step=epoch)
    
        with gen_summary_writer.as_default():
            tf.summary.scalar('1_loss', met_gen_loss.result(), step=epoch)

        with disc_summary_writer.as_default():
            tf.summary.scalar('1_loss', met_disc_loss.result(), step=epoch)

        # Reset metrics every epoch
        met_gen_loss.reset_states()
        met_disc_loss.reset_states()

    # Final cell generation output
    image = _generate_example_profiles(generator, df_gene_names, sc_test, epoch)
    with image_summary_writer.as_default():
        tf.summary.image("Generated profile UMAP", image, step=epoch)

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

    parser.add_argument(
        '--epochs', required=False,
        default=1, action='store_true',
        help='Number of epochs to train on')

    parser.add_argument(
        '--write_freq', required=False,
        default=1, action='store_true',
        help='Frequency to write logging data')
    
    parser.add_argument(
        '--training_output', required=False,
        default='', action='store_true',
        help='')

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
        params_train_number_epochs = parsedArgs.epochs
        params_train_write_freq = parsedArgs.write_freq
        params_training_output = parsedArgs.training_output
        train_pbmc()
        sys.exit