#!/usr/bin/env python
# coding: utf-8

import logging
import datetime
import os
import glob
import pathlib

import numpy as np
import pandas as pd
import scanpy as sc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

from .scalecell import ScaleCell

class Net:
    # Paths
    tf_dir = 'tf'
    figure_dir = 'figures/'
    profile_dir = 'profiles'
    input_data = 'all_preprocessed.h5ad'

    # NN Params
    train_gene_count = 7296 # Number of input genes TODO: parameterize this from the params file generated during preprocessing
    train_dataset_buffer_size = 1000 # Number of examples to buffer when loading training data
    train_dataset_batch_size = 32 # Batch size for training
    train_number_epochs = 1 # Epochs to run
    train_write_freq = 1 # How often to write a checkpoint and save some example images
    example_dataset_batch_size = 500 # How many profiles to generate for an example dataset

    params_pre_scale = 1000000

    # NN shape
    nn_latent_var_size = 100
    nn_gen_l1_size = 600
    nn_gen_l2_size = 600
    nn_gen_l3_size = train_gene_count
    nn_disc_in_size = train_gene_count
    nn_disc_l1_size = 200
    nn_disc_l2_size = 200

    # NN params
    nn_epsilon = 1e-6
    nn_disc_dropout = 0.3
    nn_learning_rate = 0.001
    nn_gp_lambda = 10

    # Misc
    util_video_framerate = 2

    feature_map = {'scg': tf.io.SparseFeature(
        index_key='indices',
        value_key='values',
        dtype=tf.float32,
        size=train_gene_count)}

    def __init__(self, logger, number_epochs, write_freq, output_dir, data_dir):
        self.logger = logger
        self.train_number_epochs = number_epochs
        self.train_write_freq = write_freq
        self.output_dir = output_dir
        self.data_dir = data_dir
        
        #sc.settings.autosave=True
        sc.settings.autoshow=False
        #sc.settings.figdir=path.join(self.output_dir, self.figure_dir)
        sc.settings.verbosity=3
        sc.set_figure_params(format='png')

    def create_movie_from_images(self):
        img_array = []

        for filename in glob.glob(os.path.join(self.output_dir, "figures", "*.png")):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
 
        out = cv2.VideoWriter(os.path.join(self.output_dir, "umap_training.avi"), cv2.VideoWriter_fourcc(*'XVID'), self.util_video_framerate, size)
 
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

    def create_directories(self):
        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(self.output_dir, self.figure_dir)).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(self.output_dir, self.profile_dir)).mkdir(parents=True, exist_ok=True)
        #pathlib.Path(path.join(self.data_dir, self.tf_dir)).mkdir(parents=True, exist_ok=True)

    def setup_tensorboard(self):
        # Define tensorboard metrics
        met_gen_loss = tf.keras.metrics.Mean('gen_loss', dtype=tf.float32)
        met_disc_loss = tf.keras.metrics.Mean('disc_loss', dtype=tf.float32)

        # Create log directories and tensor board summaries
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
   
        log_path = os.path.join(self.output_dir, "logs", "gradient_tape", current_time)
        all_loss_log_path = os.path.join(log_path, "all_train")
        gen_loss_log_path = os.path.join(log_path, current_time, "gen_trai- n")
        disc_loss_log_path = os.path.join(log_path, current_time, "disc_train")
        image_log_path = os.path.join(self.output_dir, "logs", "images", current_time, "training")

        all_summary_writer = tf.summary.create_file_writer(all_loss_log_path)
        gen_summary_writer = tf.summary.create_file_writer(gen_loss_log_path)
        disc_summary_writer = tf.summary.create_file_writer(disc_loss_log_path)
        image_summary_writer = tf.summary.create_file_writer(image_log_path)

        return met_gen_loss, met_disc_loss, all_summary_writer, gen_summary_writer, disc_summary_writer, image_summary_writer

    def parse_example(self, example_str):
        example = tf.io.parse_single_example(example_str, self.feature_map)
        return tf.sparse.to_dense(example['scg'])

    def create_generator(self):
        model = tf.keras.Sequential()
    
        #L1
        model.add(layers.Dense(self.nn_gen_l1_size, use_bias=False, input_shape=(self.nn_latent_var_size,)))
        model.add(layers.LayerNormalization(epsilon=self.nn_epsilon))
        model.add(layers.LeakyReLU())
    
        #L2
        model.add(layers.Dense(self.nn_gen_l2_size, use_bias=False))
        model.add(layers.LayerNormalization(epsilon=self.nn_epsilon))
        model.add(layers.LeakyReLU())
    
        #L3
        model.add(layers.Dense(self.nn_gen_l3_size, use_bias=False))
        model.add(layers.LayerNormalization(epsilon=self.nn_epsilon))
        model.add(layers.LeakyReLU())

        # Add scaling layer
        #model.add(ScaleCell(scale_factor=self.params_pre_scale))
    
        return model

    def create_discriminator(self):
        model = tf.keras.Sequential()
    
        #L1
        model.add(layers.Dense(self.nn_disc_l1_size, use_bias=False, input_shape=(self.nn_disc_in_size,)))
        model.add(layers.LayerNormalization(epsilon=self.nn_epsilon))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.nn_disc_dropout))
    
        #L2
        model.add(layers.Dense(self.nn_disc_l2_size, use_bias=False))
        model.add(layers.LayerNormalization(epsilon=self.nn_epsilon))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.nn_disc_dropout))
    
        #L3
        model.add(layers.Flatten())
        model.add(layers.Dense(1))
    
        return model

    def gen_noise(self, batch_size):
        noise = tf.random.normal([batch_size, self.nn_latent_var_size])
        return noise

    def generate_profile_batch(self, gen, batch_size, label, gene_names, epoch):
        # Generate from latent variable space (Gaussian)
        noise = self.gen_noise(batch_size)

        # Generate outputs
        generated_profile = gen(noise, training=False)

        # Create formatted dataframe from generator result
        df_gen_prof = pd.DataFrame(generated_profile.numpy()).T
        df_gen_prof = gene_names.join(df_gen_prof, lsuffix='', rsuffix='', how='inner')
        df_gen_prof.index = df_gen_prof.gene_name
        df_gen_prof = df_gen_prof.drop('gene_name', axis=1)
        df_gen_prof = df_gen_prof.add_prefix(label)

        # Scale back to expression data
        #df_gen_prof = df_gen_prof * float(self.params_pre_scale)

        # Get limits
        gen_min = df_gen_prof.min().min()
        gen_max = df_gen_prof.max().max()

        self.logger.info('Generated prof min/max ' + str(gen_min) + ' - ' + str(gen_max) + " at epoch " + str(epoch))
    
        return df_gen_prof

    def generate_example_profiles(self, generator, df_gene_names, sc_test, epoch):
        # Generate cell examples
        df_gen_prof = self.generate_profile_batch(generator, self.example_dataset_batch_size, 'gencell_ep' + str(epoch) + '_', df_gene_names, epoch)
        gen_prof_path = os.path.join(self.output_dir, self.profile_dir,"gen_prof_" + str(epoch) + ".csv")
        df_gen_prof.to_csv(gen_prof_path)

        # Load and format the generated cell profiles
        sc_gen = sc.read_csv(gen_prof_path, first_column_names=True)
        sc_gen = sc_gen.transpose()
        dataset_label = np.repeat('gen', sc_gen.shape[0])
        sc_gen.obs['dataset'] = dataset_label
        sc_gen.obs['split'] = dataset_label

        # Merge with raw
        sc_combined = sc_test.concatenate(sc_gen)

        # Create plot
        sc.pp.neighbors(sc_combined)
        sc.tl.umap(sc_combined)
        sc.pl.umap(sc_combined, save="_" + str(epoch) + ".png", color='dataset')

        # Load and record the image in a summary
        image = tf.io.read_file(os.path.join(self.output_dir, "figures", "umap_" + str(epoch) + ".png"))
        image = tf.image.decode_png(image)
        image = np.reshape(image, (1, image.shape[0], image.shape[1], 4))

        return image

    @tf.function
    def train_step(self, gen, disc, gen_opt, disc_opt, met_gen_loss, met_disc_loss, real_profiles):
        with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:
            # Generate noise
            noise = self.gen_noise(real_profiles.shape[0])

            # Generate some fake profiles using noise
            generated_profiles = gen(noise, training=True)
        
            # Pass the real profiles and the fake profiles through the disc
            real_output = disc(real_profiles, training=True)
            fake_output = disc(generated_profiles, training=True)

            # Calculate interpolated profile 
            shape = [tf.shape(real_profiles)[0]] + [1] * (real_profiles.shape.ndims - 1)
            epsilon = tf.random.uniform(shape=shape, minval=0.0, maxval=1.0)
            #interpolated_profiles = real_profiles + epsilon * (generated_profiles - real_profiles)
            interpolated_profiles = epsilon * real_profiles + (1.0 - epsilon) * generated_profiles

            # Run through disc with nested gradient tape
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(interpolated_profiles)
                d_interpolated = disc(interpolated_profiles, training=True)

            # Compute gradient penalty
            grad = tape2.gradient(d_interpolated, interpolated_profiles)
            norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
            gradient_penalty = tf.reduce_mean((norm - 1.0) ** 2)

            # Calc loss
            dloss = tf.reduce_mean(real_output) - tf.reduce_mean(fake_output)
            gloss = tf.reduce_mean(fake_output)

            # Calculate adjusted loss
            gploss = dloss + (self.nn_gp_lambda * gradient_penalty)

        # Save the gradients
        gradients_of_generator = gen_tape.gradient(gloss, gen.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(gploss, disc.trainable_variables)

        # Apply the gradients
        gen_opt.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
        disc_opt.apply_gradients(zip(gradients_of_discriminator, disc.trainable_variables))

        met_disc_loss(gploss)
        met_gen_loss(gloss)

    def train(self):
        # Log
        self.logger.info('Training...')

        self.logger.info("Data path: " +  os.path.abspath(self.data_dir))
        self.logger.info("Output path: " + os.path.abspath(self.output_dir))
        self.logger.info("Epochs: " + str(self.train_number_epochs))
        self.logger.info("Write freq: " + str(self.train_write_freq))

        # Set image directory
        sc.settings.figdir = os.path.join(self.output_dir, "figures")

        # Resolve data path
        h5ad_path = os.path.join(self.data_dir, self.input_data)

        # Load single cell data
        sc_raw = sc.read(h5ad_path)
        cell_count = sc_raw.shape[0]
        gene_count = sc_raw.shape[1]
        self.logger.info("Cells number is %d , with %d genes per cell." % (cell_count, gene_count))

        # Get the validation data
        sc_test = sc_raw
        #sc_test = sc_raw[sc_raw.obs['split'] == "valid", :]

        # Create gene name array
        df_gene_names = pd.DataFrame(sc_raw.var_names)
        df_gene_names.columns = ['gene_name']

        # Log TF computing resources
        print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

        # find training and validation TF records
        tf_dir = os.path.join(self.data_dir, self.tf_dir)
        train_files = [os.path.join(tf_dir, f)
            for f in os.listdir(tf_dir) if "train" in f]
        valid_files = [os.path.join(tf_dir, f)
            for f in os.listdir(tf_dir) if "valid" in f]

        self.logger.info("Found " + str(len(train_files)) + " train files")
        self.logger.info("Found " + str(len(valid_files)) + " valid files")

        # Load training set
        raw_dataset = tf.data.TFRecordDataset(train_files, compression_type='GZIP')
        #raw_dataset = tf.data.TFRecordDataset(valid_files, compression_type='GZIP')

        # Create and shuffle dataset
        train_dataset = raw_dataset.map(self.parse_example) \
            .shuffle(self.train_dataset_buffer_size) \
            .batch(self.train_dataset_batch_size)

        # Create generator and discriminator
        generator = self.create_generator()
        discriminator = self.create_discriminator()

        # Print summary
        #generator.summary()
        #discriminator.summary()

        # Create optimizers
        # generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.nn_learning_rate, amsgrad=True)
        # discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.nn_learning_rate, amsgrad=True)
        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, amsgrad=True)
        discriminator_optimizer =  tf.keras.optimizers.RMSprop(learning_rate=0.0005)


        # Create checkpoints
        checkpoint_prefix = os.path.join(self.output_dir, "training_checkpoints", "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                         discriminator_optimizer=discriminator_optimizer,
                                         generator=generator,
                                         discriminator=discriminator)

        # Setup tensorboard
        met_gen_loss, met_disc_loss, all_summary_writer, gen_summary_writer, disc_summary_writer, image_summary_writer = self.setup_tensorboard()

        # Training loop
        for epoch in range(self.train_number_epochs):
            # Save checkpoints and gen example data
            if epoch % self.train_write_freq == 0:
                self.logger.info("Epoch " + str(epoch))

                # Save checkpoint
                checkpoint.save(file_prefix = checkpoint_prefix)

                # Generate image set for epoch
                image = self.generate_example_profiles(generator, df_gene_names, sc_test, epoch)
                with image_summary_writer.as_default():
                    tf.summary.image("Generated profile UMAP", image, step=epoch)

            for data_batch in train_dataset:
                self.train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, met_gen_loss, met_disc_loss, data_batch)

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
        if epoch % self.train_write_freq != 0:
            image = self.generate_example_profiles(generator, df_gene_names, sc_test, epoch)
            with image_summary_writer.as_default():
                tf.summary.image("Generated profile UMAP", image, step=epoch)
    
        # Output movie
        #create_movie_from_images()