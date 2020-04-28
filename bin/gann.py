#!/usr/bin/env python
# coding: utf-8

# # GANN - CHECK SPELLING
# 
# ## Setup
# 
# Load modules


# Load the TensorBoard notebook extension
#get_ipython().run_line_magic('load_ext', 'tensorboard')

# Clear any logs from previous runs
#get_ipython().system('rm -rf ./logs/ ')


from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import time
import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# Setup paths
#train_feature_path = 'tpm_combined.csv'
#train_gene_name_path = 'tpm_combined_rows.csv'
#train_cell_name_path = 'tpm_combined_cols.csv'
#test_feature_path = 'tpm_combined_test.csv'
#test_gene_name_path = 'tpm_combined_rows_test.csv'
#test_cell_name_path = 'tpm_combined_cols_test.csv'
#train_nonorm_path = 'tpm_combined_train_nonorm.csv'

#tpm_combined.csv, 
#tpm_combined_cols.csv, 
#tpm_combined_cols_test.csv,
#tpm_combined_rows.csv,
#tpm_combined_rows_test.csv, /
#tpm_combined_test.csv, 
#tpm_combined_test_nonorm.csv,
#tpm_combined_train_nonorm.csv]

train_feature_path = sys.argv[1]
train_gene_name_path = sys.argv[4]
train_cell_name_path = sys.argv[2]
test_feature_path = sys.argv[6]
test_gene_name_path = sys.argv[5]
test_cell_name_path = sys.argv[3]
train_nonorm_path = sys.argv[8]

# ## Load data
# 
# Load datasets into frames and check all the shapes match up

df_gene_names = pd.read_csv(train_gene_name_path, header=None)
df_cell_names = pd.read_csv(train_cell_name_path, header=None)
df_training_data = pd.read_csv(train_feature_path, header=None)

df_gene_names.columns = ['gene_name']

#print(df_gene_names.shape)
#print(df_cell_names.shape)
#print(df_training_data.shape)

df_training_data_nonorm = pd.read_csv(train_nonorm_path)
df_training_data_nonorm = df_training_data_nonorm.drop('gene_name', axis=1)

nonorm_max = df_training_data_nonorm.max().max()
nonorm_min = df_training_data_nonorm.min().min()
del df_training_data_nonorm

#print(nonorm_max)
#print(nonorm_min)


# Load test data
df_gene_names_test = pd.read_csv(test_gene_name_path, header=None)
df_cell_names_test = pd.read_csv(test_cell_name_path, header=None)
df_test_data = pd.read_csv(test_feature_path, header=None)

#print(df_gene_names_test.shape)
#print(df_cell_names_test.shape)
#print(df_test_data.shape)


# The number of genes in the input dataset determines the generator output as well as the dicriminator inputs
num_genes = df_gene_names.shape[0]
df_gene_names.shape

# ## Define model variables

# Model params
LATENT_VARIABLE_SIZE = 100
GEN_L1_DENSE_SIZE = 600
GEN_L2_DENSE_SIZE = 600
GEN_L3_DENSE_SIZE = num_genes

DIS_INPUT_SIZE = num_genes
DIS_L1_DENSE_SIZE = 200
DIS_L2_DENSE_SIZE = 200

NOISE_STDEV = 0.1
POISSON_LAM = 1

# Training params
TRAIN_BATCH_SIZE = 10
TRAIN_BUFFER_SIZE = 10000
TEST_BATCH_SIZE = 500
TEST_BUFFER_SIZE = 500
GEN_BATCH_SIZE = 10
EPOCHS = 10

#LEARNING_RATE = 0.001
LEARNING_RATE = 1e-5

EX_GEN_BATCH_SIZE = 500
WRITE_FREQ = 100

# Get ext param
EPOCHS = int(sys.argv[9])
WRITE_FREQ = int(sys.argv[10])

# ## Create training and test datasets
# 
# Create tensors from training data - Convert to Int32 for better work on GPU with batch and shuffle

train_dataset = tf.data.Dataset.from_tensor_slices(df_training_data.T.values.astype('float32')).shuffle(TRAIN_BUFFER_SIZE).batch(TRAIN_BATCH_SIZE)
#print(train_dataset)


test_dataset = tf.data.Dataset.from_tensor_slices(df_test_data.T.values.astype('float32')).shuffle(TEST_BUFFER_SIZE).batch(TEST_BATCH_SIZE)
#print(test_dataset)


# ## Define GANN model
# 
# Define function for contructing the generator
def create_generator():
    model = tf.keras.Sequential()
    
    #L1
    model.add(layers.Dense(GEN_L1_DENSE_SIZE, use_bias=False, input_shape=(LATENT_VARIABLE_SIZE,)))
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #assert model.output_shape == (None, GEN_L1_DENSE_SIZE, 1)  # Note: None is the batch size
    
    #L2
    model.add(layers.Dense(GEN_L2_DENSE_SIZE, use_bias=False))
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #assert model.output_shape == (None, GEN_L2_DENSE_SIZE, 1)
    
    #L3
    model.add(layers.Dense(GEN_L3_DENSE_SIZE, use_bias=False))
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #assert model.output_shape == (None, GEN_L3_DENSE_SIZE, 1)
    
    return model


# Define function for constructing discriminator
def create_discriminator():
    model = tf.keras.Sequential()
    
    #L1
    model.add(layers.Dense(DIS_L1_DENSE_SIZE, use_bias=False, input_shape=(DIS_INPUT_SIZE,)))
    model.add(layers.LeakyReLU())
    #model.add(layers.Dropout(0.3))
    
    #L2
    model.add(layers.Dense(DIS_L2_DENSE_SIZE, use_bias=False))
    model.add(layers.LeakyReLU())
    #model.add(layers.Dropout(0.3))
    
    #L3
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    
    return model


# Define the noise generation function
def gen_noise(batch_size):
    # Create some random noise for the generator
    n_noise = tf.random.normal([batch_size, LATENT_VARIABLE_SIZE], mean=0.0, stddev=NOISE_STDEV)
    p_noise = tf.random.poisson([batch_size, LATENT_VARIABLE_SIZE], lam=POISSON_LAM)
    noise = tf.abs(n_noise + p_noise)
    return noise


# Define the loss functions

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
    
    #total_loss = tf.reduce_mean(real_output) - tf.reduce_mean(fake_output)
    #return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
    #total_loss = -tf.reduce_mean(fake_output)
    #return total_loss


# Util functions

def data_frame_from_gen(profile, label):
    # Create formatted dataframe from generator result
    df_gen_prof = pd.DataFrame(generated_profile.numpy()).T
    df_gen_prof = df_gene_names.join(df_gen_prof, lsuffix='', rsuffix='', how='inner')
    df_gen_prof.index = df_gen_prof.gene_name
    df_gen_prof = df_gen_prof.drop('gene_name', axis=1)
    df_gen_prof = df_gen_prof.add_prefix(label)

    # Get limits
    gen_min = df_gen_prof.min().min()
    gen_max = df_gen_prof.max().max()

    # Scale everything up to 0
    df_gen_prof = df_gen_prof + (gen_min*-1)
    gen_max = df_gen_prof.max().max()
    gen_min = df_gen_prof.min().min()

    # Rescale to between real world min maxes
    df_gen_prof = df_gen_prof / gen_max
    df_gen_prof = df_gen_prof * nonorm_max
    
    return df_gen_prof


# ## Define the training loops

# Input is a batch of real cell profiles from the training set
# @tf.function
def train_step(cell_profiles):
    noise = gen_noise(GEN_BATCH_SIZE)
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_profiles = generator(noise, training=True)
        
        real_output = discriminator(cell_profiles, training=True)
        fake_output = discriminator(generated_profiles, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        
        met_gen_loss(gen_loss)
        met_disc_loss(disc_loss)
        
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return


# ## Create GANN model
# 
# Create generator and discriminator
generator = create_generator()
discriminator = create_discriminator()


# Define optimizer
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-07)


# ## Create checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# ## Generate from test data to check network
# Generate a single test set
noise = gen_noise(EX_GEN_BATCH_SIZE)
generated_profile = generator(noise, training=False)
df_gen_prof_1 = data_frame_from_gen(generated_profile, 'gencell_ep0_')

# ## Train the GANN
# 
# Define tensorboard metrics
met_gen_loss = tf.keras.metrics.Mean('gen_loss', dtype=tf.float32)
met_disc_loss = tf.keras.metrics.Mean('disc_loss', dtype=tf.float32)
met_test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)


# Create log directories
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

gen_log_dir = 'logs/gradient_tape/' + current_time + '/gen_train'
disc_log_dir = 'logs/gradient_tape/' + current_time + '/disc_train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/disc_test'
all_log_dir = 'logs/gradient_tape/' + current_time + '/all'

all_summary_writer = tf.summary.create_file_writer(all_log_dir)
gen_summary_writer = tf.summary.create_file_writer(gen_log_dir)
disc_summary_writer = tf.summary.create_file_writer(disc_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


# Run the training model
print('Running...')

for epoch in range(EPOCHS):
    
    # Save checkpoints and gen example data
    if epoch % WRITE_FREQ == 0:   
        checkpoint.save(file_prefix = checkpoint_prefix)
        
        # Generate a profile set
        noise = gen_noise(EX_GEN_BATCH_SIZE)
        generated_profile = generator(noise, training=False)
        df_gen_prof = data_frame_from_gen(generated_profile, 'gencell_ep' + str(epoch) + '_')
        df_gen_prof.to_csv('gen/gen_prof_' + str(epoch) + '.csv')
    
    # Logging
    start = time.time()
    
    #Train the epoch
    for data_batch in train_dataset:
        train_step(data_batch)
        
    #Run test data through discriminator
    for data_batch in test_dataset:
        test_decision = discriminator(data_batch, training=False)

    test_loss = cross_entropy(tf.ones_like(test_decision), test_decision)
    met_test_loss(test_loss)
    
    #Log metrics
    with all_summary_writer.as_default():
        tf.summary.scalar('2_gen_loss', met_gen_loss.result(), step=epoch)
        tf.summary.scalar('3_disc_loss', met_disc_loss.result(), step=epoch)
        tf.summary.scalar('3_test_loss', met_test_loss.result(), step=epoch)
    
    with gen_summary_writer.as_default():
        tf.summary.scalar('1_loss', met_gen_loss.result(), step=epoch)
           
    with disc_summary_writer.as_default():
        tf.summary.scalar('1_loss', met_disc_loss.result(), step=epoch)
    
    with test_summary_writer.as_default():
        tf.summary.scalar('1_loss', met_test_loss.result(), step=epoch)

    # Logging
    #print ('Time for epoch {} is {} sec.'.format(epoch + 1, time.time()-start))
    time.time()
      
    #Log stats
    template = 'Epoch {}, Gen_loss: {}, Disc_loss: {}, Test_loss: {}'
    print (template.format(epoch+1,
                           met_gen_loss.result(), 
                           met_disc_loss.result(),
                           met_test_loss.result()))
    
    # Reset metrics every epoch
    met_gen_loss.reset_states()
    met_disc_loss.reset_states()
    met_test_loss.reset_states()
    
# Generate a profile set
noise = gen_noise(EX_GEN_BATCH_SIZE)
generated_profile = generator(noise, training=False)
df_gen_prof = data_frame_from_gen(generated_profile, 'gencell_ep' + str(EPOCHS) + '_')
df_gen_prof.to_csv('gen/gen_prof_' + str(EPOCHS) + '.csv')