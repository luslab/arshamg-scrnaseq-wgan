#!/usr/bin/env python
# coding: utf-8


# Load libs
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

# log the number of resources available in tensor flow
print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Log start
print("Initializing...")

########################################################################################
# INIT & LOAD
########################################################################################

# Setup paths
#train_feature_path = 'tpm_combined.csv'
#train_gene_name_path = 'tpm_combined_rows.csv'
#train_cell_name_path = 'tpm_combined_cols.csv'
#test_feature_path = 'tpm_combined_test.csv'
#test_gene_name_path = 'tpm_combined_rows_test.csv'
#test_cell_name_path = 'tpm_combined_cols_test.csv'
#train_nonorm_path = 'tpm_combined_train_nonorm.csv'

# Setup paths
train_feature_path = sys.argv[1]
train_gene_name_path = sys.argv[4]
train_cell_name_path = sys.argv[2]
test_feature_path = sys.argv[6]
test_gene_name_path = sys.argv[5]
test_cell_name_path = sys.argv[3]
train_nonorm_path = sys.argv[8]

# Load training data in using pandas
df_gene_names = pd.read_csv(train_gene_name_path, header=None)
df_cell_names = pd.read_csv(train_cell_name_path, header=None)
df_training_data = pd.read_csv(train_feature_path, header=None)
df_training_data_nonorm = pd.read_csv(train_nonorm_path)

# Load test data in using pandas
df_gene_names_test = pd.read_csv(test_gene_name_path, header=None)
df_cell_names_test = pd.read_csv(test_cell_name_path, header=None)
df_test_data = pd.read_csv(test_feature_path, header=None)

# Do some column corrections
df_gene_names.columns = ['gene_name']
df_training_data_nonorm = df_training_data_nonorm.drop('gene_name', axis=1)

# Find the min/max of the unnormalised data set that the generated examples can be scaled to
nonorm_max = df_training_data_nonorm.max().max()
nonorm_min = df_training_data_nonorm.min().min()
del df_training_data_nonorm

# The number of genes in the input dataset determines the 
# generator output as well as the discriminator input sizes
num_genes = df_gene_names.shape[0]

########################################################################################
# MODEL PARAMS
########################################################################################

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
TRAIN_BATCH_SIZE = 32
GEN_BATCH_SIZE = 32
TRAIN_BUFFER_SIZE = 10000
TEST_BATCH_SIZE = 500
TEST_BUFFER_SIZE = 500
EPOCHS = 10

#LEARNING_RATE = 0.001
#LEARNING_RATE = 1e-5
LEARNING_RATE = 5e-5
GP_LAMBDA = 10

EX_GEN_BATCH_SIZE = 500
WRITE_FREQ = 100

# Get ext param
EPOCHS = int(sys.argv[9])
WRITE_FREQ = int(sys.argv[10])

########################################################################################
# TENSORFLOW INIT
########################################################################################

# Create tensors from training data - Convert to Int32 for better work on GPU with batch and shuffle
train_dataset = tf.data.Dataset.from_tensor_slices(df_training_data.T.values.astype('float32')).shuffle(TRAIN_BUFFER_SIZE).batch(TRAIN_BATCH_SIZE, drop_remainder=True)

# Create tensors from test data - Convert to Int32 for better work on GPU with batch and shuffle
#test_dataset = tf.data.Dataset.from_tensor_slices(df_test_data.T.values.astype('float32')).shuffle(TEST_BUFFER_SIZE).batch(TEST_BATCH_SIZE)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

########################################################################################
# DEFINE FUNCTIONS
########################################################################################

# Define function for constructing the generator
def create_generator():
    model = tf.keras.Sequential()
    
    #L1
    model.add(layers.Dense(GEN_L1_DENSE_SIZE, use_bias=False, input_shape=(LATENT_VARIABLE_SIZE,)))
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    #L2
    model.add(layers.Dense(GEN_L2_DENSE_SIZE, use_bias=False))
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    #L3
    model.add(layers.Dense(GEN_L3_DENSE_SIZE, use_bias=False))
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    return model

# Define function for constructing discriminator
def create_discriminator():
    model = tf.keras.Sequential()
    
    #L1
    model.add(layers.Dense(DIS_L1_DENSE_SIZE, use_bias=False, input_shape=(DIS_INPUT_SIZE,)))
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    #L2
    model.add(layers.Dense(DIS_L2_DENSE_SIZE, use_bias=False))
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
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
def discriminator_loss(real_output, fake_output):
    #real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    #fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    #total_loss = real_loss + fake_loss
    #return total_loss
    
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

def real_loss(real_output):
    return cross_entropy(tf.ones_like(real_output), real_output)

def fake_loss(fake_output):
    #return cross_entropy(tf.ones_like(fake_output), fake_output)
    return - tf.reduce_mean(fake_output)

# This generates real scaled data from the generator
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

@tf.function
def train_step(cell_profiles):
    with tf.GradientTape(persistent=True) as tape:

        # Generate noise
        noise = gen_noise(GEN_BATCH_SIZE)

        # Generate some fake profiles using noise
        generated_profiles = generator(noise, training=True)
        
        # Pass the real profiles and the fake profiles through the disc
        real_output = discriminator(cell_profiles, training=True)
        fake_output = discriminator(generated_profiles, training=True)

        # Calc losses (some used for training, others for visualizing on tensorboard)
        floss = fake_loss(fake_output)
        rloss = real_loss(real_output)
        dloss = discriminator_loss(real_output, fake_output)

        # Calculate interpolated profile 
        shape = [tf.shape(cell_profiles)[0]] + [1] * (cell_profiles.shape.ndims - 1)
        epsilon = tf.random.uniform(shape=shape, minval=0., maxval=1.)
        interpolated_profiles = cell_profiles + epsilon * (generated_profiles - cell_profiles)

        # Run through disc with nested gradient tape
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(interpolated_profiles)
            d_interpolated = discriminator(interpolated_profiles, training=True)

        # Compute gradient penalty
        grad = tape2.gradient(d_interpolated, interpolated_profiles)
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        gradient_penalty = tf.reduce_mean((norm - 1.)**2)

        # Calculate adjusted loss
        gploss = dloss + (GP_LAMBDA * gradient_penalty)

    # Save the gradients
    gradients_of_generator = tape.gradient(floss, generator.trainable_variables)
    gradients_of_discriminator = tape.gradient(gploss, discriminator.trainable_variables)

    # Apply the gradients
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # Clip the weights
    #clip_v = [v.assign((tf.clip_by_value(v, -0.01, 0.01))) for v in discriminator.trainable_variables]
                
    # Record gradients for tensorboard
    tf.summary.scalar("grad_penalty", gradient_penalty)
    tf.summary.scalar("grad_norm", tf.nn.l2_loss(gradient_penalty))

    # Record losses for tensorboard
    met_fake_loss(floss)
    met_real_loss(rloss)
    met_disc_loss(dloss)
    met_gp_loss(gploss)

    # Record accuracy for tensorboard
    met_fake_acc(tf.reduce_mean(fake_output))
    met_real_acc(tf.reduce_mean(real_output))
    
    return

########################################################################################
# INITIALISE NN MODEL
########################################################################################

# Create generator and discriminator
generator = create_generator()
discriminator = create_discriminator()

generator.summary()
discriminator.summary()

# Define optimizer
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
#generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
#discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)

# Create checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Generate a single test set and write to file
noise = gen_noise(EX_GEN_BATCH_SIZE)
generated_profile = generator(noise, training=False)
df_gen_prof_1 = data_frame_from_gen(generated_profile, 'gencell_ep0_')

# Define tensorboard metrics
met_fake_loss = tf.keras.metrics.Mean('fake_loss', dtype=tf.float32)
met_real_loss = tf.keras.metrics.Mean('real_loss', dtype=tf.float32)
met_disc_loss = tf.keras.metrics.Mean('disc_loss', dtype=tf.float32)
#met_test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
met_gp_loss = tf.keras.metrics.Mean('met_gp_loss', dtype=tf.float32)

met_fake_acc = tf.keras.metrics.Mean('fake_acc', dtype=tf.float32)
met_real_acc = tf.keras.metrics.Mean('real_acc', dtype=tf.float32) 
#met_fake_acc = tf.keras.metrics.BinaryAccuracy()('fake_acc', dtype=tf.float32)
#met_real_acc = tf.keras.metrics.Accuracy('real_acc', dtype=tf.float32) 

# Create log directories and tensor board summaries
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

fake_log_dir = 'logs/gradient_tape/' + current_time + '/fake_train'
real_log_dir = 'logs/gradient_tape/' + current_time + '/real_train'
disc_log_dir = 'logs/gradient_tape/' + current_time + '/disc_train'
#test_log_dir = 'logs/gradient_tape/' + current_time + '/disc_test'
gp_log_dir = 'logs/gradient_tape/' + current_time + '/gp_train'
all_log_dir = 'logs/gradient_tape/' + current_time + '/all'
realacc_log_dir = 'logs/gradient_tape/' + current_time + '/real_acc'
fakeacc_log_dir = 'logs/gradient_tape/' + current_time + '/fake_acc'

all_summary_writer = tf.summary.create_file_writer(all_log_dir)
fake_summary_writer = tf.summary.create_file_writer(fake_log_dir)
real_summary_writer = tf.summary.create_file_writer(real_log_dir)
disc_summary_writer = tf.summary.create_file_writer(disc_log_dir)
gp_summary_writer = tf.summary.create_file_writer(gp_log_dir)
#test_summary_writer = tf.summary.create_file_writer(test_log_dir)
realacc_summary_writer = tf.summary.create_file_writer(realacc_log_dir)
fakeacc_summary_writer = tf.summary.create_file_writer(fakeacc_log_dir)

########################################################################################
# MAIN LOOP
########################################################################################

# Run the training model
print('Running...')

# Loop
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
    
    # Train the epoch
    for data_batch in train_dataset:
        train_step(data_batch)
        
    # Run test data through discriminator
    #for data_batch in test_dataset:
    #    test_decision = discriminator(data_batch, training=False)

    # Assess test loss
    #test_loss = cross_entropy(tf.ones_like(test_decision), test_decision)
    #met_test_loss(test_loss)
    
    # Log metrics
    with all_summary_writer.as_default():
        tf.summary.scalar('3_real_loss', met_real_loss.result(), step=epoch)
        tf.summary.scalar('4_fake_loss', met_fake_loss.result(), step=epoch)
        tf.summary.scalar('5_disc_loss', met_disc_loss.result(), step=epoch)
        tf.summary.scalar('6_gp_loss', met_gp_loss.result(), step=epoch)
        tf.summary.scalar('7_real_acc', met_real_acc.result(), step=epoch)
        tf.summary.scalar('8_fake_acc', met_fake_acc.result(), step=epoch)
    
    with fake_summary_writer.as_default():
        tf.summary.scalar('1_loss', met_fake_loss.result(), step=epoch)

    with real_summary_writer.as_default():
        tf.summary.scalar('1_loss', met_real_loss.result(), step=epoch)
           
    with disc_summary_writer.as_default():
        tf.summary.scalar('1_loss', met_disc_loss.result(), step=epoch)
    
    with gp_summary_writer.as_default():
        tf.summary.scalar('1_loss', met_gp_loss.result(), step=epoch)
    
    #with test_summary_writer.as_default():
    #    tf.summary.scalar('1_loss', met_test_loss.result(), step=epoch)

    with realacc_summary_writer.as_default():
        tf.summary.scalar('2_acc', met_real_acc.result(), step=epoch)
    
    with fakeacc_summary_writer.as_default():
        tf.summary.scalar('2_acc', met_fake_acc.result(), step=epoch)

    # Logging
    #print ('Time for epoch {} is {} sec.'.format(epoch + 1, time.time()-start))
    time.time()
      
    # Log stats
    template = 'Epoch {}, Fake_loss: {}, Real_loss: {}, Disc_loss: {}'
    print (template.format(epoch+1,
                           met_fake_loss.result(),
                           met_real_loss.result(),
                           met_disc_loss.result()))
    
    # Reset metrics every epoch
    met_fake_loss.reset_states()
    met_real_loss.reset_states()
    met_disc_loss.reset_states()
    #met_test_loss.reset_states()
    met_gp_loss.reset_states()
    met_real_acc.reset_states()
    met_fake_acc.reset_states()
    
# Generate a profile set at the end after the loop finishes
noise = gen_noise(EX_GEN_BATCH_SIZE)
generated_profile = generator(noise, training=False)
df_gen_prof = data_frame_from_gen(generated_profile, 'gencell_ep' + str(EPOCHS) + '_')
df_gen_prof.to_csv('gen/gen_prof_' + str(EPOCHS) + '.csv')