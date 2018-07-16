import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from numpy import genfromtxt
from sklearn.preprocessing import MinMaxScaler


n_train_steps = 10000

batch_size = 32

noise_input_size = 100
inflate_to_size = 600
gex_size = 6605

disc_internal_size = 200
num_cells_train = 1500

num_cells_generate = 500

learn_rate = 1e-5

initial_run = True

model_name = "test_run" + "_" + str(np.random.randint(100000))
model_to_use = "../models/" + model_name

tensorboard_summary_path = "../summaries/" + model_name


input_ltpm_matrix = genfromtxt('../data/four_datasets_combined_lTPM_red_small_clean.csv', delimiter=',', skip_header=1)

scaler = MinMaxScaler()
input_ltpm_matrix = np.transpose(input_ltpm_matrix)
scaler.fit(input_ltpm_matrix)

input_ltpm_matrix = scaler.transform(input_ltpm_matrix)
input_ltpm_matrix = np.transpose(input_ltpm_matrix)

print(np.ptp(input_ltpm_matrix, axis=1))
print(np.ptp(input_ltpm_matrix, axis=1).shape[0])

# In[2]:

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[3]:


input_ltpm_matrix_unwitheld = input_ltpm_matrix
input_ltpm_matrix = input_ltpm_matrix[:,0:num_cells_train]

# ### Generator network

# Latent space variable placeholder
z = tf.placeholder(tf.float32, shape=(None, noise_input_size))

def generator(z, reuse=False):

    with tf.variable_scope("generator", reuse=reuse):

        gen_dense1 = tf.layers.dense(inputs=z, 
            units=inflate_to_size, 
            activation=None, 
            name="gen_dense1")
        gen_dense1 = tf.nn.leaky_relu(gen_dense1)

        gen_dense2 = tf.layers.dense(inputs=gen_dense1, 
            units=inflate_to_size, 
            activation=None, 
            name="gen_dense2")
        gen_dense2 = tf.nn.leaky_relu(gen_dense2)

        gen_output = tf.layers.dense(inputs=gen_dense2, 
            units=gex_size, 
            activation=None, 
            name="gen_output")
        gen_output = tf.nn.leaky_relu(gen_output)

        return gen_output


# ### Latent vector / noise input
# Poisson noise (count data) + Gaussian (other artefacts)

data_max_value = np.amax(input_ltpm_matrix)

def noise_prior(batch_size, dim):
    temp_norm = np.random.normal(0.0, data_max_value/10, size=(batch_size, dim))
    temp_poisson = np.random.poisson(1, size=(batch_size, dim))
    return np.abs(temp_norm + temp_poisson)

example_noise_batch = noise_prior(100, 100)

# ### Discriminator network

# Gene expression placeholder
x = tf.placeholder(tf.float32, shape=(None, gex_size))

def discriminator(x, reuse=False):
    
    with tf.variable_scope("discriminator", reuse=reuse):

        disc_dense1 = tf.layers.dense(inputs=x, 
            units=disc_internal_size, 
            activation=None, 
            name="disc_dense1")
        disc_dense1 = tf.nn.leaky_relu(disc_dense1)

        disc_dense2 = tf.layers.dense(inputs=disc_dense1, 
            units=disc_internal_size, 
            activation=None, 
            name="disc_dense2")
        disc_dense2 = tf.nn.leaky_relu(disc_dense2)

        disc_output = tf.layers.dense(inputs=disc_dense2, 
            units=1, 
            activation=None, 
            name="disc_output")

        return disc_output


# Define the outputs we're going to train.
G = generator(z)
D_real = discriminator(x)
D_fake = discriminator(G, reuse=True)


# In[9]:

time_step = tf.placeholder(tf.int32)

obj_d = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)
obj_g = -tf.reduce_mean(D_fake)

T_vars = tf.trainable_variables()
d_params = [var for var in T_vars if var.name.startswith("discriminator")]
g_params = [var for var in T_vars if var.name.startswith("generator")]

### Summaries of D/G loss and all layer weights
tf.summary.scalar('loss/gen', obj_g)
tf.summary.scalar('loss/disc', obj_d)

summary_all_weights = [tf.summary.histogram(values=v, name=v.name) for v in T_vars if 'kernel' in v.name ]

tf.summary.histogram('output/gen', G)

### Thanks to taki0112 for the TF StableGAN implementation https://github.com/taki0112/StableGAN-Tensorflow
from Adam_prediction import Adam_Prediction_Optimizer
opt_g = Adam_Prediction_Optimizer(learning_rate=learn_rate, beta1=0.9, beta2=0.999, prediction=True).minimize(obj_d, var_list=d_params)
opt_d = Adam_Prediction_Optimizer(learning_rate=learn_rate, beta1=0.9, beta2=0.999, prediction=False).minimize(obj_g, var_list=g_params)


# ### Training

# In[10]:


import time
start_time = time.time()

global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)

sess=tf.InteractiveSession()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(tensorboard_summary_path, sess.graph)
saver = tf.train.Saver()
init = tf.global_variables_initializer().run()

if initial_run:
    print("Initial run")
    assign_step_zero = tf.assign(global_step, 0)
    init_step = sess.run(assign_step_zero)
    
if not initial_run:
    saver.restore(sess, model_to_use)

for i in range(n_train_steps):
    increment_global_step_op = tf.assign(global_step, global_step+1)
    step = sess.run(increment_global_step_op)
    
    current_step = sess.run(global_step)
    
    idx = np.random.randint(input_ltpm_matrix.shape[1], size=batch_size)

    x_data = input_ltpm_matrix[:,idx]
    x_data = np.transpose(x_data)
    
    noise = noise_prior(batch_size, 100)
    
    #train discriminator 2x more
    _, summary = sess.run([opt_d, merged], {x : x_data, z : noise, time_step : current_step})
    train_writer.add_summary(summary, current_step)
        
    sess.run([opt_g], {z : noise, x : x_data, time_step : current_step})
    
    if i % (n_train_steps/10) == 0:
        print(str(float(i)/n_train_steps) + " No. steps: " + str(current_step))
        intermediate_save_path = saver.save(sess, model_to_use, global_step=current_step)

save_path = saver.save(sess, model_to_use)
print("Model saved in file: %s" % save_path)

print("--- %s seconds ---" % (time.time() - start_time))


# ### Generate cells after training

# In[11]:


#Generate 100 cells
gen_cells = sess.run(G, {z : noise_prior(1, 100)})

for cell in range(num_cells_generate):
    out_gen_temp = sess.run(G, {z : noise_prior(1, 100)})
    gen_cells = numpy.append(gen_cells, out_gen_temp, axis=0)

