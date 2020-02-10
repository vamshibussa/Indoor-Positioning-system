
# coding: utf-8

# ### Autoencoders and Neural Network for Place recognition with WiFi fingerprints
# Implementation of algorithm discussed in <a href="https://arxiv.org/pdf/1611.02049v1.pdf">Low-effort place recognition with WiFi fingerprints using Deep Learning </a>

# In[13]:


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import scale


# In[14]:

# Read Data into Pandas DataFrame First line is labels
dataset = pd.read_csv("trainingData.csv",header=0)

# First Rows then columns
features = np.asarray(dataset.iloc[:,0:520])
# Make all the zero signals(100) to -110
features[features == 100] = -110

# Normalizing the data 
features = (features - features.mean()) / features.var()

# Extracting the two columns as a concatenated row string
labels = np.asarray(dataset["BUILDINGID"].map(str) + dataset["FLOOR"].map(str))
# Now one hot encoding the categorical rep to list
labels = np.asarray(pd.get_dummies(labels))


# #### Dividing UJIndoorLoc training data set into training and validation set

# In[15]:

# Prints for each row True or False to include in the training
train_val_split = np.random.rand(len(features)) < 0.70

# Generating Training Features and Labels
train_x = features[train_val_split]
train_y = labels[train_val_split]

# Generating Validation Features and Labels
val_x = features[~train_val_split]
val_y = labels[~train_val_split]



# #### Using UJIndoorLoc validation data set as testing set

# In[16]:

# Repeat the above process for testing set
test_dataset = pd.read_csv("validationData.csv",header = 0)

test_features = np.asarray(test_dataset.iloc[:,0:520])
test_features[test_features == 100] = -110
test_features = (test_features - test_features.mean()) / test_features.var()

test_labels = np.asarray(test_dataset["BUILDINGID"].map(str) + test_dataset["FLOOR"].map(str))
test_labels = np.asarray(pd.get_dummies(test_labels))


# In[17]:

# Core Algorithm

# Functions to initialize the Weights between layers and their biases
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape = shape)
    return tf.Variable(initial)


# In[54]:


n_input = 520 
n_hidden_1 = 256 
n_hidden_2 = 128 
n_hidden_3 = 64 

# Number of Output Classes
n_classes = labels.shape[1]

learning_rate = 0.00001
training_epochs = 30
batch_size = 15

# Number of training examples(rows)
total_batches = train_x.shape[0] // batch_size


# In[55]:

# X is input, random hence shape is none(#Rows),520features
# Y is output, depends on number of X fed
X = tf.placeholder(tf.float32, shape=[None,n_input])
Y = tf.placeholder(tf.float32,[None,n_classes])

# Neural Networks Variables

# Weight_variable and bias_variable are initialization fns
# First one is through truncated normal
# Second is zero constant

# --------------------- Encoder Variables --------------- #

e_weights_h1 = weight_variable([n_input, n_hidden_1])
e_biases_h1 = bias_variable([n_hidden_1])

e_weights_h2 = weight_variable([n_hidden_1, n_hidden_2])
e_biases_h2 = bias_variable([n_hidden_2])

e_weights_h3 = weight_variable([n_hidden_2, n_hidden_3])
e_biases_h3 = bias_variable([n_hidden_3])

# --------------------- Decoder Variables --------------- #

d_weights_h1 = weight_variable([n_hidden_3, n_hidden_2])
d_biases_h1 = bias_variable([n_hidden_2])

d_weights_h2 = weight_variable([n_hidden_2, n_hidden_1])
d_biases_h2 = bias_variable([n_hidden_1])

d_weights_h3 = weight_variable([n_hidden_1, n_input])
d_biases_h3 = bias_variable([n_input])

# --------------------- DNN Variables ------------------ #

dnn_weights_h1 = weight_variable([n_hidden_3, n_hidden_2])
dnn_biases_h1 = bias_variable([n_hidden_2])

dnn_weights_h2 = weight_variable([n_hidden_2, n_hidden_2])
dnn_biases_h2 = bias_variable([n_hidden_2])

dnn_weights_out = weight_variable([n_hidden_2, n_classes])
dnn_biases_out = bias_variable([n_classes])


# In[56]:

# Encoder, Decoder and DNN as in paper page-4
def encode(x):
    l1 = tf.nn.tanh(tf.add(tf.matmul(x,e_weights_h1),e_biases_h1))
    l2 = tf.nn.tanh(tf.add(tf.matmul(l1,e_weights_h2),e_biases_h2))
    l3 = tf.nn.tanh(tf.add(tf.matmul(l2,e_weights_h3),e_biases_h3))
    return l3
    
def decode(x):
    l1 = tf.nn.tanh(tf.add(tf.matmul(x,d_weights_h1),d_biases_h1))
    l2 = tf.nn.tanh(tf.add(tf.matmul(l1,d_weights_h2),d_biases_h2))
    l3 = tf.nn.tanh(tf.add(tf.matmul(l2,d_weights_h3),d_biases_h3))
    return l3

def dnn(x):
    l1 = tf.nn.tanh(tf.add(tf.matmul(x,dnn_weights_h1),dnn_biases_h1))
    l2 = tf.nn.tanh(tf.add(tf.matmul(l1,dnn_weights_h2),dnn_biases_h2))
    out = tf.nn.softmax(tf.add(tf.matmul(l2,dnn_weights_out),dnn_biases_out))
    return out


# In[57]:

# Nodes with operations created
encoded = encode(X)
decoded = decode(encoded) 
y_ = dnn(encoded)


# In[58]:

# Two types of cost functions
# First one with decoder and encoder
# Next one with actual output
us_cost_function = tf.reduce_mean(tf.pow(X - decoded, 2))
s_cost_function = -tf.reduce_sum(Y * tf.log(y_))
us_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(us_cost_function)
s_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(s_cost_function)


# In[59]:
# Remember: the operations are on tensors
# Argmax is used to get max prob element from softmax output along --> axis
correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
# Equal returns boolean, now accuracy measured after cast
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# #### Model architecture
# Image take from: https://arxiv.org/pdf/1611.02049v1.pdf

# <img src="AE.png">
# <img src="NN.png">

# In[60]:


with tf.Session() as session:
    tf.global_variables_initializer().run()
    
    # ------------ 1. Training Autoencoders - Unsupervised Learning ----------- #
    for epoch in range(training_epochs):
        epoch_costs = np.empty(0)
        for b in range(total_batches):
            offset = (b * batch_size) % (train_x.shape[0] - batch_size)
            batch_x = train_x[offset:(offset + batch_size), :]
            _, c = session.run([us_optimizer, us_cost_function],feed_dict={X: batch_x})
            epoch_costs = np.append(epoch_costs,c)
        print("Epoch: ",epoch," Loss: ",np.mean(epoch_costs))
    print("Unsupervised pre-training finished...")
    
    
    # ---------------- 2. Training NN - Supervised Learning ------------------ #
    for epoch in range(training_epochs):
        epoch_costs = np.empty(0)
        for b in range(total_batches):
            offset = (b * batch_size) % (train_x.shape[0] - batch_size)
            batch_x = train_x[offset:(offset + batch_size), :]
            batch_y = train_y[offset:(offset + batch_size), :]
            _, c = session.run([s_optimizer, s_cost_function],feed_dict={X: batch_x, Y : batch_y})
            epoch_costs = np.append(epoch_costs,c)
        print("Epoch: ",epoch," Loss: ",np.mean(epoch_costs)," Training Accuracy: ",             session.run(accuracy, feed_dict={X: train_x, Y: train_y}),             "Validation Accuracy:", session.run(accuracy, feed_dict={X: val_x, Y: val_y}))
            
    print("Supervised training finished...")
    
    # Instead of setting number of iterations we can also find where validation error starts increasing for generalization
    print("\nTesting Accuracy:", session.run(accuracy, feed_dict={X: test_features, Y: test_labels}))


# --------------------------------------------------------------------------------------------------------------------------
"""