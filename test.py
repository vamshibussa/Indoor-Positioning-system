
# coding: utf-8

# ### Autoencoders and Neural Network for Place recognition with WiFi fingerprints
# Implementation of algorithm discussed in <a href="https://arxiv.org/pdf/1611.02049v1.pdf">Low-effort place recognition with WiFi fingerprints using Deep Learning </a>

# In[13]:

import random
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import scale


# In[14]:

inputsize = 7

dataset = pd.read_csv("trainingData.csv",header = 0)

features = np.asarray(dataset.iloc[:,0:520])
#features = np.array(dataset.iloc[:,0:520])
# Making the zero signal equal to -110
features[features == 100] = -110

# Now we can perform the input reduction

#print(features[0])

# Extracting indices where non zero signal
#tmp = []
#tmp.append(np.where(features[0]>-110))

#################### Preprocessing Data begins #######################

print('\n')
print(""Preprocessing Data Begins"")

for i in range(len(features)):
    tmp = list(np.asarray(np.where(features[i]>-110))[0])
#for i in len(features): # Over all rows

    #print(tmp)
    np.random.shuffle(tmp)
    #print(tmp)
    j = 0

    length = len(tmp)
    while(j<inputsize and length>inputsize):
        length = len(tmp) 
        rand = random.randint(0,length-1)
        del tmp[rand]
        j+=1

    #print(tmp)
    #print(len(features[features== -110]))

    for x in tmp:
        features[i][x] = -110

    #print(features[i])

    #print()
    #print(len(features[features== -110]))



##################### Preprocessing Data over ##########################

# Normalizing the inputs
features = (features - features.mean()) / features.var()




labels = np.asarray(dataset["BUILDINGID"].map(str) + dataset["FLOOR"].map(str))
labels = np.asarray(pd.get_dummies(labels))


# #### Dividing UJIndoorLoc training data set into training and validation set

# In[15]:


train_val_split = np.random.rand(len(features)) < 0.70
train_x = features[train_val_split]
train_y = labels[train_val_split]
val_x = features[~train_val_split]
val_y = labels[~train_val_split]


# #### Using UJIndoorLoc validation data set as testing set

# In[16]:


test_dataset = pd.read_csv("validationData.csv",header = 0)

test_features = np.asarray(test_dataset.iloc[:,0:520])
test_features[test_features == 100] = -110
test_features = (test_features - test_features.mean()) / test_features.var()

test_labels = np.asarray(test_dataset["BUILDINGID"].map(str) + test_dataset["FLOOR"].map(str))
test_labels = np.asarray(pd.get_dummies(test_labels))


# In[17]:


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

n_classes = labels.shape[1]

learning_rate = 0.00001
training_epochs = 30
batch_size = 15

total_batches = train_x.shape[0] // batch_size


# In[55]:


X = tf.placeholder(tf.float32, shape=[None,n_input])
Y = tf.placeholder(tf.float32,[None,n_classes])

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


encoded = encode(X)
decoded = decode(encoded) 
y_ = dnn(encoded)


# In[58]:


us_cost_function = tf.reduce_mean(tf.pow(X - decoded, 2))
s_cost_function = -tf.reduce_sum(Y * tf.log(y_))
us_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(us_cost_function)
s_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(s_cost_function)


# In[59]:


correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
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
    

    print("\nTesting Accuracy:", session.run(accuracy, feed_dict={X: test_features, Y: test_labels}))


# --------------------------------------------------------------------------------------------------------------------------

