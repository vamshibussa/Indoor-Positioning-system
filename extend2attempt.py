
# coding: utf-8

# ### Autoencoders and Neural Network for Place recognition with WiFi fingerprints
# Implementation of algorithm discussed in <a href="https://arxiv.org/pdf/1611.02049v1.pdf">Low-effort place recognition with WiFi fingerprints using Deep Learning </a>

# In[13]:


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import scale
import copy

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

#labels = labels[0:1]
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

"""
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

# Generating a random test2.csv






"""
#######                  #######
####### Hierarchial Part #######
#######                  #######

# Training data in dataset
# Now we have to extract features and labels

#### Hierarchial Features ####
# Creating the feature set for second NN
additionalfeatures = np.asarray(dataset.iloc[:,522:528])
features2 = copy.copy(features)
df = pd.DataFrame(features2)
df1 = pd.DataFrame(additionalfeatures)
finaldf = pd.concat([df,df1], axis=1)
finalfeatures = np.asarray(finaldf.iloc[:,:])

# Now the labels
secondlabels = np.asarray(dataset.iloc[:,520:522])

# Prints for each row True or False to include in the training
train_val_split2 = np.random.rand(len(finalfeatures)) < 0.70

# Generating Training Features and Labels
train_x2 = finalfeatures[train_val_split2]
train_y2 = secondlabels[train_val_split2]

# Generating Validation Features and Labels
val_x2 = finalfeatures[~train_val_split2]
val_y2 = secondlabels[~train_val_split2]

print("HERE")

# Assume testing data present in test2.csv imported to test2_dataset
# Only the correctly predicted test rows here
#train_val_split2 = np.random.rand(len(finalfeatures)) < 0.70
test_dataset2 = pd.read_csv("test2.csv",header = 0)
testfeatures2 = np.asarray(test_dataset2.iloc[:,0:520])
# Creating the test feature set for second NN
additionalfeaturestest = np.asarray(test_dataset2.iloc[:,522:528])
testdf = pd.DataFrame(testfeatures2)
testdf1 = pd.DataFrame(additionalfeaturestest)
testfinaldf = pd.concat([testdf,testdf1], axis=1)
testfinalfeatures = np.asarray(testfinaldf.iloc[:,:])

# Now test labels

testlabels2 = np.asarray(test_dataset2.iloc[:,520:522])
print("HERE")

# Repeating the arch with minor changes

n2_input = 526
n2_hidden_1 = 256 
n2_hidden_2 = 128 
n2_hidden_3 = 64 

# Number of Output Classes
no2_outputs = secondlabels.shape[1] # Actually 2

learning_rate2 = 0.001
training_epochs2 = 15
batch_size2 = 10

# Number of training examples(rows)
total_batches2 = train_x2.shape[0] # batch_size


# In[55]:
print("HERE")

# X2 is input, random hence shape is none(#Rows),520features
# Y2 is output, depends on number of X fed
X2 = tf.placeholder(tf.float32, shape=[None,n2_input])
Y2 = tf.placeholder(tf.float32,[None,no2_outputs])

# Neural Networks Variables
print("HERE")
# Weight_variable and bias_variable are initialization fns
# First one is through truncated normal
# Second is zero constant

# --------------------- Encoder Variables --------------- #

e2_weights_h1 = weight_variable([n2_input, n2_hidden_1])
e2_biases_h1 = bias_variable([n2_hidden_1])

e2_weights_h2 = weight_variable([n2_hidden_1, n2_hidden_2])
e2_biases_h2 = bias_variable([n2_hidden_2])

e2_weights_h3 = weight_variable([n2_hidden_2, n2_hidden_3])
e2_biases_h3 = bias_variable([n2_hidden_3])

# --------------------- Decoder Variables --------------- #

d2_weights_h1 = weight_variable([n2_hidden_3, n2_hidden_2])
d2_biases_h1 = bias_variable([n2_hidden_2])

d2_weights_h2 = weight_variable([n2_hidden_2, n2_hidden_1])
d2_biases_h2 = bias_variable([n2_hidden_1])

d2_weights_h3 = weight_variable([n2_hidden_1, n2_input])
d2_biases_h3 = bias_variable([n2_input])

# --------------------- DNN Variables ------------------ #

dnn2_weights_h1 = weight_variable([n2_hidden_3, n2_hidden_2])
dnn2_biases_h1 = bias_variable([n2_hidden_2])

dnn2_weights_h2 = weight_variable([n2_hidden_2, n2_hidden_2])
dnn2_biases_h2 = bias_variable([n2_hidden_2])

dnn2_weights_out = weight_variable([n2_hidden_2, no2_outputs])
dnn2_biases_out = bias_variable([no2_outputs])


print("HERE")

# Encoder, Decoder and DNN as in paper page-4
def encode2(x):
    l1 = tf.nn.tanh(tf.add(tf.matmul(x,e2_weights_h1),e2_biases_h1))
    l2 = tf.nn.tanh(tf.add(tf.matmul(l1,e2_weights_h2),e2_biases_h2))
    l3 = tf.nn.tanh(tf.add(tf.matmul(l2,e2_weights_h3),e2_biases_h3))
    return l3
    
def decode2(x):
    l1 = tf.nn.tanh(tf.add(tf.matmul(x,d2_weights_h1),d2_biases_h1))
    l2 = tf.nn.tanh(tf.add(tf.matmul(l1,d2_weights_h2),d2_biases_h2))
    l3 = tf.nn.tanh(tf.add(tf.matmul(l2,d2_weights_h3),d2_biases_h3))
    return l3

def dnn2(x): # Note that the output is directly tanh there is no softmax
    l1 = tf.nn.tanh(tf.add(tf.matmul(x,dnn2_weights_h1),dnn2_biases_h1))
    l2 = tf.nn.tanh(tf.add(tf.matmul(l1,dnn2_weights_h2),dnn2_biases_h2))
    out = tf.add(tf.matmul(l2,dnn2_weights_out),dnn2_biases_out)
    out = tf.nn.tanh(out)   
    #out = tf.nn.softmax(tf.add(tf.matmul(l2,dnn_weights_out),dnn2_biases_out))
    return out

print("HERE")
# Nodes with operations created
encoded2 = encode2(X2)
print("HERE")
decoded2 = decode2(encoded2)
print("HERE") 

y_2 = dnn2(encoded2)
print("HERE") # Problem above

# Two types of cost functions
# First one with decoder and encoder
# Next one with actual output
us_cost_function2 = tf.reduce_mean(tf.pow(X2 - decoded2, 2))
s_cost_function2 =  tf.reduce_mean(tf.pow(Y2 - y_2, 2))  #-tf.reduce_sum(Y2 * tf.log(y_2))
us_optimizer2 = tf.train.AdamOptimizer(learning_rate2).minimize(us_cost_function2)
s_optimizer2 = tf.train.AdamOptimizer(learning_rate2).minimize(s_cost_function2)


# In[59]:
# Remember: the operations are on tensors
#correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
# Equal returns boolean, now accuracy measured after cast
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# No motion of accuracy so we use error(MSE)

# #### Model architecture
# Image take from: https://arxiv.org/pdf/1611.02049v1.pdf

# <img src="AE.png">
# <img src="NN.png">

# In[60]:


with tf.Session() as session:
    tf.global_variables_initializer().run()
    
    # ------------ 1. Training Autoencoders - Unsupervised Learning ----------- #
    for epoch in range(training_epochs2):
        epoch_costs = np.empty(0)
        for b in range(total_batches2):
            offset = (b * batch_size2) % (train_x2.shape[0] - batch_size2)
            batch_x = train_x2[offset:(offset + batch_size2), :]
            _, c = session.run([us_optimizer2, us_cost_function2],feed_dict={X2: batch_x})
            epoch_costs = np.append(epoch_costs,c)
        print("Epoch: ",epoch," Loss: ",np.mean(epoch_costs))
    print("Unsupervised pre-training finished...")
    
    
    # ---------------- 2. Training NN - Supervised Learning ------------------ #
    for epoch in range(training_epochs2):
        epoch_costs = np.empty(0)
        for b in range(total_batches2):
            offset = (b * batch_size2) % (train_x2.shape[0] - batch_size2)
            batch_x2 = train_x2[offset:(offset + batch_size2), :]
            batch_y2 = train_y2[offset:(offset + batch_size2), :]
            _, c = session.run([s_optimizer2, s_cost_function2],feed_dict={X2: batch_x2, Y2 : batch_y2})
            epoch_costs = np.append(epoch_costs,c)
        print("Epoch: ",epoch," Cost: ",np.mean(epoch_costs),"Validation Cost:", session.run(s_cost_function2, feed_dict={X2: val_x2, Y2: val_y2}))
            
    print("Supervised training finished...")
    
    # Instead of setting number of iterations we can also find where validation error starts increasing for generalization
    print("\nTesting Cost Function:", session.run(s_cost_function2, feed_dict={X2: testfinalfeatures , Y2: testlabels2}))




# Assume correct testing set is in


