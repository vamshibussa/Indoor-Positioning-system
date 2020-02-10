
# coding: utf-8

# ### Autoencoders and Neural Network for Place recognition with WiFi fingerprints
# Implementation of algorithm discussed in <a href="https://arxiv.org/pdf/1611.02049v1.pdf">Low-effort place recognition with WiFi fingerprints using Deep Learning </a>

# In[13]:


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import scale
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
import copy

# In[14]:

inputsize = 3
# Read Data into Pandas DataFrame First line is headings
dataset = pd.read_csv("trainingData.csv",header=0)

# First Rows then columns
features = np.asarray(dataset.iloc[:,0:520])

# Make all the zero signals(100) to -110
features[features == 100] = -110


#################### Preprocessing Data begins #######################

print('\n')
print("Preprocessing Data Begins")

# Extracting indices where non zero signal
#tmp = []
#tmp.append(np.where(features[0]>-110))
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


print('\n')
print("Preprocessing Data Ends")



##################### Preprocessing Data over ##########################

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
    acc, corrpred = session.run([accuracy,correct_prediction], feed_dict={X: test_features, Y: test_labels})
    print("\nTesting Accuracy:", acc )
    # To use correct_prediction to get test2.csv
    allrows = np.asarray(test_dataset.iloc[:,:])
    test2rows= allrows[corrpred]
    print(test2rows.shape[0])
    print(acc*allrows.shape[0])
    #testing = test2rows[0:2,0:3]
    #print(testing)
    # Now writing to csv file  test3
    np.savetxt("test3.csv",test2rows, delimiter=",")
    #test2rows.to_csv(test3.csv, sep='\t', encoding='utf-8')



# --------------------------------------------------------------------------------------------------------------------------

# Generating a random test2.csv

#######                  #######
####### Hierarchial Part #######
#######                  #######

# Training data in dataset
# Now we have to extract features and labels

#### HIERARCHIAL PART  ####

## Using KNN

## Appropriate Feautures and Labels for KNN
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

#print("HERE")

# Assume testing data present in test2.csv imported to test2_dataset
# Only the correctly predicted test rows here
#train_val_split2 = np.random.rand(len(finalfeatures)) < 0.70
test_dataset2 = pd.read_csv("test3.csv",header = 0)
testfeatures2 = np.asarray(test_dataset2.iloc[:,0:520])
# Creating the test feature set for second NN
additionalfeaturestest = np.asarray(test_dataset2.iloc[:,522:528])
testdf = pd.DataFrame(testfeatures2)
testdf1 = pd.DataFrame(additionalfeaturestest)
testfinaldf = pd.concat([testdf,testdf1], axis=1)
testfinalfeatures = np.asarray(testfinaldf.iloc[:,:])

# Now test labels

testlabels2 = np.asarray(test_dataset2.iloc[:,520:522])

print(train_y2[0])
print()
print()


#us_cost_function = tf.reduce_mean(tf.pow(X - decoded, 2))
for K in range(25):
    K_value = K+1
    neigh = KNeighborsRegressor(n_neighbors = K_value, weights='uniform', algorithm='auto')
    neigh.fit(train_x2,train_y2) 
    #print("One")
    y_pred = neigh.predict(val_x2)
    #us_cost_functionKNN = tf.reduce_mean(tf.pow(y_pred - val_y2, 2))
    #c = tf.Session.run(us_cost_functionKNN,feed_dict={val_y2: val_y2})
    print()
    dist = np.linalg.norm(y_pred-val_y2)
    print ("CostFunction for validationData is ", (dist),"for K-Value:",K_value)
    #print("Two")
    #print(y_pred[0])
    #print(len(y_pred[0]))
    #print ("Accuracy is ", accuracy_score(val_y2,y_pred)*100,"% for K-Value:",K_value)
	 # Assume correct testing set is in
    y_pred2 = neigh.predict(testfinalfeatures)
    dist2 = np.linalg.norm(y_pred2-testlabels2)
    print ("CostFunction for Testing is ", (dist2),"for K-Value:",K_value)



