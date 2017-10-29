# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import tflearn

# Load CSV file, indicate that the first column represents labels
from tflearn.data_utils import load_csv
#train_feature, train_label = load_csv('train_set', has_header=False)
train_feature, train_label = load_csv('dm_c2b_v3_cate_deal_train_set', has_header=True)
test_feature, test_label = load_csv('dm_c2b_v3_cate_deal_test_set', has_header=True)

train_feature=np.reshape(train_feature, [-1, 36])
train_label=np.reshape(train_label, [-1, 1])


# Build neural network
net = tflearn.input_data(shape=[None, 36])
#net = tflearn.fully_connected(net, 40, activation='linear')
net = tflearn.fully_connected(net, 20, activation='linear')
net = tflearn.dropout(net, 0.8)
net = tflearn.fully_connected(net, 10, activation='linear')
net = tflearn.dropout(net, 0.8)
net = tflearn.fully_connected(net, 1, activation='linear')
regression = tflearn.regression(net, optimizer='sgd', loss='mean_square',
                                        metric='R2', learning_rate=0.01)
# Define model
model = tflearn.DNN(regression)

# Start training (apply gradient descent algorithm)
model.fit(train_feature, train_label, n_epoch=80, batch_size=10000, show_metric=True)

pred = model.predict(train_feature)
print("train set R2: ", model.evaluate(train_feature, train_label))
mape=0
print(pred.shape)
#print(pred.size)
#pred=np.reshape(pred, [-1, 1])
#train_label=np.reshape(train_label, [-1, 1])
for i in range(pred.size):
    mape+=np.abs(pred[i].astype(np.float) - train_label[i].astype(np.float))/train_label[i].astype(np.float)
print("train set : ", pred.size, " : ", mape, " : ", mape/len(pred))



test_feature=np.reshape(test_feature, [-1, 36])
test_label=np.reshape(test_label, [-1, 1])
pred = model.predict(test_feature)
print("test set R2: ", model.evaluate(test_feature, test_label))
mape=0
print(pred.shape)
for i in range(pred.size):
    mape+=np.abs(pred[i].astype(np.float) - test_label[i].astype(np.float))/test_label[i].astype(np.float)
print("test set : ", pred.size, " : ", mape, " : ", mape/len(pred))
