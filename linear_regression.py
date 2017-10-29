""" Linear Regression Example """

from __future__ import absolute_import, division, print_function

import numpy as np
import tflearn

# Regression data

# Load CSV file, indicate that the first column represents labels
from tflearn.data_utils import load_csv
X, Y = load_csv('train_set', has_header=False)
#test_feature, test_label = load_csv('dm_c2b_v3_cate_deal_test_set', has_header=False)

#print(X.shape)
#print(Y.shape)

X=np.reshape(X, [-1, 36])
Y=np.reshape(Y, [-1, 1])

print(X.shape)
print(Y.shape)

# Linear Regression graph
input_ = tflearn.input_data(shape=[None,36])
#linear = tflearn.single_unit(input_)
linear = tflearn.fully_connected(input_, 1, activation='linear')
regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square',
                                metric='R2', learning_rate=0.01)
m = tflearn.DNN(regression)
m.fit(X, Y, n_epoch=70, show_metric=True,batch_size=10000)

#print("\nRegression result:")
#print("Y = " + str(m.get_weights(linear.W)) +
      #"*X + " + str(m.get_weights(linear.b)))

