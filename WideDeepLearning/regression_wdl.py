# encoding: utf-8

# Author: Shaoguang Cheng
# Email: shaoguang.csg@alibaba-inc.com

import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib.learn import monitors
from tensorflow.contrib.layers import (
    real_valued_column,
    sparse_column_with_hash_bucket,
    sparse_column_with_integerized_feature,
    sparse_column_with_keys,
    #sparse_feature_cross,
    embedding_column,
    crossed_column,
    bucketized_column
)

from tensorflow.contrib.framework import get_or_create_global_step


class regression_wdl(object):
    """
    A naive version wide & deep learning
    """

    def __init__(self, model_conf, data_conf):
        print('building model ...')
        self.model_conf = model_conf
        self.data_conf = data_conf

        category_feature_columns = {}
        continuous_feature_columns = {}
        crossed_feature_columns = []
        bucketized_feature_columns = []
        embedding_feature_column = []

        # binary category feature
        if data_conf.binary_columns is not None:
            category_feature_columns = {
            column: sparse_column_with_integerized_feature(column_name=column, bucket_size=2) \
            for column in data_conf.binary_columns}

        # multiple category feature
        if data_conf.multi_category_columns is not None:
            category_feature_columns.update(
                {column: sparse_column_with_hash_bucket(column_name=column, hash_bucket_size=10000) \
                 for column in data_conf.multi_category_columns})

        # continuous feature
        if data_conf.continuous_columns is not None:
            continuous_feature_columns = {column: real_valued_column(column_name=column) \
                                          for column in data_conf.continuous_columns}

        # crossed feature
        if data_conf.crossed_columns is not None:
            for item in data_conf.crossed_columns:
                crossed_feature = []
                for _ in item:
                    crossed_feature.append(category_feature_columns[_])
                crossed_feature_columns.append(crossed_column(crossed_feature, hash_bucket_size=1e6))

        # bucketized feature
        if data_conf.bucketized_columns is not None:
            [bucketized_feature_columns.append(
                bucketized_column(continuous_feature_columns[column], boundaries=boundary)) \
             for column, boundary in data_conf.bucketized_columns.items()]

        # feature embedding
        if len(category_feature_columns) > 0:
            [embedding_feature_column.append(embedding_column(_, dimension=model_conf.embedding_dimension)) \
             for _ in category_feature_columns.values()]

        wide_columns = category_feature_columns.values() + \
                       crossed_feature_columns + \
                       bucketized_feature_columns

        deep_columns = continuous_feature_columns.values() + \
                       embedding_feature_column

        if model_conf.model_type == 0:  # wide
            print "use wide model: tf.contrib.learn.LinearRegressor"
            self.model = tf.contrib.learn.LinearRegressor(model_dir=model_conf.model_dir
                                                           , feature_columns=wide_columns
                                                           )
        elif model_conf.model_type == 1:  # deep
            print "use deep model: tf.contrib.learn.DNNRegressor"
            self.model = tf.contrib.learn.DNNRegressor(model_dir=model_conf.model_dir
                                                        , feature_columns=deep_columns
                                                        , hidden_units=model_conf.hidden_units
                                                        , dropout=model_conf.dropout
                                                        )
        else:  # wide and deep
            print "use wide and deep model: tf.contrib.learn.DNNLinearCombinedRegressor"
            self.model = tf.contrib.learn.DNNLinearCombinedRegressor(model_dir=model_conf.model_dir
                , linear_feature_columns=wide_columns
                , dnn_feature_columns=deep_columns
                , dnn_hidden_units=model_conf.hidden_units
                , dnn_dropout=model_conf.dropout
            )
        print(self.model)

    def fit(self, input_fn, monitors=None):
        print('train model ...')
        #self.model.fit(input_fn=input_fn)
        self.model.fit(input_fn=input_fn, steps=self.model_conf.max_iter)
        for f in self.model.get_variable_names():
            print f, self.model.get_variable_value(f)

    def predict(self, input_fn):
        print('do prediction ...')
        self.predicted_result = self.model.predict(input_fn=input_fn)
        return self.predicted_result

    def evaluate(self, input_fn):
        print('evaluate model ...')
        results = self.model.evaluate(input_fn=input_fn, steps=1)
        for key in sorted(results):
            print("%s: %s" % (key, results[key]))

