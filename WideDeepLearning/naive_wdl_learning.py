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
    sparse_feature_cross,
    embedding_column,
    crossed_column,
    bucketized_column
)

from tensorflow.contrib.framework import get_or_create_global_step


class naive_wdl(object):
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
            #            continuous_feature_columns = {column:real_valued_column(column_name=column)
            #             for column in data_conf.continuous_columns}
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
            self.model = tf.contrib.learn.LinearClassifier(model_dir=model_conf.model_dir,
                                                           feature_columns=wide_columns,
                                                           n_classes=model_conf.n_classes,
                                                           optimizer=self._get_linear_optimizer,
                                                           config=tf.contrib.learn.RunConfig(save_checkpoints_secs=600)
                                                           )
        elif model_conf.model_type == 1:  # deep
            self.model = tf.contrib.learn.DNNClassifier(model_dir=model_conf.model_dir,
                                                        feature_columns=deep_columns,
                                                        n_classes=model_conf.n_classes,
                                                        hidden_units=model_conf.hidden_units,
                                                        dropout=model_conf.dropout,
                                                        optimizer=self._get_dnn_optimizer,
                                                        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=600)
                                                        )
        else:  # wide and deep
            self.model = tf.contrib.learn.DNNLinearCombinedClassifier(
                model_dir=model_conf.model_dir,
                n_classes=model_conf.n_classes,
                linear_feature_columns=wide_columns,
                dnn_feature_columns=deep_columns,
                dnn_hidden_units=model_conf.hidden_units,
                dnn_dropout=model_conf.dropout,
                linear_optimizer=self._get_linear_optimizer,
                dnn_optimizer=self._get_dnn_optimizer,
                config=tf.contrib.learn.RunConfig(save_checkpoints_secs=600)
            )
        print(self.model)

    def fit(self, input_fn, monitors=None):
        print('train model ...')
        self.model.fit(input_fn=input_fn, steps=self.model_conf.max_iter, monitors=monitors)

    def predict(self, input_fn):
        print('do prediction ...')
        predicted_result = self.model.predict(input_fn=input_fn)
        return predicted_result

    def predict_proba(self, input_fn):
        print('do prediction ...')
        proba_result = self.model.predict_proba(input_fn=input_fn)
        return proba_result

    def evaluate(self, input_fn):
        print('evaluate model ...')
        results = self.model.evaluate(input_fn=input_fn, steps=1)
        for key in sorted(results):
            print("%s: %s" % (key, results[key]))

    def _get_lr(self):
        global_step = get_or_create_global_step()
        if self.model_conf.lr_policy == 'fixed':
            return self.model_conf.base_lr
        elif self.model_conf.lr_policy == 'step':
            return tf.train.exponential_decay(learning_rate=self.model_conf.base_lr,
                                              global_step=global_step,
                                              decay_steps=self.model_conf.step_size,
                                              decay_rate=0.95)
        else:
            raise Exception("Unkown learning rate policy")

    def _get_dnn_optimizer(self):
        return tf.train.AdagradOptimizer(learning_rate=self._get_lr())

    @property
    def _get_linear_optimizer(self):
        return tf.train.FtrlOptimizer(learning_rate=self._get_lr(),
                                      l1_regularization_strength=self.model_conf.alpha,
                                      l2_regularization_strength=self.model_conf.beta)
