# encoding: utf-8

# Author: Shaoguang Cheng
# Email: shaoguang.csg@alibaba-inc.com

import os

import tensorflow as tf
import pandas as pd
import numpy as np

from parse_args import *
from regression_wdl import *


COLUMNS = ["cate_level1_id","cate_level2_id","leaf_cate_id","price_range","user_group_id","channel","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14","f15","f16","f17","f18","f19","f20","f21","f22","f23","f24","f25","f26","f27","f28","f29","f30","f31","f32","f33","f34","f35","target"]

tf.logging.set_verbosity(tf.logging.INFO)

class CateGMVWDL(object):

    def __init__(self, model_conf, data_conf):
        self.model = regression_wdl(model_conf, data_conf)
        self.data_conf = data_conf
        self.model_conf = model_conf

    def _load_data(self, df, do_prediction=False):
        #zscore = lambda x:(x-x.mean())/ (x.std() + 1.0e-8)
        zscore = lambda x:x.mean()
        continuous_cols = {k: tf.constant(df[k].groupby(df['leaf_cate_id']).transform(zscore).values.astype(np.float64))
                           for k in self.data_conf.continuous_columns}
        sess = tf.InteractiveSession()
        for k in self.data_conf.continuous_columns:
            print k, sess.run(continuous_cols[k])
            print k, df[k]
        
        multi_category_cols = {k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values.astype(str),
            dense_shape=[df[k].size, 1])
            for k in self.data_conf.multi_category_columns}
      
        binary_cols = {k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values.astype(int),
            shape=[df[k].size, 1])
            for k in self.data_conf.binary_columns}

        feature_cols = dict(continuous_cols)
        feature_cols.update(binary_cols)
        feature_cols.update(multi_category_cols)

        if not do_prediction:
            label = tf.constant( df[self.data_conf.target_column].groupby(df['leaf_cate_id']).transform(zscore).values.astype(np.float64) )
            #print self.data_conf.target_column, sess.run(label)
            return feature_cols, tf.reshape(label, [-1,1])

        return feature_cols

    def train(self):
        df_train = pd.read_csv(self.data_conf.train_file, names=COLUMNS, skipinitialspace=True)
        df_evaluation = pd.read_csv(self.data_conf.evaluate_file, names=COLUMNS, skipinitialspace=True)

        df_train = df_train.dropna(how='any', axis=0)
        df_evaluation = df_evaluation.dropna(how='any', axis=0)

        #self.mean = {k:np.mean(df_train[k]) for k in self.data_conf.continuous_columns}
        #self.std = {k:np.std(df_train[k])+1.0e-8 for k in self.data_conf.continuous_columns}

        #self.target_mean = df_train[self.data_conf.target_column].mean()
        #self.target_std = df_train[self.data_conf.target_column].std() + 1.0e-8

        self.model.fit(input_fn=lambda :self._load_data(df_train))
        self.model.evaluate(input_fn=lambda: self._load_data(df_evaluation))

    def predict(self, is_save=True):
        df_test = pd.read_csv(self.data_conf.test_file, names=COLUMNS, skipinitialspace=True)
        df_test = df_test.dropna(how='any', axis=0)
        predicted_result = self.model.predict(input_fn=lambda : self._load_data(df_test, do_prediction=True))
        if is_save:
            self._save_result(self._load_data(df_test), list(predicted_result))

    def _save_result(self, data, predicted_result):
        filename = os.path.join(self.model_conf.model_dir, 'prediction_result')
        append_cols = []
        format_str = ''
        if self.model_conf.append_cols is not None:
            for k in self.model_conf.append_cols:
                append_cols.append(data[k])
                format_str += '%s\t'
        append_cols.append(predicted_result)
        format_str += '%4f'

        result_data = np.asarray(append_cols).transpose()
        #print("predict:", predicted_result)
        #print("result:", result_data)
        #print("format:", format_str)
        np.savetxt(filename, result_data, format_str)


if __name__ == '__main__':
    data_conf = DataConf()
    model_conf = ModelConf()

    model = CateGMVWDL(model_conf=model_conf, data_conf=data_conf)
    model.train()
    model.predict()
