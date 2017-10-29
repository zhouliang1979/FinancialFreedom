# encoding: utf-8

# Author: Shaoguang Cheng
# Email: shaoguang.csg@alibaba-inc.com

import os

import tensorflow as tf
import pandas as pd
import numpy as np

from parse_args import *
from naive_wdl_learning import *


tf.logging.set_verbosity(tf.logging.INFO)

class TeseSellerWDL(object):

    def __init__(self, model_conf, data_conf):
        self.model = naive_wdl(model_conf, data_conf)
        self.data_conf = data_conf
        self.model_conf = model_conf

    def _load_data(self, df, do_prediction=False):
        continuous_cols = {k: tf.constant((df[k].values.astype(float)-self.mean[k])/self.std[k])
                           for k in self.data_conf.continuous_columns}

        multi_category_cols = {k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values.astype(str),
            shape=[df[k].size, 1])
            for k in self.data_conf.multi_category_columns}

        binary_cols = {k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values.astype(int),
            shape=[df[k].size, 1])
            for k in self.data_conf.binary_columns}

        print('multi_category_cols:')
        print({k: df[k].dtype for k in multi_category_cols})
        print('binary_cols:')
        print({k: df[k].dtype for k in binary_cols})
        print('continuous_cols:')
        print({k: df[k].dtype for k in continuous_cols})

        feature_cols = dict(continuous_cols)
        feature_cols.update(binary_cols)
        feature_cols.update(multi_category_cols)

        if not do_prediction:
            label = tf.constant(df[self.data_conf.target_column].astype(int).values)
            return feature_cols, tf.reshape(label, [-1,1])

        return feature_cols

    def train(self):
        df_train = pd.read_csv(self.data_conf.train_file, sep=',', header='infer')
        df_evaluation = pd.read_csv(self.data_conf.evaluate_file, sep=',', header='infer')

        df_train = df_train.dropna(how='any', axis=0)
        df_evaluation = df_evaluation.dropna(how='any', axis=0)

        self.mean = {k:np.mean(df_train[k]) for k in self.data_conf.continuous_columns}
        self.std = {k:np.std(df_train[k])+1.0e-8 for k in self.data_conf.continuous_columns}

        # evaluate every N steps
        validation_monitor1 = tf.contrib.learn.monitors.ValidationMonitor(
            input_fn=lambda: self._load_data(df_evaluation),
            eval_steps = 100,
            every_n_steps=1000)

        validation_monitor2 = tf.contrib.learn.monitors.ValidationMonitor(
            input_fn=lambda: self._load_data(df_train),
            eval_steps = 100,
            every_n_steps=1000,
            early_stopping_rounds=200)

        self.model.fit(input_fn=lambda :self._load_data(df_train),
                       monitors=[validation_monitor1, validation_monitor2]
                       )
 #       self.model.evaluate(input_fn=lambda: self._load_data(df_evaluation))

    def predict(self):
        df_data = pd.read_csv(self.data_conf.test_file, sep=',', header='infer')
        df_data = df_data.dropna(how='any', axis=0)

        predicted_result = self.model.predict(input_fn=lambda: self._load_data(df_data, do_prediction=True))

        return predicted_result

    def predict_proba(self, is_save=True):
        chunks = pd.read_csv(self.data_conf.test_file, sep=',', header='infer', chunksize=50000)
        append_cols = pd.DataFrame()
        pred_prob = []

        for chunk in chunks:
            pred_prob_tmp = self.model.predict_proba(input_fn=lambda: self._load_data(chunk, do_prediction=True))
            pred_prob += pred_prob_tmp.tolist()
            append_cols_tmp = pd.DataFrame()
            for k in self.model_conf.append_cols:
                if append_cols_tmp.size == 0:
                    append_cols_tmp = chunk[k]
                else:
                    append_cols_tmp = pd.concat((append_cols_tmp, chunk[k]), axis=1)

            if append_cols.size == 0:
                append_cols = append_cols_tmp
            else:
                append_cols = pd.concat((append_cols, append_cols_tmp), axis=0)

        if is_save:
            self._save_result(pd.DataFrame(append_cols), np.asarray(pred_prob))

        return pred_prob

    def _save_result(self, data, prediction_result):
        """

        :param data: DataFrame
        :param prediction_result:
        :return:
        """
        filename = os.path.join(self.model_conf.model_dir, 'prediction_result')

        append_cols = []
        format_str = ''
        if self.model_conf.append_cols is not None:
            for k in self.model_conf.append_cols:
                append_cols.append(data[k])
                format_str += '%s,'
        format_str += '%4f,%4f,%d'
        predicted_label = [0 if x[1] > x[0] else 1 for x in prediction_result]
        print(prediction_result)
        append_cols.append(prediction_result[:,0])
        append_cols.append(prediction_result[:,1])
        append_cols.append(predicted_label)

        result_data = np.asarray(append_cols).transpose()
        np.savetxt(filename, result_data, format_str)


if __name__ == '__main__':
    data_conf = DataConf()
    model_conf = ModelConf()

    model = TeseSellerWDL(model_conf=model_conf, data_conf=data_conf)
    model.train()
    model.predict_proba()