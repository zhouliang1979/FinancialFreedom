# encoding: utf-8

# Author: Shaoguang Cheng
# Email: shaoguang.csg@alibaba-inc.com

import yaml

import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_conf_file', 'conf/model_conf.yaml', 'Path to the model config yaml file')
tf.app.flags.DEFINE_string('data_conf_file', 'conf/data_conf.yaml', 'Path to the data config yaml file')


# singleton pattern
def singleton(cls):
    _instances = {}

    def _singleton(*args, **kwargs):
        if cls not in _instances:
            _instances[cls] = cls(*args, **kwargs)
        return _instances[cls]
    return _singleton


@singleton
class DataConf(object):

    def __init__(self):
        with open(FLAGS.data_conf_file) as data_conf_file:
            data_conf = yaml.load(data_conf_file)

        self.train_and_test_file = data_conf.get('train_and_test_file', None)
        self.train_file = data_conf.get('train_file', None)
        self.evaluate_file = data_conf.get('evaluate_file', None)
        self.test_file = data_conf.get('test_file', None)
        self.target_column = data_conf.get('target_column', None)
        self.binary_columns = data_conf.get('binary_columns', None)
        self.multi_category_columns = data_conf.get('multi_category_columns', None)
        self.continuous_columns = data_conf.get('continuous_columns', None)
        self.crossed_columns = data_conf.get('crossed_columns', None)
        self.bucketized_columns = data_conf.get('bucketized_columns', None)

        self._check_param()

    def _check_param(self):
        pass


@singleton
class ModelConf(object):

    def __init__(self):
        with open(FLAGS.model_conf_file) as model_conf_file:
            model_conf = yaml.load(model_conf_file)

        self.display = model_conf.get('display', 1)
        self.model_dir = model_conf.get('model_dir', '/tmp/')
        self.log_dir = model_conf.get('log_dir', '/tmp/')
        self.append_cols = model_conf.get('append_cols', None)
        self.model_type = model_conf.get('model_type', 0)
        self.problem_type = model_conf.get('problem_type', 0)
        self.n_classes = model_conf.get('n_classes', 2)
        self.max_iter = model_conf.get('max_iter', 100000)
        self.base_lr = model_conf.get('base_lr', 0.01)
        self.lr_policy = model_conf.get('lr_policy', 'step')
        self.step_size = model_conf.get('step_size', 10000)
        self.alpha = model_conf.get('alpha', 0.0)
        self.beta = model_conf.get('beta', 0.0)
        self.hidden_units = model_conf.get('hidden_units', [])
        self.embedding_dimension = model_conf.get('embedding_dimension', 16)
        self.dropout = model_conf.get('dropout', None)

        self._check_param()

    def _check_param(self):
        pass

if __name__ == '__main__':
    x = DataConf()
    print x.bucketized_columns
    print x.crossed_columns
