import sys
import os
import math
import collections
from abc import ABCMeta, abstractmethod

import cv2
import numpy as np
import tensorflow as tf

from .cnn_wrapper.aslfeat import ASLFeatNet

sys.path.append('..')


def load_frozen_model(pb_path, prefix='', print_nodes=False):
    """Load frozen model (.pb file) for testing.
    After restoring the model, operators can be accessed by
    graph.get_tensor_by_name('<prefix>/<op_name>')
    Args:
        pb_path: the path of frozen model.
        prefix: prefix added to the operator name.
        print_nodes: whether to print node names.
    Returns:
        graph: tensorflow graph definition.
    """
    if os.path.exists(pb_path):
        with tf.io.gfile.GFile(pb_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                name=prefix
            )
            if print_nodes:
                for op in graph.get_operations():
                    print(op.name)
            return graph
    else:
        print('Model file does not exist', pb_path)
        exit(-1)


def recoverer(sess, model_path):
    """
    Recovery parameters from a pretrained model.
    Args:
        sess: The tensorflow session instance.
        model_path: Checkpoint file path.
    Returns:
        Nothing
    """
    restore_var = tf.compat.v1.global_variables()
    restorer = tf.compat.v1.train.Saver(restore_var)
    restorer.restore(sess, model_path)


def dict_update(d, u):
    """Improved update for nested dictionaries.

    Arguments:
        d: The dictionary to be updated.
        u: The update dictionary.

    Returns:
        The updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class BaseModel(metaclass=ABCMeta):
    """Base model class."""

    @abstractmethod
    def _run(self, data):
        raise NotImplementedError

    @abstractmethod
    def _init_model(self):
        raise NotImplementedError

    @abstractmethod
    def _construct_network(self):
        raise NotImplementedError

    def run_test_data(self, data):
        """"""
        out_data = self._run(data)
        return out_data

    def __init__(self, model_path, **config):
        self.model_path = model_path
        # Update config
        self.config = dict_update(getattr(self, 'default_config', {}), config)
        self._init_model()
        ext = os.path.splitext(model_path)[1]

        sess_config = tf.compat.v1.ConfigProto()
        sess_config.gpu_options.allow_growth = True

        if ext.find('.pb') == 0:
            graph = load_frozen_model(self.model_path, print_nodes=False)
            self.sess = tf.compat.v1.Session(graph=graph, config=sess_config)
        elif ext.find('.ckpt') == 0:
            self._construct_network()
            self.sess = tf.compat.v1.Session(config=sess_config)
            recoverer(self.sess, model_path)

    def close(self):
        self.sess.close()
        tf.compat.v1.reset_default_graph()


class FeatModel(BaseModel):
    endpoints = None
    default_config = {'max_dim': 2048}

    def _init_model(self):
        return

    def _run(self, data):
        assert len(data.shape) == 3
        max_dim = max(data.shape[0], data.shape[1])
        H, W, _ = data.shape

        if max_dim > self.config['max_dim']:
            downsample_ratio = self.config['max_dim'] / float(max_dim)
            data = cv2.resize(data, (0, 0), fx=downsample_ratio, fy=downsample_ratio)
            data = data[..., np.newaxis]
        data_size = data.shape

        if self.config['config']['multi_scale']:
            scale_f = 1 / (2**0.50)
            min_scale = max(0.3, 128 / max(H, W))
            n_scale = math.floor(max(math.log(min_scale) / math.log(scale_f), 1))
            sigma = 0.8
        else:
            n_scale = 1
        
        descs, kpts, scores = [], [], []
        for i in range(n_scale):
            if i > 0:
                data = cv2.GaussianBlur(data, None, sigma / scale_f)
                data = cv2.resize(data, dsize=None, fx=scale_f, fy=scale_f)[..., np.newaxis]
            
            feed_dict = {"input:0": np.expand_dims(data, 0)}
            returns = self.sess.run(self.endpoints, feed_dict=feed_dict)
            descs.append(np.squeeze(returns['descs'], axis=0))
            kpts.append(np.squeeze(returns['kpts'], axis=0) * np.array([W / data.shape[1], H / data.shape[0]], dtype=np.float32))
            scores.append(np.squeeze(returns['scores'], axis=0))
        
        descs = np.concatenate(descs, axis=0)
        kpts = np.concatenate(kpts, axis=0)
        scores = np.concatenate(scores, axis=0)

        idxs = np.negative(scores).argsort()[0:self.config['config']['kpt_n']]

        descs = descs[idxs]
        kpts = kpts[idxs] * np.array([W / data_size[1], H / data_size[0]], dtype=np.float32)
        scores = scores[idxs]
        return descs, kpts, scores

    def _construct_network(self):
        ph_imgs = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 1), name='input')
        mean, variance = tf.nn.moments(
            tf.cast(ph_imgs, tf.float32), axes=[1, 2], keep_dims=True)
        norm_input = tf.nn.batch_normalization(ph_imgs, mean, variance, None, None, 1e-5)
        config_dict = {'det_config': self.config['config']}
        tower = ASLFeatNet({'data': norm_input}, is_training=False, resue=False, **config_dict)
        self.endpoints = tower.endpoints
