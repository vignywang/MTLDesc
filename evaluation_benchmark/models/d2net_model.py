#
# Created by ZhangYuyang on 2020/2/25
#
import time
import torch
import numpy as np
import scipy.io
import scipy.misc
from torchsummary import summary

from nets.d2net import D2Net
from utils.d2net_utils import preprocess_image
from utils.d2net_pyramid import process_multiscale


class D2netModel(object):

    def __init__(self, **config):
        default_config = {
            'weights': '',
            'preprocessing': 'caffe',
            'max_edge': 1600,
            'max_sum_edges': 2800,
            'multiscale': False,
            'use_relu': True,
        }
        default_config.update(config)

        self.name = "d2net"
        print("d2net using multiscale: %s" % default_config['multiscale'])

        self.preprocessing = default_config['preprocessing']
        self.max_edge = default_config['max_edge']
        self.max_sum_edges = default_config['max_sum_edges']
        self.multiscale = default_config['multiscale']

        if torch.cuda.is_available():
            print('gpu is available, set device to cuda !')
            self.device = torch.device('cuda:0')
            self.gpu_count = 1
        else:
            print('gpu is not available, set device to cpu !')
            self.device = torch.device('cpu')

        if default_config['weights'] == '':
            assert False

        # 初始化d2net
        self.model = D2Net(model_file=default_config['weights'], use_relu=default_config['use_relu'],
                           use_cuda=torch.cuda.is_available())

        self.time_collect = []

    def size(self):
        print('D2Net.dense_feature_extraction Size Summary:')
        summary(self.model.dense_feature_extraction, input_size=(3, 240, 320))

    def average_inference_time(self):
        average_time = sum(self.time_collect) / len(self.time_collect)
        info = ('D2Net average inference time: {}ms / {}fps'.format(
            round(average_time*1000), round(1/average_time))
        )
        print(info)
        return info

    def predict(self, image, keys='*'):
        """
        """

        resized_image = image
        shape = image.shape
        assert len(shape) == 3

        if max(resized_image.shape) > self.max_edge:
            resized_image = scipy.misc.imresize(
                resized_image,
                self.max_edge / max(resized_image.shape)
            ).astype('float')
        if sum(resized_image.shape[: 2]) > self.max_sum_edges:
            resized_image = scipy.misc.imresize(
                resized_image,
                self.max_sum_edges / sum(resized_image.shape[: 2])
            ).astype('float')

        fact_i = image.shape[0] / resized_image.shape[0]
        fact_j = image.shape[1] / resized_image.shape[1]

        input_image = preprocess_image(
            resized_image,
            preprocessing=self.preprocessing
        )

        # time start
        start_time = time.time()

        with torch.no_grad():
            if self.multiscale:
                keypoints, scores, descriptors = process_multiscale(
                    torch.tensor(
                        input_image[np.newaxis, :, :, :].astype(np.float32),
                        device=self.device
                    ),
                    self.model
                )
            else:
                keypoints, scores, descriptors = process_multiscale(
                    torch.tensor(
                        input_image[np.newaxis, :, :, :].astype(np.float32),
                        device=self.device
                    ),
                    self.model,
                    scales=[1]
                )

        # Input image coordinates
        keypoints[:, 0] *= fact_i
        keypoints[:, 1] *= fact_j

        point = keypoints[:, :2]
        desp = descriptors

        # time end
        self.time_collect.append(time.time()-start_time)

        max_idx = np.argsort(-scores, axis=0)
        point = point[max_idx]
        desp = desp[max_idx]
        scores = scores[max_idx]

        assert np.all(scores[:-1] - scores[1:] >= 0)

        predictions = {
            'shape': shape,
            "keypoints": point[:, ::-1],  # change to x-y
            "descriptors": desp,
            "scores": scores,
            "input_image": image,
        }
        if keys != '*':
            predictions = {k: predictions[k] for k in keys}

        return predictions

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass



# class HPatchD2Net(BasicD2Net):
#     """专用于hpatch测试的SuperPoint模型"""
#
#     def __init__(self, d2net_ckpt, logger, multiscale):
#         super(HPatchD2Net, self).__init__(d2net_ckpt, logger, multiscale=multiscale)
#
#     def __call__(self, first_image, second_image, *args, **kwargs):
#         first_point, first_desp, first_point_num = self.generate_feature(first_image)
#         second_point, second_desp, second_point_num = self.generate_feature(second_image)
#
#         return first_point, first_point_num, first_desp, second_point, second_point_num, second_desp


