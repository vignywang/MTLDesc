#
# Created by ZhangYuyang on 2020/6/28
#
import time
import numpy as np
import cv2
import tensorflow as tf

from nets.feat_model import FeatModel


def extractor(patch_queue, model, consumer_queue):
    while True:
        queue_data = patch_queue.get()
        if queue_data is None:
            consumer_queue.put(None)
            return
        img, gt_homo = queue_data
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        descs, kpts, _ = model.run_test_data(np.expand_dims(gray, axis=-1))
        consumer_queue.put([img, kpts, descs, gt_homo])
        patch_queue.task_done()


class AslfeatModel(object):

    def __init__(self, **config):
        default_config = {
            'model_path': '',
            'max_dim': 2000,
            'config': {
                'kpt_n': 5000,
                'kpt_refinement': True,
                'deform_desc': 1,
                'score_thld': 0.5,
                'edge_thld': 10,
                'multi_scale': False,
                'multi_level': True,
                'nms_size': 3,
                'eof_mask': 5,
                'need_norm': True,
                'use_peakiness': True,
            },
        }
        default_config.update(config)

        if default_config['model_path'] == '':
            assert False

        self.model = FeatModel(**default_config)

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        self.time_collect = []

    def size(self):
        pass

    def average_inference_time(self):
        average_time = sum(self.time_collect) / len(self.time_collect)
        info = ('ASLFeat average inference time: {}ms / {}fps'.format(
            round(average_time*1000), round(1/average_time))
        )
        print(info)
        return info

    def predict(self, img, keys="*"):
        shape = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]

        # time start
        start_time = time.time()

        descriptors, kpts, scores = self.model.run_test_data(img)

        # time end
        self.time_collect.append(time.time()-start_time)

        predictions = {
            'shape': shape,
            'descriptors': descriptors,
            'keypoints': kpts,
            'scores': scores,
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





