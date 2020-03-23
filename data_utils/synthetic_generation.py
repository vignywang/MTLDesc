# 
# Created by ZhangYuyang on 2019/8/30
# 摘自superpoint，待整理
#
import argparse

import numpy as np
import cv2
import os
from pathlib import Path  # 路径合并
from tqdm import tqdm  # 进度条
from data_utils import synthetic_tools


default_config = {
    'primitives': 'all',
    'truncate': {},
    'validation_size': -1,
    'test_size': -1,
    'on-the-fly': False,
    'cache_in_memory': False,
    'suffix': None,
    'add_augmentation_to_test_set': False,
    'num_parallel_calls': 10,
    'generation': {
        'split_sizes': {'training': 10000, 'validation': 200, 'test': 500},
        'image_size': [960, 1280],
        'random_seed': 0,
        'params': {
            'generate_background': {
                'min_kernel_size': 150, 'max_kernel_size': 500,
                'min_rad_ratio': 0.02, 'max_rad_ratio': 0.031},
            'draw_stripes': {'transform_params': (0.1, 0.1)},
            'draw_multiple_polygons': {'kernel_boundaries': (50, 100)}
        },
    },
    'preprocessing': {
        'resize': [240, 320],
        'blur_size': 11,
    },
    'augmentation': {
        'photometric': {
            'enable': False,
            'primitives': 'all',
            'params': {},
            'random_order': True,
        },
        'homographic': {
            'enable': False,
            'params': {},
            'valid_border_margin': 0,
        },
    }
}
drawing_primitives = [
    'draw_lines',
    'draw_polygon',
    'draw_multiple_polygons',
    'draw_ellipses',
    'draw_star',
    'draw_checkerboard',
    'draw_stripes',
    'draw_cube',
    'gaussian_noise'
]


def dump_primitive_data(primitive, config, project_data_root):  # 其中primitive指定生成图形化的种类（圆，线，多边形，星。。）
    temp_dir = Path(project_data_root + "/synthetic", primitive)
    if os.path.exists(temp_dir):
        print("{} is already prepared".format(primitive))
        return
    synthetic_tools.set_random_state(np.random.RandomState(
        config['generation']['random_seed']))  # 随机种子生成
    for split, size in config['generation']['split_sizes'].items():
        # 针对训练，验证，测试集分别生成文件夹（每个均含有image核points两个分别存储图像核兴趣点的文件夹）
        im_dir, pts_dir = [Path(temp_dir, i, split) for i in ['images', 'points']]
        im_dir.mkdir(parents=True, exist_ok=True)
        pts_dir.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(size), desc=split, leave=False):  # tqdm 以进度条显示进度,对每一类数据集(t,v,t)分别 生成对应size的图像
            image = synthetic_tools.generate_background(  # 先生成背景
                config['generation']['image_size'],
                **config['generation']['params']['generate_background'])
            points = np.array(getattr(synthetic_tools, primitive)(
                image, **config['generation']['params'].get(primitive, {})))  # 完成作图和兴趣点记录
            points = np.flip(points, 1)  # 交换点x，y的坐标，之前作图使用的坐标和tensorflo对图像坐标的定义是反的(笛卡尔坐标与矩阵坐标）

            b = config['preprocessing']['blur_size']
            image = cv2.GaussianBlur(image, (b, b), 0)  # 模糊图像，也是为了使图形更真实而不是点阵形式
            points = (points * np.array(config['preprocessing']['resize'], np.float)
                      / np.array(config['generation']['image_size'], np.float))
            image = cv2.resize(image, tuple(config['preprocessing']['resize'][::-1]),
                               interpolation=cv2.INTER_LINEAR)
# 写入图像并以npy形式保存对应的兴趣点
            cv2.imwrite(str(Path(im_dir, '{}.png'.format(i))), image)
            np.save(Path(pts_dir, '{}.npy'.format(i)), points)

