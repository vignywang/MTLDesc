#
# Created by ZhangYuyang on 2019/8/9
#


class Parameters:

    synthetic_dataset_dir = "/data/MegPoint/dataset/synthetic"

    height = 240
    width = 320
    do_augmentation = True
    homography_params = {
        'translation': True,
        'rotation': True,
        'scaling': True,
        'perspective': True,
        'scaling_amplitude': 0.2,
        'perspective_amplitude_x': 0.2,
        'perspective_amplitude_y': 0.2,
        'patch_ratio': 0.8,
        'max_angle': 1.57,  # 3.14
        'allow_artifacts': True,
        'translation_overflow': 0.0,
    }

















