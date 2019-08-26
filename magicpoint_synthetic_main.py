#
# Created by ZhangYuyang on 2019/8/9
#
import os
import torch
import numpy as np
import argparse
import glob

from utils.logger import get_logger
from utils.trainers import MagicPointSyntheticTrainer
from utils.testers import MagicPointSyntheticTester
from utils.testers import HPatchTester

# make the result reproducible
torch.manual_seed(3928)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(2933)


class Parameters:

    synthetic_dataset_dir = '/data/MegPoint/dataset/synthetic'
    hpatch_dataset_dir = '/data/MegPoint/dataset/hpatch'

    ckpt_root = './magicpoint_ckpt'
    ckpt_dir = ''
    log_root = './magicpoint_log'
    log_dir = ''
    logger = None
    gpus = None

    # common params
    height = 240
    width = 320
    hpatch_height = 240  # 480
    hpatch_width = 320  # 640

    # training relating params
    lr = 0.001
    batch_size = 64
    epoch_num = 60
    log_freq = 100
    num_workers = 8
    prefix = 'exp1'
    do_augmentation = True

    # testing relating params
    save_threshold_curve = True
    nms_threshold = 4

    # HPatch tester relating params
    detection_threshold = 0.015  # 0.005
    correct_epsilon = 3
    rep_top_k = 300
    desp_top_k = 1000


def myparser():
    parser = argparse.ArgumentParser(description="Pytorch MagicPoint Training")
    parser.add_argument("--gpus", type=str, default='0')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_threshold_curve", type=bool, default=True)
    parser.add_argument("--prefix", type=str, default='exp1')
    return parser.parse_args()


args = myparser()
params = Parameters()

# read some params from bash
params.gpus = args.gpus
params.lr = args.lr
params.batch_size = args.batch_size
params.prefix = args.prefix
params.num_workers = args.num_workers
params.save_threshold_curve = args.save_threshold_curve

# set and mkdir relative dir when necessary
if not os.path.exists(params.ckpt_root):
    os.mkdir(params.ckpt_root)
if not os.path.exists(params.log_root):
    os.mkdir(params.log_root)
params.ckpt_dir = os.path.join(params.ckpt_root, params.prefix + '_%.4f' % params.lr + '_%d' % params.batch_size)
params.log_dir = os.path.join(params.log_root, params.prefix + '_%.4f' % params.lr + '_%d' % params.batch_size)
if not os.path.exists(params.ckpt_dir):
    os.mkdir(params.ckpt_dir)
if not os.path.exists(params.log_dir):
    os.mkdir(params.log_dir)
params.logger = get_logger(params.log_dir)

# set gpu devices
os.environ['CUDA_VISIBLE_DEVICES'] = params.gpus
params.gpus = [i for i in range(len(params.gpus.split(',')))]
params.logger.info("Set CUDA_VISIBLE_DEVICES to %s" % params.gpus)

# log the parameters
params.logger.info('batch size is %d' % params.batch_size)
params.logger.info('training epoch is %d' % params.epoch_num)
params.logger.info('training input size is [%d, %d]' % (params.height, params.width))
params.logger.info('non-maximum suppression threshold: %d' % params.nms_threshold)
params.logger.info('haptch testing input size is [%d, %d]' % (params.hpatch_height, params.hpatch_width))
params.logger.info('learning rate is %.4f' % params.lr)
params.logger.info('number worker is %d' % params.num_workers)
params.logger.info('prefix is %s' % params.prefix)

# initialize the trainer and train
# magicpoint_trainer = MagicPointSyntheticTrainer(params)
# magicpoint_trainer.train()

# initialize the tester and test all checkpoint file in the folder
magicpoint_synthetic_tester = MagicPointSyntheticTester(params)
magicpoint_hpatch_tester = HPatchTester(params)

# # choose test mode
# mode = 'all'
mode = 'only_hpatch'
# mode = 'only_synthetic'
# mode = 'only_synthetic_one_image'

# ckpt_file = '/home/zhangyuyang/project/development/MegPoint/magicpoint_ckpt/good_results/adam_adaption_0.0010_64/model_99.pt'
# ckpt_file = '/home/zhangyuyang/project/development/MegPoint/magicpoint_ckpt/good_results/adam_0.0010_64/model_59.pt'
ckpt_file = '/home/zhangyuyang/project/development/MegPoint/magicpoint_ckpt/good_results/superpoint_magicleap.pth'

if mode == 'all':
    ckpt_files = glob.glob(os.path.join(params.ckpt_dir, "model_*"))
    ckpt_files = sorted(ckpt_files)
    for ckpt_file in ckpt_files:
        magicpoint_synthetic_tester.test(ckpt_file)
        magicpoint_hpatch_tester.test_model_repeatability(ckpt_file)

elif mode == 'only_hpatch':
    # magicpoint_hpatch_tester.test_FAST_repeatability()
    magicpoint_hpatch_tester.test_model_repeatability(ckpt_file)
    # magicpoint_hpatch_tester.test_model_descriptors(ckpt_file)
    # magicpoint_hpatch_tester.test_orb_descriptors()

elif mode == 'only_synthetic':
    magicpoint_synthetic_tester.test(ckpt_file)

elif mode == 'only_synthetic_one_image':
    image_dir = '/data/MegPoint/dataset/synthetic/draw_multiple_polygons/images/test/76.png'
    magicpoint_synthetic_tester.test_single_image(ckpt_file, image_dir)







