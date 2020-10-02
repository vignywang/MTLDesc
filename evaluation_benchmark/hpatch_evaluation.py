#
# Created by ZhangYuyang on 2020/6/29
#
import os
import argparse

import numpy as np

from utils.evaluator import Evaluator
from utils.evaluator import evaluate


def generate_read_function(prediction_path, method):
    def read_function(seq_name, im_idx):
        aux = np.load(os.path.join(prediction_path, method, seq_name, '%d.npz' % im_idx))
        return aux['shape'], aux['keypoints'], aux['descriptors']

    return read_function


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method_name', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, default='/data/yuyang/hpatches-sequences-release')
    parser.add_argument('--output_root', type=str, default='results')

    args = parser.parse_args()

    read_function = generate_read_function(args.output_root, args.method_name)

    evaluator = Evaluator()
    errors = evaluate(read_function, args.dataset_path, evaluator)

    print('----------------------------')
    evaluator.print_stats('i_eval_stats')
    evaluator.print_stats('v_eval_stats')
    evaluator.print_stats('all_eval_stats')

    with open(os.path.join(args.output_root, args.method_name, 'evaluation_results.txt'), 'w') as f:
        evaluator.save_results(f)





