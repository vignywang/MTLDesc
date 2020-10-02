#
# Created by ZhangYuyang on 2020/6/30
#
# demo of extracting points and descriptors for different methods
import os
import argparse

import yaml
import torch
import numpy as np
import cv2 as cv

from models import get_model


def mnn_matcher(descriptors_a, descriptors_b):
    device = descriptors_a.device
    sim = descriptors_a @ descriptors_b.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t().data.cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='choose which method to test')
    parser.add_argument('--image1_dir', type=str, required=True, help='The first image dir')
    parser.add_argument('--image2_dir', type=str, required=True, help='The second image dir')
    parser.add_argument('--out_dir', type=str, required=True, help='The output path')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    image1_name = args.image1_dir.split('/')[-1].split('.')[0]
    image2_name = args.image2_dir.split('/')[-1].split('.')[0]
    if not os.path.exists(args.out_dir):
        print('The output path does not exist: %s' % args.out_dir)
        assert False

    # image1 = cv.imread('tmp_test/1.ppm')[:, :, ::-1].copy() # convert bgr to rgb
    # image2 = cv.imread('tmp_test/2.ppm')[:, :, ::-1].copy()
    image1 = cv.imread(args.image1_dir)[:, :, ::-1].copy()
    image2 = cv.imread(args.image2_dir)[:, :, ::-1].copy()

    image1 = cv.resize(image1, None, None, fx=0.25, fy=0.25, interpolation=cv.INTER_AREA)
    image2 = cv.resize(image2, None, None, fx=0.25, fy=0.25, interpolation=cv.INTER_AREA)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu:0')

    with get_model(config['model']['name'])(**config['model']) as net:
        predictions1 = net.predict(image1)
        predictions2 = net.predict(image2)

        keypoints1, descriptors1 = predictions1['keypoints'], predictions1['descriptors']
        keypoints2, descriptors2 = predictions2['keypoints'], predictions2['descriptors']

        # match
        matches = mnn_matcher(torch.from_numpy(descriptors1).to(device),
                              torch.from_numpy(descriptors2).to(device))
        matches = [cv.DMatch(matches[i, 0], matches[i, 1], 0) for i in range(matches.shape[0])]

        # random choose 10% to show
        np.random.seed(1234)
        choices = np.random.choice(np.arange(0, len(matches)), int(0.1*len(matches)), replace=False)
        matches = [matches[c] for c in choices]

        # only to keep the matched keypoints
        keep_matches = []
        keep_keypoints1 = []
        keep_keypoints2 = []
        for i, match in enumerate(matches):
            keep_keypoints1.append(cv.KeyPoint(keypoints1[match.queryIdx, 0], keypoints1[match.queryIdx, 1], 0))
            keep_keypoints2.append(cv.KeyPoint(keypoints2[match.trainIdx, 0], keypoints2[match.trainIdx, 1], 0))
            keep_matches.append(cv.DMatch(i, i, 0))

        # convert keypoints to cv type
        # keypoints1 = [cv.KeyPoint(keypoints1[i, 0], keypoints1[i, 1], 0) for i in range(keypoints1.shape[0])]
        # keypoints2 = [cv.KeyPoint(keypoints2[i, 0], keypoints2[i, 1], 0) for i in range(keypoints2.shape[0])]

        # draw matches
        match_image = cv.drawMatches(image1, keep_keypoints1, image2, keep_keypoints2, keep_matches, None, None)
        cv.imwrite(os.path.join(args.out_dir, '{}_{}_{}.jpg'.format(image1_name, image2_name, config['model']['name'])),
                   match_image[:, :, ::-1])





