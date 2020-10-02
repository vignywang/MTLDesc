#
# Created by ZhangYuyang on 2020/7/11
#
import argparse
import yaml
from pathlib import Path

import cv2 as cv
import torch
import numpy as np
from tqdm import tqdm

from datasets import get_sub_dataset
from models import get_model


class RatioMatcher(object):

    def __init__(self, ratio):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu:0")
        self.ratio = ratio

    def __call__(self, desp_0, desp_1):
        # norm_desp_0 = torch.norm(desp_0, dim=1, keepdim=True) ** 2  # [n,1]
        # norm_desp_1 = torch.norm(desp_1, dim=1, keepdim=True).transpose(0, 1) ** 2  # [1,m]
        xty = torch.matmul(desp_0, desp_1.transpose(0, 1))
        # dist = torch.sqrt((norm_desp_0 + norm_desp_1 - 2*xty + 1e-4))
        dist = 2 * (1 - xty)

        dist, matches = torch.topk(dist, k=2, dim=1, largest=False)

        # ratio check
        valid_idx = dist[:, 0] < (self.ratio**2) * dist[:, 1]

        # dist check
        # valid_idx = valid_idx & (dist[:, 0] < 0.5)
        valid_idx = valid_idx & (dist[:, 0] < 0.45)

        src_idx = torch.arange(0, dist.shape[0], dtype=torch.int, device=self.device)
        dist = dist[valid_idx][:, 0].contiguous().detach().cpu().numpy()

        # mean_dist = np.mean(dist)

        tar_idx = matches[valid_idx][:, 0].contiguous().detach().cpu().numpy()
        src_idx = src_idx[valid_idx].contiguous().detach().cpu().numpy()
        matches = np.stack((src_idx, tar_idx), axis=1)

        # cross check
        keypoint_matches = []
        smallest_distances = dict()

        for i, m in enumerate(matches):
            src, tar = m
            if tar not in smallest_distances:
                smallest_distances[tar] = (dist[i], src)
                keypoint_matches.append((src, tar))
            else:
                old_dist, old_src = smallest_distances[tar]
                if dist[i] < old_dist:
                    smallest_distances[tar] = (dist[i], src)

                    keypoint_matches.remove((old_src, tar))
                    keypoint_matches.append((src, tar))

        return keypoint_matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--ratio', type=float, default=0.9, help='The ratio used in match')
    # parser.add_argument('--keep_ratio', type=float, default=0.5, help='The ratio of showed and all')
    parser.add_argument('--keep_ratio', type=float, default=1.0, help='The ratio of showed and all')
    args = parser.parse_args()

    with open(args.configs, 'r') as cf:
        config = yaml.load(cf)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu:0')

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    print('The output_path: %s' % output_path)

    dataset = get_sub_dataset(config['dataset']['name'])(**config['dataset'])
    matcher = RatioMatcher(args.ratio)

    # prior
    prior = [124, 126, 130, 249, 541, 702, 706, 742, 748, 750, 751, 1248, 1352, 1366, 1411, 1482, 1571, 1734]

    # random choose 50% to show
    np.random.seed(1902)
    idx = 0
    with get_model(config['model']['name'])(**config['model']) as net:
        for data in tqdm(dataset, total=len(dataset)):
            if idx not in prior:
                idx += 1
                continue

            image1 = data['image1']
            image2 = data['image2']

            # predict
            predictions1 = net.predict(image1)
            predictions2 = net.predict(image2)

            keypoints1, descriptors1 = predictions1['keypoints'], predictions1['descriptors']
            keypoints2, descriptors2 = predictions2['keypoints'], predictions2['descriptors']

            # match
            matches = matcher(torch.from_numpy(descriptors1).to(device),
                              torch.from_numpy(descriptors2).to(device))

            # generate matched points
            keypoints1 = np.stack([keypoints1[m[0]] for m in matches])
            keypoints2 = np.stack([keypoints2[m[1]] for m in matches])
            matches = np.arange(0, keypoints1.shape[0])


            choices = np.random.choice(
                np.arange(0, keypoints1.shape[0]), int(args.keep_ratio * keypoints1.shape[0]), replace=False)

            keep_matches = []
            keep_keypoints1 = []
            keep_keypoints2 = []
            for i, c in enumerate(choices):
                keep_keypoints1.append(cv.KeyPoint(keypoints1[c, 0], keypoints1[c, 1], 0))
                keep_keypoints2.append(cv.KeyPoint(keypoints2[c, 0], keypoints2[c, 1], 0))
                keep_matches.append(cv.DMatch(i, i, 0))

            # draw keypoints
            # image1_keypoints = cv.drawKeypoints(image1[:, :, ::-1], keep_keypoints1, None, color=(0, 255, 0))
            # image2_keypoints = cv.drawKeypoints(image2[:, :, ::-1], keep_keypoints2, None, color=(0, 255, 0))

            # draw matches
            image = cv.drawMatches(image1, keep_keypoints1, image2, keep_keypoints2, keep_matches, None, (0, 255, 0))

            # draw good matches
            # image = draw_matches(good_keypoints1, good_keypoints2, image1, image2, color=(0, 255, 0))

            # draw bad matches
            # image = draw_matches(bad_keypoints1, bad_keypoints2, prior_image=image, prior_shape=image2.shape,
            #                      color=(255, 0, 0))

            # save
            cv.imwrite(str(Path(output_path, "%04d.jpg" % idx)), image[:, :, ::-1])
            # cv.imwrite(str(Path(output_path, "%04d_image1_keypoints.jpg" % idx)), image1_keypoints)
            # cv.imwrite(str(Path(output_path, "%04d_image2_keypoints.jpg" % idx)), image2_keypoints)
            # cv.imwrite(str(Path(output_path, "%04d_image1.jpg" % idx)), image1[:, :, ::-1])
            # cv.imwrite(str(Path(output_path, "%04d_image2.jpg" % idx)), image2[:, :, ::-1])
            idx += 1


