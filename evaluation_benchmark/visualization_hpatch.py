#
# Created by ZhangYuyang on 2020/7/6
#
import argparse
from pathlib import Path

import yaml
import torch
import numpy as np
import cv2 as cv
from tqdm import tqdm

from hpatch_related.hpatch_dataset import HPatchDataset
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


def check_match(point1, point2, homo, threshold=3):
    point1 = np.concatenate((point1, np.ones((point1.shape[0], 1))), axis=1)[:, :, np.newaxis]  # [n,3,1]
    point1_proj = np.matmul(homo, point1)
    point1_proj = point1_proj[:, :2, 0] / point1_proj[:, 2:3, 0]

    dist = np.linalg.norm(point1_proj - point2, axis=1)

    return dist <= threshold


def draw_matches(point1, point2, image1=None, image2=None, prior_image=None, prior_shape=None, color=(0, 0, 255)):
    """
    if there has prior image, then draw the matches on the prior_image
    """
    assert point1.shape == point2.shape

    if prior_image is not None:
        assert prior_shape is not None
        offset = np.array((prior_shape[1], 0))
        image = prior_image
    else:
        assert image1 is not None and image2 is not None
        offset = np.array((image2.shape[1], 0))
        image = np.concatenate((image1, image2), axis=1)

    point2 = point2 + offset
    point1 = point1.astype(np.int)
    point2 = point2.astype(np.int)

    # draw lines
    for i in range(point1.shape[0]):
        cv.line(image, tuple(point1[i]), tuple(point2[i]), color=color, thickness=1)

    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--ratio', type=float, default=0.5, help='The ratio of showed and all')
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

    dataset = HPatchDataset(**config['hpatches'])

    idx = 0
    with get_model(config['model']['name'])(**config['model']) as net:
        for data in tqdm(dataset, total=len(dataset)):
            image1 = data['first_image']
            image2 = data['second_image']
            gt_homo = data['gt_homography']
            image_type = data['image_type']

            if image1.shape[0] < 600 or image1.shape[1] < 600 or image2.shape[0] < 600 or image2.shape[1] < 600:
                continue

            # predict
            predictions1 = net.predict(image1)
            predictions2 = net.predict(image2)

            keypoints1, descriptors1 = predictions1['keypoints'], predictions1['descriptors']
            keypoints2, descriptors2 = predictions2['keypoints'], predictions2['descriptors']

            # match
            matches = mnn_matcher(torch.from_numpy(descriptors1).to(device),
                                  torch.from_numpy(descriptors2).to(device))

            # generate matched points
            keypoints1 = np.stack([keypoints1[m[0]] for m in matches])
            keypoints2 = np.stack([keypoints2[m[1]] for m in matches])
            matches = np.arange(0, keypoints1.shape[0])

            # generate correct matched points
            correct = check_match(keypoints1, keypoints2, gt_homo)
            keypoints1 = keypoints1[correct]
            keypoints2 = keypoints2[correct]

            # random choose 50% to show
            if image_type == 'illumination':
                ratio = 0.5
            else:
                ratio = 0.5
            np.random.seed(1902)
            choices = np.random.choice(np.arange(0, keypoints1.shape[0]), int(ratio * keypoints1.shape[0]), replace=False)

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






