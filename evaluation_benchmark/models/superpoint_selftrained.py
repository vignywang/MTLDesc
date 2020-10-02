#
# Created by ZhangYuyang on 2020/2/25
#
import os
import time

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as f
from torchsummary import summary

from nets.MLIFeat_net import SuperPointNet


class SuperpointSelftrained(object):

    def __init__(self,  **config):
        self.name = 'superpoint_selftrained'
        self.config = {
            "detection_threshold": 0.015,
            "num_keypoints": 10000,
            "nms_dist": 4,
            "nms_radius": 4,
            "border_remove": 4,
            'weights': '',
        }
        self.config.update(config)

        self.detection_threshold = self.config["detection_threshold"]
        self.num_keypoints = self.config["num_keypoints"]
        self.nms_dist = self.config["nms_dist"]
        print("detection_threshold=%.4f, num_keypoints=%d" % (self.detection_threshold, self.num_keypoints))

        if torch.cuda.is_available():
            print('gpu is available, set device to cuda !')
            self.device = torch.device('cuda:0')
            self.gpu_count = 1
        else:
            print('gpu is not available, set device to cpu !')
            self.device = torch.device('cpu')

        # 初始化resnet18_fast
        print("Initialize superpoint selftrained")
        self.model = SuperPointNet().to(self.device)

        if self.config['weights'] == '':
            assert False
        self.load(self.config['weights'])

        self.time_collect = []

    def size(self):
        print('SuperPoint Size Summary:')
        summary(self.model, input_size=(3, 240, 320))

    def average_inference_time(self):
        average_time = sum(self.time_collect) / len(self.time_collect)
        info = ('SuperPoint average inference time: {}ms / {}fps'.format(
            round(average_time*1000), round(1/average_time))
        )
        print(info)
        return info

    def _load_model_params(self, ckpt_file, previous_model):
        if ckpt_file is None:
            print("Please input correct checkpoint file dir!")
            return False

        print("Load pretrained model %s " % ckpt_file)

        model_dict = previous_model.state_dict()
        pretrain_dict = torch.load(ckpt_file, map_location=self.device)
        model_dict.update(pretrain_dict)
        previous_model.load_state_dict(model_dict)
        return previous_model

    def load(self, checkpoint_root):
        backbone_ckpt = os.path.join(checkpoint_root, "model.pt")
        self.model = self._load_model_params(backbone_ckpt, self.model)

    def _generate_predict_point(self, heatmap, height, width, top_k=0):
        xs, ys = np.where(heatmap >= self.config['detection_threshold'])
        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        if len(xs) > 0:
            pts[0, :] = ys
            pts[1, :] = xs
            pts[2, :] = heatmap[xs, ys]

            if self.config['nms_radius']:
                pts, _ = self.nms_fast(
                    pts, height, width, dist_thresh=self.config['nms_radius'])
            inds = np.argsort(pts[2, :])
            pts = pts[:, inds[::-1]]  # Sort by confidence.

            # Remove points along border.
            bord = self.config['border_remove']
            toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (width-bord))
            toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (height-bord))
            toremove = np.logical_or(toremoveW, toremoveH)
            pts = pts[:, ~toremove]

            if self.config['num_keypoints']:
                pts = pts[:, :self.config['num_keypoints']]

            pts = pts.transpose()

        point = pts[:, :2][:, ::-1]
        score = pts[:, 2]

        return point, score

    def nms_fast(self, in_corners, H, W, dist_thresh):
        """
        Run a faster approximate Non-Max-Suppression on numpy corners shaped:
          3xN [x_i,y_i,conf_i]^T

        Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
        are zeros. Iterate through all the 1's and convert them either to -1 or 0.
        Suppress points by setting nearby values to 0.

        Grid Value Legend:
        -1 : Kept.
         0 : Empty or suppressed.
         1 : To be processed (converted to either kept or supressed).

        NOTE: The NMS first rounds points to integers, so NMS distance might not
        be exactly dist_thresh. It also assumes points are within image boundaries.

        Inputs
          in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          H - Image height.
          W - Image width.
          dist_thresh - Distance to suppress, measured as an infinty norm distance.
        Returns
          nmsed_corners - 3xN numpy matrix with surviving corners.
          nmsed_inds - N length numpy vector with surviving corner indices.
        """
        grid = np.zeros((H, W)).astype(int) # Track NMS data.
        inds = np.zeros((H, W)).astype(int) # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2,:])
        corners = in_corners[:,inds1]
        rcorners = corners[:2,:].round().astype(int) # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1,i], rcorners[0,i]] = 1
            inds[rcorners[1,i], rcorners[0,i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
          # Account for top and left padding.
            pt = (rc[0]+pad, rc[1]+pad)
            if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
                grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid==-1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]

        return out, out_inds

    def predict(self, img, keys="*"):
        """
        获取一幅灰度图像对应的特征点及其描述子
        Args:
            img: [h,w] 灰度图像,要求h,w能被16整除
        Returns:
            point: [n,2] 特征点,输出点以y,x为顺序
            descriptor: [n,128] 描述子
        """
        # switch to eval mode
        self.model.eval()

        shape = img.shape
        assert shape[2] == 3  # must be rgb

        org_h, org_w = shape[0], shape[1]

        # rescale to 16*
        if org_h % 16 != 0:
            scale_h = int(np.round(org_h / 16.) * 16.)
            sh = org_h / scale_h
        else:
            scale_h = org_h
            sh = 1.0

        if org_w % 16 != 0:
            scale_w = int(np.round(org_w / 16.) * 16.)
            sw = org_w / scale_w
        else:
            scale_w = org_w
            sw = 1.0

        img = cv.resize(img, dsize=(scale_w, scale_h), interpolation=cv.INTER_LINEAR)
        input_img = img.copy()

        # to torch and scale to [-1,1]
        img = torch.from_numpy(img).to(torch.float).unsqueeze(dim=0).permute((0, 3, 1, 2)).to(self.device)
        img = (img / 255.) * 2. - 1.

        # time_start
        start_time = time.time()

        # detector
        _, desp, prob = self.model(img)
        prob = f.pixel_shuffle(prob, 8)

        # 得到对应的预测点
        prob = prob.detach().cpu().numpy()
        prob = prob[0, 0]
        point, score = self._generate_predict_point(prob, height=scale_h, width=scale_w, top_k=self.num_keypoints)  # [n,2]

        # descriptor
        desp = self._generate_descriptor_for_superpoint_desp_head(point, desp, scale_h, scale_w)

        # time end
        self.time_collect.append(time.time()-start_time)

        # scale point back to the original scale and change to x-y
        point = (point * np.array((sh, sw)))[:, ::-1]

        predictions = {
            "shape": shape,
            "keypoints": point,
            "descriptors": desp,
            "scores": score,
            "input_image": input_img,
        }

        if keys != '*':
            predictions = {k: predictions[k] for k in keys}

        return predictions

    def _generate_descriptor_for_superpoint_desp_head(self, point, desp, height, width):
        """
        构建superpoint描述子端的描述子
        """
        point = torch.from_numpy(point[:, ::-1].copy()).to(torch.float).to(self.device)
        # 归一化采样坐标到[-1,1]
        point = point * 2. / torch.tensor((width-1, height-1), dtype=torch.float, device=self.device) - 1
        point = point.unsqueeze(dim=0).unsqueeze(dim=2)  # [1,n,1,2]

        desp = f.grid_sample(desp, point, mode="bilinear", align_corners=True)[0, :, :, 0].transpose(0, 1)

        desp = desp.detach().cpu().numpy()

        return desp

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


