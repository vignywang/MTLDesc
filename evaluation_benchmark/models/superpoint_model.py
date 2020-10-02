#
# Created by ZhangYuyang on 2020/2/24
#
import torch
import numpy as np

from nets.superpoint import SuperPointNetFloat, SuperPointNetFloatNoL2


class SuperpointModel(object):

    def __init__(self, **config):
        default_config = {
            'weights': '',
            'detection_thrshold': 0.015,
            "nms_dist": 4,
            "nms_radius": 4,
            "border_remove": 4,
            'cell': 8,
            'l2': True
        }
        default_config.update(config)
        self.config = default_config
        self.name = "superpoint"
        self.cell = default_config['cell']

        self.detection_threshold = default_config['detection_thrshold']
        print("detection_threshold=%.4f" % self.detection_threshold)

        if torch.cuda.is_available():
            print('gpu is available, set device to cuda !')
            self.device = torch.device('cuda:0')
            self.gpu_count = 1
            self.cuda = True
        else:
            print('gpu is not available, set device to cpu !')
            self.device = torch.device('cpu')
            self.cuda = False

        # 初始化superpoint
        if self.config['l2']:
            print("Initialize SuperPointNetFloat")
            self.model = SuperPointNetFloat().to(self.device)
        else:
            print('Initialize SuperPointNetFloatNoL2')
            self.model = SuperPointNetFloatNoL2().to(self.device)
        self.model = self._load_model_params(default_config['weights'], self.model)

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

    def predict(self, img, keys):
        """ Process a numpy image to extract points and descriptors.
        Input
          img - HxW numpy float32 input image in range [0,1].
        Output
          corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          desc - 256xN numpy array of corresponding unit normalized descriptors.
          heatmap - HxW numpy heatmap in range [0,1] of point confidences.
          """
        assert img.ndim == 2, 'Image must be grayscale.'
        img = (img / 255.).astype(np.float32)

        shape = img.shape
        H, W = img.shape[0], img.shape[1]
        inp = img.copy()
        inp = (inp.reshape(1, H, W))
        inp = torch.from_numpy(inp)
        inp = torch.autograd.Variable(inp).view(1, 1, H, W)
        if self.cuda:
            inp = inp.cuda()

        # Forward pass of network.
        outs = self.model.forward(inp)
        semi, coarse_desc = outs[0], outs[1]
        # Convert pytorch -> numpy.
        semi = semi.data.cpu().numpy().squeeze()
        # --- Process points.
        dense = np.exp(semi)  # Softmax.
        dense = dense / (np.sum(dense, axis=0) + .00001)  # Should sum to 1.
        # Remove dustbin.
        nodust = dense[:-1, :, :]
        # Reshape to get full resolution heatmap.
        Hc = int(H / self.cell)
        Wc = int(W / self.cell)
        nodust = nodust.transpose(1, 2, 0)
        heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        heatmap = np.reshape(heatmap, [Hc * self.cell, Wc * self.cell])
        xs, ys = np.where(heatmap >= self.detection_threshold)  # Confidence threshold.
        if len(xs) == 0:
            return np.zeros((3, 0)), None, None
        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.config['nms_dist'])  # Apply NMS.
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]  # Sort by confidence.
        # Remove points along border.
        bord = self.config['border_remove']
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        # --- Process descriptor.
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            # Interpolate into descriptor map using 2D point locations.
            samp_pts = torch.from_numpy(pts[:2, :].copy())
            samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            if self.cuda:
                samp_pts = samp_pts.cuda()
            desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts, align_corners=True)
            desc = desc.data.cpu().numpy().reshape(D, -1)
            if self.config['l2']:
                desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]

        predictions = {
            'shape': shape,
            'keypoints': pts[:2, ::].T,
            'descriptors': desc.T,
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


# class HPatchSuperPoint(BasicSuperPoint):
#     """专用于hpatch测试的SuperPoint模型"""
#
#     def __init__(self, superpoint_ckpt, logger):
#         super(HPatchSuperPoint, self).__init__(superpoint_ckpt, logger)
#
#     def __call__(self, first_image, second_image, *args, **kwargs):
#         first_point, first_desp, first_point_num = self.generate_feature(first_image)
#         second_point, second_desp, second_point_num = self.generate_feature(second_image)
#
#         return first_point, first_point_num, first_desp, second_point, second_point_num, second_desp



