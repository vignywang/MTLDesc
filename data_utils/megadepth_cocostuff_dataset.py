#
# Created by ZhangYuyang on 2020/6/29
#
import os
import random
from glob import glob

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from data_utils.dataset_tools import HomographyAugmentation
from data_utils.dataset_tools import ImgAugTransform
from data_utils.dataset_tools import draw_image_keypoints


class MegadepthCocostuffDataset(Dataset):
    """
    Combination of MegaDetph and COCOStuff
    """
    def __init__(self, **config):
        self.config = config
        self.data_list = self._format_file_list(
            config['coco_dataset_dir'],
            config['megadepth_dataset_dir'],
            config['megadepth_label_dir'],
        )

        self.homography = HomographyAugmentation()
        self.photometric = ImgAugTransform()
        self.fix_grid = self._generate_fixed_grid()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_info = self.data_list[idx]
        if data_info['type'] == 'coco-stuff':
            return self._get_coco_data(data_info)
        elif data_info['type'] == 'mega':
            return self._get_mega_data(data_info)
        else:
            assert False

    def _get_mega_data(self, data_info):
        image_dir = data_info['image']
        info_dir = data_info['info']
        label_dir = data_info['label']

        image12 = cv.imread(image_dir)[:, :, ::-1].copy()  # 交换BGR为RGB
        image1, image2 = np.split(image12, 2, axis=1)
        h, w, _ = image1.shape

        if torch.rand([]).item() < 0.5:
            image1 = self.photometric(image1)
            image2 = self.photometric(image2)

        info = np.load(info_dir)
        desp_point1 = info["desp_point1"]
        desp_point2 = info["desp_point2"]
        valid_mask = info["valid_mask"]
        not_search_mask = info["not_search_mask"]

        label = np.load(label_dir)
        points1 = label["points_0"]
        points2 = label["points_1"]

        # 2.1 得到第一副图点构成的热图
        heatmap1 = self._convert_points_to_heatmap(points1)
        point_mask1 = torch.ones_like(heatmap1)

        # 2.2 得到第二副图点构成的热图
        heatmap2 = self._convert_points_to_heatmap(points2)
        point_mask2 = torch.ones_like(heatmap2)

        image1 = (torch.from_numpy(image1).to(torch.float) * 2. / 255. - 1.).permute((2, 0, 1)).contiguous()
        image2 = (torch.from_numpy(image2).to(torch.float) * 2. / 255. - 1.).permute((2, 0, 1)).contiguous()

        desp_point1 = torch.from_numpy(desp_point1)
        desp_point2 = torch.from_numpy(desp_point2)

        valid_mask = torch.from_numpy(valid_mask).to(torch.float)
        not_search_mask = torch.from_numpy(not_search_mask).to(torch.float)

        fake_label = torch.ones((self.config['height'], self.config['width']), dtype=torch.int64) * 255

        return {
            "image": image1,
            "point_mask": point_mask1,
            "heatmap": heatmap1,
            "warped_image": image2,
            "warped_point_mask": point_mask2,
            "warped_heatmap": heatmap2,
            "desp_point": desp_point1,
            "warped_desp_point": desp_point2,
            "valid_mask": valid_mask,
            "not_search_mask": not_search_mask,
            'label': fake_label,
            'warped_label': fake_label.clone(),
        }

    def _get_coco_data(self, data_info):
        image, label, point = self._load_data(data_info)
        image, label, point = self._preprocess(image, label, point)

        # construct image pair
        point_mask = np.ones_like(label, dtype=np.float32)
        if torch.rand([]).item() < 0.5:
            warped_image, warped_label, homography = image.copy(), label.copy(), np.eye(3)
            warped_point = point.copy()
            warped_point_mask = point_mask.copy()
        else:
            homography = self.homography.sample(self.config['height'], self.config['width'])
            warped_image = cv.warpPerspective(image, homography, None, None, cv.INTER_AREA, borderValue=0)
            warped_label = cv.warpPerspective(label, homography, None, None, cv.INTER_NEAREST, borderValue=0).astype(np.int64)
            warped_point = self.homography.warp_keypoints(point, homography, self.config['height'], self.config['width'])
            warped_point_mask = cv.warpPerspective(point_mask, homography, None, None, cv.INTER_AREA, borderValue=0).astype(np.float32)

        if torch.rand([]).item() < 0.5:
            image = self.photometric(image)
            warped_image = self.photometric(warped_image)

        # construct heatmap for point training
        heatmap = self._convert_points_to_heatmap(point)
        warped_heatmap = self._convert_points_to_heatmap(warped_point)

        # sample descriptor points for descriptor learning
        desp_point = self._random_sample_point()
        shape = image.shape

        warped_desp_point, valid_mask, not_search_mask = self._generate_warped_point(
            desp_point, homography, shape[0], shape[1])

        # debug use
        # image_point = draw_image_keypoints(image, point, show=False)
        # warped_image_point = draw_image_keypoints(warped_image, warped_point, show=False)
        # cat_all = np.concatenate((image, warped_image), axis=1)
        # cat_all = np.concatenate((image_point, warped_image_point), axis=1)
        # cv2.imwrite("/home/yuyang/tmp/coco_tmp/%d.jpg" % idx, cat_all)
        # cv.imshow("cat_all", cat_all)
        # cv.waitKey()

        image = image.astype(np.float32) * 2. / 255. - 1.
        warped_image = warped_image.astype(np.float32) * 2. / 255. - 1.

        label -= 1
        label[label == -1] = 255
        label = torch.from_numpy(label.astype(np.int64))
        warped_label -= 1
        warped_label[warped_label == - 1] = 255
        warped_label = torch.from_numpy(warped_label.astype(np.int64))

        image = torch.from_numpy(image).permute((2, 0, 1))
        warped_image = torch.from_numpy(warped_image).permute((2, 0, 1))

        desp_point = torch.from_numpy(self._scale_point_for_sample(desp_point))
        warped_desp_point = torch.from_numpy(self._scale_point_for_sample(warped_desp_point))

        valid_mask = torch.from_numpy(valid_mask)
        not_search_mask = torch.from_numpy(not_search_mask)

        return {
            'image': image,  # [1,h,w]
            'label': label,
            'warped_image': warped_image,  # [1,h,w]
            'warped_label': warped_label,
            'desp_point': desp_point,  # [n,1,2]
            'warped_desp_point': warped_desp_point,  # [n,1,2]
            'valid_mask': valid_mask,  # [n]
            'not_search_mask': not_search_mask,  # [n,n]
            'heatmap': heatmap,
            'point_mask': torch.from_numpy(point_mask),
            'warped_heatmap': warped_heatmap,
            'warped_point_mask': torch.from_numpy(warped_point_mask),
        }

    def _preprocess(self, image, label, point):
        # Scaling
        h, w = label.shape

        scale_factor = random.choice(self.config['scales'])
        h, w = (int(h * scale_factor), int(w * scale_factor))
        image = cv.resize(image, (w, h), interpolation=cv.INTER_LINEAR)
        label = Image.fromarray(label).resize((w, h), resample=Image.NEAREST)
        label = np.asarray(label, dtype=np.int64)
        point = point * scale_factor

        # Padding to fit for crop_size
        h, w = label.shape
        pad_h = max(self.config['height'] - h, 0)
        pad_w = max(self.config['width'] - w, 0)
        pad_kwargs = {
            "top": int(pad_h/2),
            "bottom": pad_h - int(pad_h/2),
            "left": pad_w,
            "right": pad_w - int(pad_w/2),
            "borderType": cv.BORDER_CONSTANT,
        }
        if pad_h > 0 or pad_w > 0:
            image = cv.copyMakeBorder(image, value=0, **pad_kwargs)
            label = cv.copyMakeBorder(label, value=0, **pad_kwargs)
            point -= np.array((int(pad_h/2), int(pad_w/2)))

        # Cropping
        h, w = label.shape
        start_h = random.randint(0, h - self.config['height'])
        start_w = random.randint(0, w - self.config['width'])
        end_h = start_h + self.config['height']
        end_w = start_w + self.config['width']
        image = image[start_h:end_h, start_w:end_w]
        label = label[start_h:end_h, start_w:end_w]
        point = point - np.array((start_h, start_w))

        if self.config['flip']:
            # Random flipping
            if random.random() < 0.5:
                image = np.fliplr(image).copy()  # HWC
                label = np.fliplr(label).copy()  # HW
                point[:, 1] = self.config['width'] - 1 - point[:, 1]

        return image, label, point

    def _load_data(self, data_info):
        # Set paths
        image_path = data_info['image_path']
        label_path = data_info['label_path']
        point_path = data_info['point_path']

        # Load an image and label
        # change bgt to rgb
        image = cv.imread(image_path, cv.IMREAD_COLOR)[:, :, ::-1].astype(np.float32)
        label = np.load(label_path)
        point = np.load(point_path)

        return image, label, point

    @ staticmethod
    def _generate_warped_point(point, homography, height, width, threshold=16):
        """
        根据投影变换得到变换后的坐标点，有效关系及不参与负样本搜索的矩阵
        Args:
            point: [n,2] 与warped_point一一对应
            homography: 点对之间的变换关系

        Returns:
            not_search_mask: [n,n] type为float32的mask,不搜索的位置为1
        """
        # 得到投影点的坐标
        point = np.concatenate((point[:, ::-1], np.ones((point.shape[0], 1))), axis=1)[:, :, np.newaxis]  # [n,3,1]
        project_point = np.matmul(homography, point)[:, :, 0]
        project_point = project_point[:, :2] / project_point[:, 2:3]
        project_point = project_point[:, ::-1]  # 调换为y,x的顺序

        # 投影点在图像范围内的点为有效点，反之则为无效点
        boarder_0 = np.array((0, 0), dtype=np.float32)
        boarder_1 = np.array((height-1, width-1), dtype=np.float32)
        valid_mask = (project_point >= boarder_0) & (project_point <= boarder_1)
        valid_mask = np.all(valid_mask, axis=1)
        invalid_mask = ~valid_mask

        # 根据无效点及投影点之间的距离关系确定不搜索的负样本矩阵

        dist = np.linalg.norm(project_point[:, np.newaxis, :] - project_point[np.newaxis, :, :], axis=2)
        not_search_mask = ((dist <= threshold) | invalid_mask[np.newaxis, :]).astype(np.float32)
        return project_point.astype(np.float32), valid_mask.astype(np.float32), not_search_mask

    def _scale_point_for_sample(self, point):
        """
        将点归一化到[-1,1]的区间范围内，并调换顺序为x,y，方便采样
        Args:
            point: [n,2] y,x的顺序，原始范围为[0,height-1], [0,width-1]
        Returns:
            point: [n,1,2] x,y的顺序，范围为[-1,1]
        """
        org_size = np.array((self.config['height']-1, self.config['width']-1), dtype=np.float32)
        point = ((point * 2. / org_size - 1.)[:, ::-1])[:, np.newaxis, :].copy()
        return point

    def _random_sample_point(self):
        """
        根据预设的输入图像大小，随机均匀采样坐标点
        """
        grid = self.fix_grid.copy()
        # 随机选择指定数目个格子

        point_list = []
        for i in range(grid.shape[0]):
            y_start, x_start, y_end, x_end = grid[i]
            rand_y = np.random.randint(y_start, y_end)
            rand_x = np.random.randint(x_start, x_end)
            point_list.append(np.array((rand_y, rand_x), dtype=np.float32))
        point = np.stack(point_list, axis=0)

        return point

    def _generate_fixed_grid(self, option=None):
        """
        预先采样固定间隔的225个图像格子
        """
        if option == None:
            y_num = 20
            x_num = 20
        else:
            y_num = option[0]
            x_num = option[1]

        grid_y = np.linspace(0, self.config['height']-1, y_num+1, dtype=np.int)
        grid_x = np.linspace(0, self.config['width']-1, x_num+1, dtype=np.int)

        grid_y_start = grid_y[:y_num].copy()
        grid_y_end = grid_y[1:y_num+1].copy()
        grid_x_start = grid_x[:x_num].copy()
        grid_x_end = grid_x[1:x_num+1].copy()

        grid_start = np.stack((np.tile(grid_y_start[:, np.newaxis], (1, x_num)),
                               np.tile(grid_x_start[np.newaxis, :], (y_num, 1))), axis=2).reshape((-1, 2))
        grid_end = np.stack((np.tile(grid_y_end[:, np.newaxis], (1, x_num)),
                             np.tile(grid_x_end[np.newaxis, :], (y_num, 1))), axis=2).reshape((-1, 2))
        grid = np.concatenate((grid_start, grid_end), axis=1)

        return grid

    def _convert_points_to_heatmap(self, points):
        """
        将原始点位置经下采样后得到heatmap与incmap，heatmap上对应下采样整型点位置处的值为1，其余为0；incmap与heatmap一一对应，
        在关键点位置处存放整型点到亚像素角点的偏移量，以及训练时用来屏蔽非关键点inc量的incmap_valid
        Args:
            points: [n,2]

        Returns:
            heatmap: [h,w] 关键点位置为1，其余为0
            incmap: [2,h,w] 关键点位置存放实际偏移，其余非关键点处的偏移量为0
            incmap_valid: [h,w] 关键点位置为1，其余为0，用于训练时屏蔽对非关键点偏移量的训练，只关注关键点的偏移量

        """
        height = self.config['height']
        width = self.config['width']

        # localmap = self.localmap.clone()
        # padded_heatmap = torch.zeros(
        #     (height+self.g_paddings*2, width+self.g_paddings*2), dtype=torch.float)
        heatmap = torch.zeros((height, width), dtype=torch.float)

        num_pt = points.shape[0]
        if num_pt > 0:
            for i in range(num_pt):
                pt = points[i]
                pt_y_float, pt_x_float = pt

                pt_y_int = round(pt_y_float)
                pt_x_int = round(pt_x_float)

                pt_y = int(pt_y_int)  # 对真值点位置进行下采样,这里有量化误差
                pt_x = int(pt_x_int)

                # 排除掉经下采样后在边界外的点
                if pt_y < 0 or pt_y > height - 1:
                    continue
                if pt_x < 0 or pt_x > width - 1:
                    continue

                # 关键点位置在heatmap上置1，并在incmap上记录该点离亚像素点的偏移量
                heatmap[pt_y, pt_x] = 1.0

        return heatmap

    @staticmethod
    def _format_file_list(coco_dataset_root, mega_dataset_dir, mega_label_dir):
        data_list = []

        # format megadepth related list
        mega_image_list = glob(os.path.join(mega_dataset_dir, '*.jpg'))
        mega_image_list = sorted(mega_image_list)
        mega_type = 'mega'
        for img in mega_image_list:
            img_name = img.split('/')[-1].split('.')[0]
            info = img[:-3] + 'npz'
            label = os.path.join(mega_label_dir, img_name + '.npz')
            data_list.append(
                {
                    'type': mega_type,
                    'image': img,
                    'info': info,
                    'label': label,
                }
            )

        # format coco related list
        coco_type = 'coco-stuff'
        file_list = os.path.join(coco_dataset_root, 'imageLists', 'train.txt')
        file_list = tuple(open(file_list, "r"))
        file_list = [id_.rstrip() for id_ in file_list]

        for image_id in file_list:
            image_path = os.path.join(coco_dataset_root, 'processed_dataset', image_id + ".jpg")
            label_path = os.path.join(coco_dataset_root, 'processed_dataset', image_id + "_label.npy")
            point_path = os.path.join(coco_dataset_root, 'processed_dataset', image_id + '_point.npy')

            data_list.append(
                {
                    'type': coco_type,
                    'image_path': image_path,
                    'point_path': point_path,
                    'label_path': label_path,
                }
            )

        return data_list



