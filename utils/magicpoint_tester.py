#
# Created by ZhangYuyang on 2019/8/14
#
import time
import torch
import torch.nn.functional as f

from nets.superpoint_net import SuperPointNet
from data_utils.synthetic_dataset import SyntheticValTestDataset
from utils.evaluation_tools import mAPCalculator
from utils.utils import spatial_nms


class MagicPointTester(object):

    def __init__(self, params):
        self.params = params
        self.logger = params.logger
        if torch.cuda.is_available():
            self.logger.info('gpu is available, set device to cuda!')
            self.device = torch.device('cuda:0')
        else:
            self.logger.info('gpu is not available, set device to cpu!')
            self.device = torch.device('cpu')

        # 初始化测试数据集
        test_dataset = SyntheticValTestDataset(params, dataset_type='validation', add_noise=True)

        # 初始化模型
        model = SuperPointNet()

        # 初始化测评计算子
        mAP_calculator = mAPCalculator()

        self.test_dataset = test_dataset
        self.test_length = len(test_dataset)
        self.model = model
        self.mAP_calculator = mAP_calculator

    def test(self, ckpt_file):

        if ckpt_file == None:
            print("Please input correct checkpoint file dir!")
            return

        # 从预训练的模型中恢复参数
        model_dict = self.model.state_dict()
        pretrain_dict = torch.load(ckpt_file)
        model_dict.update(pretrain_dict)
        self.model.load_state_dict(model_dict)
        self.model.to(self.device)

        # 重置测评算子参数
        self.mAP_calculator.reset()

        self.model.eval()

        self.logger.info("*****************************************************")
        self.logger.info("Testing model %s" % ckpt_file)

        start_time = time.time()
        count = 0

        for i, data in enumerate(self.test_dataset):
            image = data['image']
            gt_point = data['gt_point']
            gt_point = gt_point.numpy()

            image = image.to(self.device).unsqueeze(dim=0)
            # 得到原始的经压缩的概率图，概率图每个通道64维，对应空间每个像素是否为关键点的概率
            _, _, prob = self.model(image)
            # 将概率图展开为原始图像大小
            prob = f.pixel_shuffle(prob, 8)
            # 进行非极大值抑制
            prob = spatial_nms(prob)
            prob = prob.detach().cpu().numpy()[0, 0]

            self.mAP_calculator.update(prob, gt_point)
            if i % 10 == 0:
                print("Having tested %d samples, which takes %.3fs" % (i, (time.time()-start_time)))
                start_time = time.time()
            count += 1
            # if count % 1000 == 0:
            #     break

        mAP, _, _ = self.mAP_calculator.compute_mAP()

        self.logger.info("The mean Average Precision : %.4f of %d samples" % (mAP, count))
        self.logger.info("Testing done.")
        self.logger.info("*****************************************************")





