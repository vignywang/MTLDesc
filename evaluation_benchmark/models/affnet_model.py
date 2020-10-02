#
# Created by ZhangYuyang on 2020/6/28
#
import torch
import numpy as np

from nets.affnet_utils.SparseImgRepresenter import ScaleSpaceAffinePatchExtractor
from nets.affnet_utils.HardNet import HardNet, L2Norm, HardTFeatNet
from nets.affnet_utils.architectures import AffNetFast


class AffnetModel(object):

    def __init__(self, **config):
        default_config = {
            'weights': '',
            'descriptor_weights': '',
        }
        default_config.update(config)

        if default_config['weights'] == '' or default_config['descriptor_weights'] == '':
            assert False

        if torch.cuda.is_available():
            print('gpu is available, set device to cuda !')
            self.device = torch.device('cuda:0')
            self.gpu_count = 1
            self.cuda = True
        else:
            print('gpu is not available, set device to cpu !')
            self.device = torch.device('cpu')
            self.cuda = False

        self.model = AffNetFast(PS=32)
        self.model.eval()

        # resume parameters
        print('=> loading checkpoint {}'.format(default_config['weights']))
        checkpoint = torch.load(default_config['weights'], map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])

        self.detector = ScaleSpaceAffinePatchExtractor(mrSize=5.192, num_features=3000,
                                                  border=5, num_Baum_iters=1,
                                                  AffNet=self.model)
        self.detector.eval()

        self.descriptor = HardNet()
        hncheckpoint = torch.load(default_config['descriptor_weights'])
        self.descriptor.load_state_dict(hncheckpoint['state_dict'])
        self.descriptor.eval()

        if self.cuda:
            self.detector = self.detector.cuda()
            self.descriptor = self.descriptor.cuda()

    def load_grayscale_var(self, img):
        img = np.mean(np.array(img), axis=2)
        var_image = torch.from_numpy(img.astype(np.float32))
        var_image_reshape = var_image.view(1, 1, var_image.size(0), var_image.size(1))
        if self.cuda:
            var_image_reshape = var_image_reshape.cuda()

        return var_image_reshape

    @staticmethod
    def get_geometry_and_descriptors(img, det, desc, do_ori=True):
        with torch.no_grad():
            LAFs, resp = det(img, do_ori=do_ori)
            patches = det.extract_patches_from_pyr(LAFs, PS=32)
            descriptors = desc(patches)
        return LAFs, descriptors

    def predict(self, img, keys='*'):
        shape = img.shape
        img = self.load_grayscale_var(img)

        with torch.no_grad():
            LAFs, descriptors = self.get_geometry_and_descriptors(img, self.detector, self.descriptor)
            keypoints = torch.stack((LAFs[:, 0, 2], LAFs[:, 1, 2]), dim=1)  # [n,2]
            keypoints = keypoints.cpu().numpy()
            descriptors = descriptors.cpu().numpy()

        predictions = {
            'shape': shape,
            'keypoints': keypoints,
            'descriptors': descriptors,
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


