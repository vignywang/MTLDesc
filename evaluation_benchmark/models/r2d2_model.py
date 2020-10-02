# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use
import time
import os, pdb
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as tvf
from torchsummary import summary


from utils.utils import model_size, torch_set_gpu
from nets.patchnet import *


RGB_mean = [0.485, 0.456, 0.406]
RGB_std  = [0.229, 0.224, 0.225]

norm_RGB = tvf.Compose([tvf.ToTensor(), tvf.Normalize(mean=RGB_mean, std=RGB_std)])

def load_network(model_fn): 
    checkpoint = torch.load(model_fn)
    print("\n>> Creating net = " + checkpoint['net']) 
    net = eval(checkpoint['net'])
    nb_of_weights = model_size(net)
    print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")

    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
    return net.eval()


class NonMaxSuppression (torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr
    
    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        maxima *= (reliability   >= self.rel_thr)

        return maxima.nonzero().t()[2:4]


def extract_multiscale( net, img, detector, scale_f=2**0.25, 
                        min_scale=0.0, max_scale=1, 
                        min_size=256, max_size=1024, 
                        verbose=False):
    old_bm = torch.backends.cudnn.benchmark 
    torch.backends.cudnn.benchmark = False # speedup
    
    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"
    
    assert max_scale <= 1
    s = 1.0 # current scale factor
    
    X,Y,S,C,Q,D = [],[],[],[],[],[]
    while  s+0.001 >= max(min_scale, min_size / max(H,W)):
        if s-0.001 <= min(max_scale, max_size / max(H,W)):
            nh, nw = img.shape[2:]
            if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
            # extract descriptors
            with torch.no_grad():
                res = net(imgs=[img])
                
            # get output and reliability map
            descriptors = res['descriptors'][0]
            reliability = res['reliability'][0]
            repeatability = res['repeatability'][0]

            # normalize the reliability for nms
            # extract maxima and descs
            y,x = detector(**res) # nms
            c = reliability[0,0,y,x]
            q = repeatability[0,0,y,x]
            d = descriptors[0,:,y,x].t()
            n = d.shape[0]

            # accumulate multiple scales
            X.append(x.float() * W/nw)
            Y.append(y.float() * H/nh)
            S.append((32/s) * torch.ones(n, dtype=torch.float32, device=d.device))
            C.append(c)
            Q.append(q)
            D.append(d)
        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H*s), round(W*s)
        img = F.interpolate(img, (nh,nw), mode='bilinear', align_corners=False)

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    Y = torch.cat(Y)
    X = torch.cat(X)
    S = torch.cat(S) # scale
    scores = torch.cat(C) * torch.cat(Q) # scores = reliability * repeatability
    XYS = torch.stack([X,Y,S], dim=-1)
    D = torch.cat(D)
    return XYS, D, scores


class R2d2Model(object):

    def __init__(self, **config):
        default_config = {
            'weights': '',
            'top_k': 5000,
            'scale_f': 2**0.25,
            'min_size': 256,
            'max_size': 1024,
            'min_scale': 0,
            'max_scale': 1.0,
            'reliability_thr': 0.7,
            'repeatability_thr': 0.7,
        }
        default_config.update(config)
        self.config = default_config

        # create the non-maxima detector
        self.detector = NonMaxSuppression(
            rel_thr=self.config['reliability_thr'],
            rep_thr=self.config['repeatability_thr'])

        if default_config['weights'] == '':
            assert False

        # load the network
        if torch.cuda.is_available():
            print('gpu is available, set device to cuda !')
            self.device = torch.device('cuda:0')
            self.gpu_count = 1
            self.iscuda = True
        else:
            print('gpu is not available, set device to cpu !')
            self.device = torch.device('cpu')
            self.iscuda = False

        self.net = load_network(default_config['weights'])
        if self.iscuda:
            self.net = self.net.to(self.device)

        self.time_collect = []

    def size(self):
        print('R2D2 Size Summary:')
        summary(self.net, input_size=(1, 3, 240, 320))

    def average_inference_time(self):
        average_time = sum(self.time_collect) / len(self.time_collect)
        info = ('R2D2 average inference time: {}ms / {}fps'.format(
            round(average_time*1000), round(1/average_time))
        )
        print(info)
        return info

    def predict(self, img, keys='*'):
        shape = img.shape
        img = norm_RGB(img)[None]
        if self.iscuda:
            img = img.to(self.device)

        # time start
        start_time = time.time()

        # extract keypoints/descriptors for a single image
        xys, desc, scores = extract_multiscale(self.net, img, self.detector,
                                               scale_f=self.config['scale_f'],
                                               min_scale=self.config['min_scale'],
                                               max_scale=self.config['max_scale'],
                                               min_size=self.config['min_size'],
                                               max_size=self.config['max_size'],
                                               verbose=True)

        self.time_collect.append(time.time()-start_time)

        xys = xys.cpu().numpy()
        desc = desc.cpu().numpy()
        scores = scores.cpu().numpy()
        idxs = scores.argsort()[-self.config['top_k'] or None:]

        predictions = {
            'shape': shape,
            'keypoints': xys[idxs][:, :2],
            'descriptors': desc[idxs],
            'scores': scores[idxs],
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

