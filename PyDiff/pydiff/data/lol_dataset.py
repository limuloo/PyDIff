import glob
import random
import os

import cv2
import math
import numpy as np
import torch
import torch.utils.data as data
from basicsr.data import degradations as degradations
from basicsr.data.data_util import paths_from_folder
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torchvision.transforms.functional import normalize
from scripts.utils import pad_tensor, hiseq_color_cv2_img, generate_position_encoding

@DATASET_REGISTRY.register()
class LOL_Dataset(data.Dataset):

    def __init__(self, opt):
        super(LOL_Dataset, self).__init__()
        self.opt = opt

        self.gt_root = opt['gt_root']
        self.input_root = opt['input_root']
        self.gt_paths = glob.glob(os.path.join(self.gt_root, '*.png')) + \
                        glob.glob(os.path.join(self.gt_root, '*.jpg'))
        self.mean = self.opt['mean']
        self.std = self.opt['std']
        
    

    def __getitem__(self, index):
        gt_path = self.gt_paths[index]
        gt_name = os.path.split(gt_path)[-1]
        gt_name = gt_name.replace('normal', 'low')
        input_path = os.path.join(self.input_root, gt_name)

        gt_img = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB) / 255.
        input_img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB) / 255.

        if self.opt.get('bright_aug', False):
            bright_aug_range = self.opt.get('bright_aug_range', [0.5, 1.5])
            input_img = input_img * np.random.uniform(*bright_aug_range)
        
        if self.opt.get('concat_with_hiseq', False):
            hiseql = cv2.cvtColor(hiseq_color_cv2_img(cv2.imread(input_path)), cv2.COLOR_BGR2RGB) / 255.
            if self.opt.get('hiseq_random_cat', False) and np.random.uniform(0, 1) < self.opt.get('hiseq_random_cat_p', 0.5):
                input_img = np.concatenate([hiseql, input_img], axis=2)
            else:
                input_img = np.concatenate([input_img, hiseql], axis=2)
            if self.opt.get('random_drop', False):
                if np.random.uniform() <= self.opt.get('random_drop_p', 1.0):
                    random_drop_val = self.opt.get('random_drop_val', 0)
                    if np.random.uniform() < 0.5:
                        input_img[:, :, :3] = random_drop_val
                    else:
                        input_img[:, :, 3:] = random_drop_val
            if self.opt.get('random_drop_hiseq', False):
                if np.random.uniform() < 0.5:
                    input_img[:, :, 3:] = 0

        if self.opt.get('use_flip', False) and np.random.uniform() < 0.5:
            gt_img = cv2.flip(gt_img, 1, gt_img)
            input_img = cv2.flip(input_img, 1, input_img)
        
        if self.opt.get('input_with_low_resolution_hq', False):
            low_resolution_hq_size = self.opt.get('low_resolution_hq_size', 256)
            self.low_resolution_hq = cv2.resize(
                gt_img,
                (low_resolution_hq_size, low_resolution_hq_size)
            )
        
        
        if self.opt.get('concat_with_position_encoding', False):
            H, W, _ = input_img.shape
            L = self.opt.get('position_encoding_L', 1)
            position_encoding = generate_position_encoding(H, W, L)
            input_img = np.concatenate([input_img, position_encoding], axis=2)
        
        if self.opt.get('resize', False):
            resize_size = self.opt['resize_size']
            if self.opt.get('resize_nearest', False):
                gt_img = cv2.resize(gt_img, dsize=(resize_size[1], resize_size[0]), interpolation=cv2.INTER_NEAREST)
                input_img = cv2.resize(input_img, dsize=(resize_size[1], resize_size[0]), interpolation=cv2.INTER_NEAREST)
            else:
                gt_img = cv2.resize(gt_img, dsize=(resize_size[1], resize_size[0]))
                input_img = cv2.resize(input_img, dsize=(resize_size[1], resize_size[0]))

        if self.opt['input_mode'] == 'crop':
            crop_size = self.opt['crop_size']
            H, W, _ = input_img.shape
            assert input_img.shape[:2] == gt_img.shape[:2], f"{input_img.shape}, {gt_img.shape}, {gt_path}"
            h = np.random.randint(0, H - crop_size + 1)
            w = np.random.randint(0, W - crop_size + 1)
            gt_img = gt_img[h: h + crop_size, w: w + crop_size, :]
            input_img = input_img[h: h + crop_size, w: w + crop_size, :]
        if self.opt['input_mode'] == 'pad':
            divide = self.opt['divide']
            gt_img_pt = torch.from_numpy(gt_img.transpose((2, 0, 1)))
            input_img_pt = torch.from_numpy(input_img.transpose((2, 0, 1)))
            gt_img_pt = torch.unsqueeze(gt_img_pt, 0)
            input_img_pt = torch.unsqueeze(input_img_pt, 0)
            gt_img_pt, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(gt_img_pt, divide)
            input_img_pt, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input_img_pt, divide)
            gt_img_pt = gt_img_pt[0, ...]
            input_img_pt = input_img_pt[0, ...]
            gt_img = gt_img_pt.numpy().transpose((1, 2, 0))
            input_img = input_img_pt.numpy().transpose((1, 2, 0))

        gt_img_pt = torch.from_numpy(gt_img.transpose((2, 0, 1)))
        input_img_pt = torch.from_numpy(input_img.transpose((2, 0, 1)))
        if hasattr(self, 'low_resolution_hq'):
            self.low_resolution_hq = torch.from_numpy(
                self.low_resolution_hq.transpose((2, 0, 1))
            ).float()

        input_img_pt = input_img_pt.float()
        gt_img_pt = gt_img_pt.float()
        normalize(input_img_pt, [0.5] * input_img_pt.shape[0], [0.5] * input_img_pt.shape[0], inplace=True)
        normalize(gt_img_pt, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
        if hasattr(self, 'low_resolution_hq'):
            normalize(
                self.low_resolution_hq, 
                [0.5, 0.5, 0.5], 
                [0.5, 0.5, 0.5], 
                inplace=True
            )
        return_dict = {"LR": input_img_pt, "HR": gt_img_pt, "lq_path": gt_path}
        if self.opt['input_mode'] == 'pad':
            return_dict["pad_left"] = pad_left
            return_dict["pad_right"] = pad_right
            return_dict["pad_top"] = pad_top
            return_dict["pad_bottom"] = pad_bottom
        if self.opt.get('input_with_low_resolution_hq', False):
            return_dict["low_resolution_hq"] = self.low_resolution_hq
        return return_dict

    def __len__(self):
        return len(self.gt_paths)

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader
    from collections import OrderedDict
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

if __name__ == '__main__':
    import argparse
    import yaml


    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_args()
    args.launcher = 'none'
    # parse yml to dict
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])

    dataset = LOL_Dataset(opt['datasets']['train'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=True)
    cnt = 0
    for x in dataloader:
        print(cnt)
        print(x['LR'].shape, x['HR'].shape)
        cnt += 1
        if cnt >= 30:
            break