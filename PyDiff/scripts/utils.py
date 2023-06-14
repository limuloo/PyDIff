import os

import numpy as np
import torch
from PIL import Image
import warnings
import cv2
import torch.nn.functional as F


def pad_tensor(input, divide=16):
    
    height_org, width_org = input.shape[2], input.shape[3]

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div  / 2)
            pad_bottom = int(height_div  - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = torch.nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom

def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    if len(input.size()) == 3:
        height, width = input.shape[1], input.shape[2]
        return input[:, pad_top: height - pad_bottom, pad_left: width - pad_right]
    else:
        height, width = input.shape[2], input.shape[3]
        return input[:,:, pad_top: height - pad_bottom, pad_left: width - pad_right]

def hiseq_color_cv2_img(img):
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    result = cv2.merge((bH, gH, rH))
    return result



def generate_position_encoding(H, W, L=1):
    x_range = torch.arange(H) / H
    y_range = torch.arange(W) / W
    x, y = torch.meshgrid(x_range, y_range)
    y_sin, y_cos = torch.sin(y), torch.cos(y)
    x_sin, x_cos = torch.sin(x), torch.cos(x)
    position_encoding = []
    for _ in range(L):
        position_encoding += [2**L * y_sin, 
                              2**L * y_cos, 
                              2**L * x_sin, 
                              2**L * x_cos]
    position_encoding = [coding[:, :, None] for coding in position_encoding]
    
    position_encoding = torch.cat(position_encoding, dim=2)
    return position_encoding.numpy()


def check_mid_beta(mid_beta, beta0, n):
    betas = np.linspace(beta0, mid_beta, n, dtype=np.float64)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    amplify_scale = np.sqrt((1 - alphas_cumprod) / alphas_cumprod)
    return amplify_scale[-1] <= 1.0

def check_end_beta(beta0, mid_beta, end_beta, n, alphas_cumprod_limit):
    assert n % 2 == False, "okkk"
    betas = np.concatenate(
        [
            np.linspace(beta0, mid_beta, n // 2),
            np.linspace(mid_beta, end_beta, n // 2)
        ],
        axis = 0
    )
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    return alphas_cumprod[-1] <= alphas_cumprod_limit

def stretch_linear(betas):
    betas = betas
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    n = betas.shape[0]

    ans_l = betas[0]
    ans_r = betas[-1]
    mid_beta = -1
    while ans_r - ans_l >= 1e-7:
        ans_m = (ans_l + ans_r) / 2
        if check_mid_beta(mid_beta=ans_m, beta0=betas[0] ,n=n//2):
            mid_beta = ans_m
            ans_l = ans_m
        else:
            ans_r = ans_m
    assert mid_beta != -1, "didn't find proper mid_beta!"
    
    end_beta = -1
    ans_l = mid_beta
    ans_r = 1.0
    while ans_r - ans_l >= 1e-7:
        ans_m = (ans_l + ans_r) / 2
        if check_end_beta(beta0=betas[0], mid_beta=mid_beta, end_beta=ans_m, n=n, alphas_cumprod_limit=alphas_cumprod[-1]):
            end_beta = ans_m
            ans_r = ans_m
        else:
            ans_l = ans_m
    assert end_beta != -1, "didn't find proper end_beta!"

    betas = np.concatenate(
        [
            np.linspace(betas[0], mid_beta, n // 2, dtype=np.float64), 
            np.linspace(mid_beta, end_beta, n // 2, dtype=np.float64), 
        ],
        axis = 0
    )
    return betas

