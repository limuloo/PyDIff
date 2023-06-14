from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from .niqe import calculate_niqe
from .psnr_ssim import calculate_psnr, calculate_ssim
from .psnr_pytorch import calculate_psnr_pytorch
from .ssim_pytorch import calculate_ssim_pytorch
from .ssim_lol import calculate_ssim_lol
from .lpips_lol import calculate_lpips_lol

__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe', 'calculate_psnr_pytorch', \
           'calculate_ssim_pytorch', 'calculate_ssim_lol', 'calculate_lpips_lol']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
