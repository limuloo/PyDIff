'''
ArcFace MS1MV3 r50
https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
'''
import math
import os.path as osp
import torch
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.losses.losses import r1_penalty
from basicsr.metrics import calculate_metric
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from torch.nn import functional as F
from torchvision.ops import roi_align
from tqdm import tqdm
import sys
sys.path.append('.')
import numpy as np
import cv2
import torch.nn as nn
cv2.setNumThreads(1)
import torchvision
from torch.nn.parallel import DataParallel, DistributedDataParallel
from scripts.utils import pad_tensor_back

@MODEL_REGISTRY.register()
class PyDiffModel(BaseModel):

    def __init__(self, opt):
        super(PyDiffModel, self).__init__(opt)

        # define u-net network
        self.unet = build_network(opt['network_unet'])
        self.unet = self.model_to_device(self.unet)
        opt['network_ddpm']['denoise_fn'] = self.unet

        self.global_corrector = build_network(opt['network_global_corrector'])
        self.global_corrector = self.model_to_device(self.global_corrector)
        opt['network_ddpm']['color_fn'] = self.global_corrector

        self.ddpm = build_network(opt['network_ddpm'])
        self.ddpm = self.model_to_device(self.ddpm)
        if isinstance(self.ddpm, (DataParallel, DistributedDataParallel)):
            self.bare_model = self.ddpm.module
        else:
            self.bare_model = self.ddpm

        self.bare_model.set_new_noise_schedule(schedule_opt=opt['ddpm_schedule'],
                                            device=self.device)
        self.bare_model.set_loss(device=self.device)
        self.print_network(self.ddpm)

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.ddpm, load_path, self.opt['path'].get('strict_load_g', True), param_key)
        if 'metrics' in self.opt['val'] and 'lpips' in self.opt['val']['metrics']:
            import lpips
            self.lpips = lpips.LPIPS(net='alex')
            self.lpips = self.model_to_device(self.lpips)
            if isinstance(self.lpips, (DataParallel, DistributedDataParallel)):
                self.lpips_bare_model = self.lpips.module
            else:
                self.lpips_bare_model = self.lpips

    def feed_data(self, data):
        self.LR = data['LR'].to(self.device)
        self.HR = data['HR'].to(self.device)
        if 'pad_left' in data:
            self.pad_left = data['pad_left'].to(self.device)
            self.pad_right = data['pad_right'].to(self.device)
            self.pad_top = data['pad_top'].to(self.device)
            self.pad_bottom = data['pad_bottom'].to(self.device)

    def optimize_parameters(self, current_iter):
        pass

    def test(self):
        if self.opt['val'].get('test_speed', False):
            assert self.opt['val'].get('ddim_pyramid', False), "please use ddim_pyramid"
            with torch.no_grad():
                iterations = self.opt['val'].get('iterations', 100)
                input_size = self.opt['val'].get('input_size', [400, 600])

                LR = torch.randn(1, 10, input_size[0], input_size[1]).to(self.device)
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                self.bare_model.denoise_fn.eval()
                
                # GPU warm up
                print('GPU warm up')
                for _ in tqdm(range(50)):
                    self.output = self.bare_model.ddim_pyramid_sample(LR, 
                                                    pyramid_list=self.opt['val'].get('pyramid_list'),
                                                    continous=self.opt['val'].get('ret_process', False), 
                                                    ddim_timesteps = self.opt['val'].get('ddim_timesteps', 50),
                                                    return_pred_noise=self.opt['val'].get('return_pred_noise', False),
                                                    return_x_recon=self.opt['val'].get('ret_x_recon', False),
                                                    ddim_discr_method=self.opt['val'].get('ddim_discr_method', 'uniform'),
                                                    ddim_eta=self.opt['val'].get('ddim_eta', 0.0),
                                                    pred_type=self.opt['val'].get('pred_type', 'noise'),
                                                    clip_noise=self.opt['val'].get('clip_noise', False),
                                                    save_noise=self.opt['val'].get('save_noise', False),
                                                    color_gamma=self.opt['val'].get('color_gamma', None),
                                                    color_times=self.opt['val'].get('color_times', 1),
                                                    return_all=self.opt['val'].get('ret_all', False))
                
                # speed test
                times = torch.zeros(iterations)     # Store the time of each iteration
                for iter in tqdm(range(iterations)):
                    starter.record()
                    self.output = self.bare_model.ddim_pyramid_sample(LR, 
                                                    pyramid_list=self.opt['val'].get('pyramid_list'),
                                                    continous=self.opt['val'].get('ret_process', False), 
                                                    ddim_timesteps = self.opt['val'].get('ddim_timesteps', 50),
                                                    return_pred_noise=self.opt['val'].get('return_pred_noise', False),
                                                    return_x_recon=self.opt['val'].get('ret_x_recon', False),
                                                    ddim_discr_method=self.opt['val'].get('ddim_discr_method', 'uniform'),
                                                    ddim_eta=self.opt['val'].get('ddim_eta', 0.0),
                                                    pred_type=self.opt['val'].get('pred_type', 'noise'),
                                                    clip_noise=self.opt['val'].get('clip_noise', False),
                                                    save_noise=self.opt['val'].get('save_noise', False),
                                                    color_gamma=self.opt['val'].get('color_gamma', None),
                                                    color_times=self.opt['val'].get('color_times', 1),
                                                    return_all=self.opt['val'].get('ret_all', False))
                    ender.record()
                    # Synchronize GPU
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)
                    times[iter] = curr_time
                    # print(curr_time)

                mean_time = times.mean().item()
                logger = get_root_logger()
                logger.info("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))
                import sys
                sys.exit()
        with torch.no_grad():
            self.bare_model.denoise_fn.eval()
            self.output = self.bare_model.ddim_pyramid_sample(self.LR, 
                                                pyramid_list=self.opt['val'].get('pyramid_list'),
                                                continous=self.opt['val'].get('ret_process', False), 
                                                ddim_timesteps = self.opt['val'].get('ddim_timesteps', 50),
                                                return_pred_noise=self.opt['val'].get('return_pred_noise', False),
                                                return_x_recon=self.opt['val'].get('ret_x_recon', False),
                                                ddim_discr_method=self.opt['val'].get('ddim_discr_method', 'uniform'),
                                                ddim_eta=self.opt['val'].get('ddim_eta', 0.0),
                                                pred_type=self.opt['val'].get('pred_type', 'noise'),
                                                clip_noise=self.opt['val'].get('clip_noise', False),
                                                save_noise=self.opt['val'].get('save_noise', False),
                                                color_gamma=self.opt['val'].get('color_gamma', None),
                                                color_times=self.opt['val'].get('color_times', 1),
                                                return_all=self.opt['val'].get('ret_all', False),
                                                fine_diffV2=self.opt['val'].get('fine_diffV2', False),
                                                fine_diffV2_st=self.opt['val'].get('fine_diffV2_st', 200),
                                                fine_diffV2_num_timesteps=self.opt['val'].get('fine_diffV2_num_timesteps', 20),
                                                do_some_global_deg=self.opt['val'].get('do_some_global_deg', False),
                                                use_up_v2=self.opt['val'].get('use_up_v2', False))
            self.bare_model.denoise_fn.train()
            
            if hasattr(self, 'pad_left') and not self.opt['val'].get('ret_process', False):
                self.output = pad_tensor_back(self.output, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
                self.LR = pad_tensor_back(self.LR, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
                self.HR = pad_tensor_back(self.HR, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def find_lol_dataset(self, name):
        if name[0] == 'r':
            return 'SYNC'
        elif name[0] == 'n' or name[0] == 'l':
            return 'REAL'
        else:
            return 'LOL'

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if self.opt['val'].get('fix_seed', False):
            next_seed = np.random.randint(10000000)
            logger = get_root_logger()
            logger.info(f'next_seed={next_seed}')
        if self.opt['val'].get('ret_process', False):
            with_metrics = False
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        metric_data = dict()
        metric_data_pytorch = dict()
        pbar = tqdm(total=len(dataloader), unit='image')
        if self.opt['val'].get('split_log', False):
            self.split_results = {}
            self.split_results['SYNC'] = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self.split_results['REAL'] = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self.split_results['LOL'] = {metric: 0 for metric in self.opt['val']['metrics'].keys()}

        for idx, val_data in enumerate(dataloader):
            if self.opt['val'].get('fix_seed', False):
                from basicsr.utils import set_random_seed
                set_random_seed(0)
            if not self.opt['val'].get('cal_all', False) and \
               not self.opt['val'].get('cal_score', False) and \
               int(self.opt['ddpm_schedule']['n_timestep']) >= 4 and idx >= 3:
                break
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['sr']], min_max=(-1, 1))
            gt_img = tensor2img([visuals['gt']], min_max=(-1, 1))
            lq_img = tensor2img([visuals['lq']], min_max=(-1, 1))
            if self.opt['val'].get('use_kind_align', False):
                '''
                References:
                https://github.com/zhangyhuaee/KinD/blob/master/evaluate_LOLdataset.py
                https://github.com/wyf0912/LLFlow/blob/main/code/test.py
                '''
                gt_mean = np.mean(gt_img)
                sr_mean = np.mean(sr_img)
                sr_img = sr_img * gt_mean / sr_mean
                sr_img = np.clip(sr_img, 0, 255)
                sr_img = sr_img.astype('uint8')

            metric_data['img'] = sr_img
            metric_data['img2'] = gt_img
            metric_data_pytorch['img'] = self.output
            metric_data_pytorch['img2'] = self.HR
            path = val_data['lq_path'][0]
            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                # print(save_img_path)
                if idx < self.opt['val'].get('show_num', 3) or self.opt['val'].get('show_all', False):
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                            f'{img_name}_{current_iter}.png')
                    if not self.opt['val'].get('ret_process', False):
                        if self.opt['val'].get('only_save_sr', False):
                            save_img_path = osp.join(self.opt['path']['visualization'],
                                            f'{img_name}.png')
                            imwrite(sr_img, save_img_path)
                        else:
                            imwrite(np.concatenate([lq_img, sr_img, gt_img], axis=1), save_img_path)
                    else:
                        imwrite(sr_img, save_img_path)
            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if 'lpips' in opt_['type']:
                        opt_['device'] = self.device
                        opt_['model'] = self.lpips_bare_model
                    if 'pytorch' in opt_['type']:
                        if self.opt['val'].get('split_log', False):
                            self.split_results[self.find_lol_dataset(img_name)][name] += calculate_metric(metric_data_pytorch, opt_).item()
                        self.metric_results[name] += calculate_metric(metric_data_pytorch, opt_).item()
                    else:
                        if self.opt['val'].get('split_log', False):
                            self.split_results[self.find_lol_dataset(img_name)][name] += calculate_metric(metric_data, opt_)
                        self.metric_results[name] += calculate_metric(metric_data, opt_)
            
            # tentative for out of GPU memory
            del self.LR
            del self.output
            torch.cuda.empty_cache()
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
            if self.opt['val'].get('cal_score_num', None):
                if idx >= self.opt['val'].get('cal_score_num', None):
                    break
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        if self.opt['val'].get('cal_score', False):
            import sys
            sys.exit()
        if self.opt['val'].get('fix_seed', False):
            from basicsr.utils import set_random_seed
            set_random_seed(next_seed)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        logger = get_root_logger()
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger.info(log_str)
        if self.opt['val'].get('split_log', False):
            for dataset_name, num in zip(['LOL', 'REAL', 'SYNC'], [15, 100, 100]):
                log_str = f'Validation {dataset_name}\n'
                for metric, value in self.split_results[dataset_name].items():
                    log_str += f'\t # {metric}: {value/num:.4f}\n'
                logger.info(log_str)
        
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        if self.LR.shape != self.output.shape:
            self.LR = F.interpolate(self.LR, self.output.shape[2:])
            self.HR = F.interpolate(self.HR, self.output.shape[2:])
        out_dict['gt'] = self.HR.detach().cpu()
        out_dict['sr'] = self.output.detach().cpu()
        out_dict['lq'] = self.LR[:, :3, :, :].detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network([self.ddpm], 'net_g', current_iter, param_key=['params'])
