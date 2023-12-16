[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pyramid-diffusion-models-for-low-light-image/low-light-image-enhancement-on-lol)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lol?p=pyramid-diffusion-models-for-low-light-image)

# [IJCAI 2023 oral] Pyramid Diffusion Models For Low-light Image Enhancement
### [Paper](https://arxiv.org/pdf/2305.10028.pdf) | [Project Page](https://github.com/limuloo/PyDIff) | [Supplement Materials](https://drive.google.com/file/d/1_c5nM_bQkdDMWASpY-3aoxf_YYzfWCwf/view)
**Pyramid Diffusion Models For Low-light Image Enhancement**
<br>_Dewei Zhou, Zongxin Yang, Yi Yang_<br>
In IJCAI'2023
## Overall
![Framework](images/framework.png)

## Quantitative results
### Evaluation on LOL
The evauluation results on LOL are as follows
| Method | PSNR | SSIM | LPIPS |
| :-- | :--: | :--: | :--: |
| KIND | 20.87 | 0.80 | 0.17 |
| KIND++ | 21.30 | 0.82 | 0.16 |
| Bread | 22.96 | 0.84 | 0.16 |
| IAT | 23.38 | 0.81 | 0.26 | 
| HWMNet | 24.24 | 0.85 | 0.12 |
| LLFLOW | 24.99 | 0.92 | 0.11 |
| **PyDiff (Ours)** | **27.09** | **0.93** | **0.10** |

## Dependencies and Installation
```bash
git clone https://github.com/limuloo/PyDIff.git
cd PyDiff
conda create -n PyDiff python=3.7
conda activate PyDiff
conda install pytorch==1.7.0 torchvision torchaudio cudatoolkit=11.0 -c pytorch
cd BasicSR-light
pip install -r requirements.txt
BASICSR_EXT=True sudo $(which python) setup.py develop
cd ../PyDiff
pip install -r requirements.txt
BASICSR_EXT=True sudo $(which python) setup.py develop
```

## Dataset
You can refer to the following links to download the [LOL](https://daooshee.github.io/BMVC2018website/) dataset and put it in the following way:
```bash
PyDiff/
    BasicSR-light/
    PyDiff/
    dataset/
        LOLdataset/
            our485/
            eval15/
```

## Pretrained Model
You can refer to the following links to download the [pretrained model](https://drive.google.com/file/d/1-WScg2H0jwzVvdbw2HrXxLjs4We5A_SI/view) and put it in the following way:
```bash
PyDiff/
    BasicSR-light/
    PyDiff/
    pretrained_models/
        LOLweights.pth
```

## Test
```bash
cd PyDiff/
CUDA_VISIBLE_DEVICES=0 python pydiff/train.py -opt options/infer.yaml
```
**NOTE: When testing on your own dataset, set 'use_kind_align' in 'infer.yaml' to false.** For details, please refer to https://github.com/limuloo/PyDIff/issues/6.

## Train
### Training with 2 GPUs 

For training purposes, the utilization of the following commands is advised if you possess 2 GPUs with a memory capacity of 24GB or higher, as outlined in the paper. 

```bash
cd PyDiff/
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=22666 pydiff/train.py -opt options/train_v1.yaml --launcher pytorch
```

### Training with a single GPU 

Otherwise, you can use the following commands for training, which requires 1 GPU with memory >=24GB.

```bash
cd PyDiff/
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=22666 pydiff/train.py -opt options/train_v2.yaml --launcher pytorch
```

### Training on a Custom Low-Level Task Dataset

Please update the following fields in the `PyDiff/options/train_v3.yaml` file: `YOUR_TRAIN_DATASET_GT_ROOT`, `YOUR_TRAIN_DATASET_INPUT_ROOT`, `YOUR_EVAL_DATASET_GT_ROOT`, and `YOUR_EVAL_DATASET_INPUT_ROOT`. If required, please also update the `PyDiff/pydiff/data/lol_dataset.py`. Finally, please employ the subsequent commands to initiate the training process:

```bash
cd PyDiff/
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=22666 pydiff/train.py -opt options/train_v3.yaml --launcher pytorch
```

Please feel free to customize additional parameters to meet your specific requirements. To enable PyDiff to train on a single GPU, the `PyDiff/options/train_v2.yaml` file can be consulted.

## Citation
If you find our work useful for your research, please cite our paper
```
@article{zhou2023pyramid,
  title={Pyramid Diffusion Models For Low-light Image Enhancement},
  author={Zhou, Dewei and Yang, Zongxin and Yang, Yi},
  journal={arXiv preprint arXiv:2305.10028},
  year={2023}
}
```

## Acknowledgement
Our code is partly built upon [BasicSR](https://github.com/XPixelGroup/BasicSR). Thanks to the contributors of their great work.
