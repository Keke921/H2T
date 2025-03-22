<h1 align="center"> Feature Fusion from Head to Tail for Long-Tailed Visual Recognition </h1>
<p align="center">
    <a href="https://arxiv.org/abs/2306.06963"><img src="https://img.shields.io/badge/arXiv-2306.06963-b31b1b.svg" alt="Paper"></a>
    <a href="https://vcc.tech/research/2024/H2T"><img alt="overview" src="https://img.shields.io/static/v1?label=overview&message=VCC%20Project&color=blue"></a>
    <!-- <a href="https://github.com/Keke921/H2T"><img src="https://img.shields.io/badge/-Github-grey?logo=github" alt="Github"></a> -->
    <!-- <a href=""><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"></a> -->
    <!-- <a href="https://openreview.net/forum?id=xxx"> <img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=AAAI%2723&color=blue"></a> -->
    <a href="https://github.com/Keke921/H2T/blob/main/LICENSE"> <img alt="License" src="https://img.shields.io/github/license/LFhase/PAIR?color=blue"></a>
    <!-- <a href="https://nips.cc/virtual/2023/poster/70939"> <img src="https://img.shields.io/badge/Video-grey?logo=Kuaishou&logoColor=white" alt="Video"></a> -->
    <a href="https://github.com/Keke921/H2T/blob/main/slides%20and%20poster/AAAI24-H2T-slides_422.pptx"> <img src="https://img.shields.io/badge/Slides-grey?&logo=MicrosoftPowerPoint&logoColor=white" alt="Slides"></a>
    <a href="https://github.com/Keke921/H2T/blob/main/slides%20and%20poster/AAAI24-H2T-slides_422.pdf"> <img src="https://img.shields.io/badge/Slides-grey?logo=airplayvideo&logoColor=white" alt="Slides"></a>
    <a href="https://github.com/Keke921/H2T/blob/main/slides%20and%20poster/AAAI24_H2T-poster_422.pdf"> <img src="https://img.shields.io/badge/Poster-grey?logo=airplayvideo&logoColor=white" alt="Poster"></a>
</p>

This repo contains the sample code for our AAAI 2024: *[Feature Fusion from Head to Tail for Long-Tailed Visual Recognition](https://ojs.aaai.org/index.php/AAAI/article/view/29262)*.
The core code is in methods.py: H2T.

## To do list:
- [x] Camera-ready version including the appendix of the paper is updated ! [[link](https://arxiv.org/abs/2306.06963)]
- [x] Slides and the poster are released. [[Slides (pptx)](https://github.com/Keke921/H2T/blob/main/slides%20and%20poster/AAAI24-H2T-slides_422.pptx), [Slides (pdf)](https://github.com/Keke921/H2T/blob/main/slides%20and%20poster/AAAI24-H2T-slides_422.pdf), [Poster](https://github.com/Keke921/H2T/blob/main/slides%20and%20poster/AAAI24_H2T-poster_422.pdf)]
- [x] CE loss for CIFAR-100-LT is realsed.
- [ ] Code for other datasets and baseline methods are some what messy 😆😆😆. Detailed running instructions and the orignized code for more datasets and baselines will be released latter. (This repository reserves some interfaces for other loss functions and backbones, which have not yet been integrated into the training and configuration files.)


## Training

**Stage-1**:

(e.g. CIFAR100-LT, imbalance ratio = 100, CrossEntropy Loss, MixUp, training from scratch)

```
python train_stage1.py --cfg ./config/cifar100_imb001_stage1_ce_mixup
```


**Stage-2**:

(e.g. CIFAR100-LT, imbalance ratio = 100, CrossEntropy Loss, H2T)

```
python train_stage2.py --cfg ./config/cifar100_imb001_stage2_ce_H2T.yaml resume /path/to/checkpoint/stage1
```

The saved folder (including logs, code, and checkpoints) is organized as follows.
```
H2T
├── saved
│   ├── modelname_date
│   │   ├── ckps
│   │   │   ├── current.pth.tar
│   │   │   └── model_best.pth.tar
│   │   └── logs
│   │       └── modelname.txt
│   │   └── codes
│   │       └── relevant code without data
│   ...   
```

## Evaluation

To evaluate a trained model, run:

(e.g. CIFAR100-LT, imbalance ratio = 100, CrossEntropy Loss, Stage-1)

```
python eval-modified.py --cfg ./config/cifar100_imb001_stage1_ce_mixup resume /path/to/checkpoint/stage1
```


(e.g. CIFAR100-LT, imbalance ratio = 100, CrossEntropy Loss, Stage-2)

```
python eval.py --cfg ./config/cifar100_imb001_stage2_ce_H2T.yaml resume /path/to/checkpoint/stage2
```


## Results and Models

**1) CIFAR-10-LT and CIFAR-100-LT**

* Stage-1 (*CE with mixup*):

| Dataset              | Top-1 Accuracy | Model |
| -------------------- | -------------- | ----- |
| CIFAR-100-LT IF=50   | 45.40%         | [link](https://www.dropbox.com/scl/fi/dc673e7vgz6rpv3nbdxsu/cifar100_imb001_stage1.pth.tar?rlkey=64v00anjp9udtceij6tgl7ni7&dl=0)  |
| CIFAR-100-LT IF=100  | 39.55%         | [link](https://www.dropbox.com/scl/fi/dc673e7vgz6rpv3nbdxsu/cifar100_imb001_stage1.pth.tar?rlkey=64v00anjp9udtceij6tgl7ni7&dl=0)  |
| CIFAR-100-LT IF=200  | 36.01%         | [link](https://www.dropbox.com/scl/fi/498bvi7zpmi69j301dd4r/cifar100_imb0005_stage1.pth.tar?rlkey=lt8tzpxcje3j52bafgqxr91sm&dl=0)  |

* Stage-2 (*CE with H2T*):

| Dataset              | Top-1 Accuracy  | Model |
| -------------------- | --------------  | ----- |
| CIFAR-100-LT IF=50   | 52.95%           | [link](https://www.dropbox.com/scl/fi/ssucewnxfr3dvxmudgud0/cifar100_imb002_stage2.pth.tar?rlkey=xxj7jijsquix4zf9xl45woxkx&dl=0)  |
| CIFAR-100-LT IF=100  | 47.80%           | [link](https://www.dropbox.com/scl/fi/uhrpw32b3clbll23no6l7/cifar100_imb001_stage2.pth.tar?rlkey=hl5bsyxov1sybd6pxmd5gdavb&dl=0)  |
| CIFAR-100-LT IF=200  | 43.95%           | [link](https://www.dropbox.com/scl/fi/tar8641c5pmpywogvx9xr/cifar100_imb0005_stage2.pth.tar?rlkey=nkvakl2q1h2ur5v3b57ldtsv9&dl=0)  |

*Note: I reran Stage-2 with the [config](https://github.com/Keke921/H2T/tree/main/config/cifar100) from this respository and got slightly better results than in the AAAI paper.*

## You May Find Our Additional Works of Interest

* [CVPR'22] Long-tailed visual recognition via Gaussian clouded logit adjustment [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Long-Tailed_Visual_Recognition_via_Gaussian_Clouded_Logit_Adjustment_CVPR_2022_paper.pdf)] [[code](https://github.com/Keke921/GCLLoss)]

* [TPAMI'23] Key point sensitive loss for long-tailed visual recognition [[paper](https://drive.google.com/file/d/1gOJDHBJ_M7RmU6Iw2p6uXIyo8pNgVMrv/view?pli=1)] [[code](https://github.com/Keke921/KPSLoss)]

* [CVPR'23] Long-tailed visual recognition via self-heterogeneous integration with knowledge excavation [[paper](https://arxiv.org/pdf/2304.01279)] [[code](https://github.com/jinyan-06/SHIKE)]

* [NeurIPS'24] Improving Visual Prompt Tuning by Gaussian Neighborhood Minimization for Long-Tailed Visual Recognition [[paper](https://arxiv.org/pdf/2410.21042)] [[code](https://github.com/Keke921/GNM-PT)]

* [TAI'24] Adjusting logit in Gaussian form for long-tailed visual recognition [[paper](https://arxiv.org/pdf/2305.10648)] [[code](https://github.com/Keke921/GCLLoss)]



## Citation

If you find our paper and repo useful, please cite our paper:
```bibtex
@inproceedings{li2024feature,
  title={Feature Fusion from Head to Tail for Long-Tailed Visual Recognition},
  author={Li, Mengke and Zhikai, HU and Lu, Yang and Lan, Weichao and Cheung, Yiu-ming and Huang, Hui},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={12},
  pages={13581--13589},
  year={2024}
}
```
## Acknowledgment
We refer to the code architecture from [MisLAS](https://github.com/dvlab-research/MiSLAS). Many thanks to the authors.
