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

This repo contains the sample code for our AAAI 2024: *[Feature Fusion from Head to Tail for Long-Tailed Visual Recognition](https://arxiv.org/abs/2306.06963)*.
The core code is in methods.py: H2T.

## To do list:
- [x] Camera-ready version including the appendix of the paper is updated ! [[link](https://arxiv.org/abs/2306.06963)]
- [x] Slides and the poster are released. [[Slides (pptx)](https://github.com/Keke921/H2T/blob/main/slides%20and%20poster/AAAI24-H2T-slides_422.pptx), [Slides (pdf)](https://github.com/Keke921/H2T/blob/main/slides%20and%20poster/AAAI24-H2T-slides_422.pdf), [Poster](https://github.com/Keke921/H2T/blob/main/slides%20and%20poster/AAAI24_H2T-poster_422.pdf)]
- [x] CE loss for CIFAR-100-LT is realsed.
- [ ] Code for ther datasets and baseline methods are some what messy 😆😆😆. Detailed running instructions and the orignized code for more will be released. 


## Results and Models

**1) CIFAR-10-LT and CIFAR-100-LT**

* Stage-1 (*mixup*):

| Dataset              | Top-1 Accuracy | Model |
| -------------------- | -------------- | ----- |
| CIFAR-100-LT IF=50   | 45.40%         | [link](https://www.dropbox.com/scl/fi/dc673e7vgz6rpv3nbdxsu/cifar100_imb001_stage1.pth.tar?rlkey=64v00anjp9udtceij6tgl7ni7&dl=0)  |
| CIFAR-100-LT IF=100  | 39.55%         | [link](https://www.dropbox.com/scl/fi/dc673e7vgz6rpv3nbdxsu/cifar100_imb001_stage1.pth.tar?rlkey=64v00anjp9udtceij6tgl7ni7&dl=0)  |
| CIFAR-100-LT IF=200  | 36.01%         | [link](https://www.dropbox.com/scl/fi/498bvi7zpmi69j301dd4r/cifar100_imb0005_stage1.pth.tar?rlkey=lt8tzpxcje3j52bafgqxr91sm&dl=0)  |

* Stage-2 (*MiSLAS*):

| Dataset              | Top-1 Accuracy  | Model |
| -------------------- | --------------  | ----- |
| CIFAR-100-LT IF=50   | 52.95%           | [link](https://www.dropbox.com/scl/fi/ssucewnxfr3dvxmudgud0/cifar100_imb002_stage2.pth.tar?rlkey=xxj7jijsquix4zf9xl45woxkx&dl=0)  |
| CIFAR-100-LT IF=100  | 47.80%           | [link](https://www.dropbox.com/scl/fi/uhrpw32b3clbll23no6l7/cifar100_imb001_stage2.pth.tar?rlkey=hl5bsyxov1sybd6pxmd5gdavb&dl=0)  |
| CIFAR-100-LT IF=200  | 43.95%           | [link](https://www.dropbox.com/scl/fi/tar8641c5pmpywogvx9xr/cifar100_imb0005_stage2.pth.tar?rlkey=nkvakl2q1h2ur5v3b57ldtsv9&dl=0)  |

*Note: To obtain better performance, we highly recommend changing the weight decay 2e-4 to 5e-4 on CIFAR-LT.*


## Misc

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
