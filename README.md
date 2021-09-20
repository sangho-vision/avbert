# Parameter Efficient Multimodal Transformers for Video Representation Learning

This repository contains the code and models for our ICLR 2021 paper:

**Parameter Efficient Multimodal Transformers for Video Representation Learning** <br>
Sangho Lee, Youngjae Yu, Gunhee Kim, Thomas Breuel, Jan Kautz, Yale Song <br>
[[paper]](https://openreview.net/pdf?id=6UdQLhqJyFD) [[poster]](https://sangho-vision.github.io/assets/poster/iclr2021_lee_poster.png) [[slides]](https://sangho-vision.github.io/assets/slides/iclr2021_lee_slides.pdf)

```bibtex
@inproceedings{lee2021avbert,
    title="{Parameter Efficient Multimodal Transformers for Video Representation Learning}",
    author={Sangho Lee and Youngjae Yu and Gunhee Kim and Thomas Breuel and Jan Kautz and Yale Song},
    booktitle={ICLR},
    year=2021
}
```


## System Requirements
- Python >= 3.7.6
- FFMpeg 4.3.1
- CUDA >= 10.1 supported GPUs with at least 24GB memory

## Installation
1. Install PyTorch 1.6.0, torchvision 0.7.0 and torchaudio 0.6.0 for your environment.
Follow the instructions in
[HERE](https://pytorch.org/get-started/previous-versions/).

2. Install other required packages.
```bash
pip install -r requirements.txt
```


## Download Data
```bash
python download_ucf101.py
python download_esc50.py
python download_ks.py
python download_checkpoint.py
```

## Experiments

To run experiments with a single GPU.

UCF101 (split: 1, 2 or 3)
```bash
cd code
python run_net.py \
    --cfg_file configs/ucf101/config.yaml \
    --configuration ucf101 \
    --pretrain_checkpoint_path checkpoints/checkpoint.pyth \
    TRAIN.DATASET_SPLIT <split>
    TEST.DATASET_SPLIT <split>
```

ESC-50 (split: 1, 2, 3, 4 or 5)
```bash
cd code
python run_net.py \
    --cfg_file configs/esc50/config.yaml \
    --configuration esc50 \
    --pretrain_checkpoint_path checkpoints/checkpoint.pyth \
    TRAIN.DATASET_SPLIT <split>
    TEST.DATASET_SPLIT <split>
```

Kinetics-Sounds
```bash
cd code
python run_net.py \
    --cfg_file configs/kinetics-sounds/config.yaml \
    --configuration kinetics-sounds \
    --pretrain_checkpoint_path checkpoints/checkpoint.pyth
```

After submission, we further adjusted hyperparameters and achieved the following results.

| Dataset | Top-1 Accuracy | Top-5 Accuracy |
| ------- | -------------- | -------------- |
| UCF101 | 87.5 | 97.4 |
| ESC-50 | 85.9 | 96.9 |
| Kinetis-Sounds | 85.8 | 97.8 |

## Acknowledgments
This source code is based on [PySlowFast](https://github.com/facebookresearch/SlowFast).
