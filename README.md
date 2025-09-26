# Towards generalizable deep ptychography neural networks

This repository contains the codebase for the workflow and model in the paper "[Towards generalizable deep ptychography neural networks]()". Note that this is a snapshot of the repository at the time of publication submission. The up-to-date repository can be found [here](https://github.com/hoidn/PtychoPINN).

## Overview
PtychoPINN-torch is an unsupervised physics-informed neural network reconstruction method for scanning transmission ptychography. This library is a PyTorch implementation based on PtychoPINN. There are small differences in architecture/inductive biases that are included in the manuscript.

This library contains sufficient tools to re-create all results presented in the manuscript. This includes training and inference scripts whose instructions are posted below. Artifacts and data can be found at the following "[zenodo link](https://doi.org/10.5281/zenodo.16968020)". This package supports training with both experimental and synthetic data.

Note that having GPU access is highly recommended for both training and inference due to the large image tensor sizes present.


## Features
- **Unsupervised / self-supervised learning**: There is no need for extensive labeled training data, making the model more practical to train on experiments
- **Resolution**: PtychoPINN outperforms existing deep learning models for ptychographic reconstruction in terms of image quality, with a 10 dB PSNR increase and a 3- to 6-fold gain in linear resolution. Generalizability and robustness are also improved.
- **Scalability and Speed**: PtychoPINN is two or three orders of magnitude as fast as iterative ptychography algorithms
- **Multi-experiment loading**: PtychoPINN-torch can load an arbitrary number of experiments due to memory-mapped dataloading via a custom dataloader. 
- **Re-usability**: Reusable/repeatable training experiments thanks to MLFlow and saving all data/model related artifacts.


## Installation
Download `data.tar.gz` and `mlruns.tar.gz` from [zenodo link](https://doi.org/10.5281/zenodo.16968020)

Install conda: https://conda.io/miniconda.html

```
conda install mamba -c conda-forge
mamba env create -f environment.yml
conda activate ptychopinn_torch
tar -xzf data.tar.gz -xzf mlruns.tar.gz
python initialize_data.py --no_dry_run
```

## Usage
```
Training
$ ptychopinn_torch/train_full.py
usage: [--ptycho_dir ][--config] [--mode]

Inference
$ ptychopinn_torch/inference.py
usage: [--run_id] [--infer_dir] [--file_index] [--config_override]
```

Examples:

Note that all demo configuration files included have a default number of devices at 1. Increase this if you want faster training with multiple gpus.

```
Training
$ ptychopinn_torch/train_full.py --ptycho_dir data/TP2 --config ptychopinn_torch/configs/velociprobe_config.json --mode synth

Inference
$ ptychopinn_torch/inference.py --run_id 6fb4668f21e44e0b80056f64fdfedf01 --infer_dir data/TP2 --config ptychopinn_torch/configs/hxn_demo_config.json
```

For interactive usage, see `notebooks/ptycho_lines.ipynb` and `notebooks/non_grid_CDI_example.ipynb`. These demonstrate reconstruction with scanning CDI + grid scan pattern + simulated data and fresnel CDI + random scan pattern + experimental data, respectively.

## Interactive Notebooks

Use `recreate_results.ipynb` to use any of the models described in the paper to infer on the datasets provided. The models are provided as unique MLFlow run ids which can be referenced through a dictionary. 
It also provides a small demo showcasing creation of the synthetic objects used to train the model.




