# Enhanced Hyperspectral Image Classification via Spectral-Spatial Fusion of Swin Transformer and Mamba Network

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxxxx) 

## Introduction
This repository contains the official PyTorch implementation of **S2F2STM**. 

Hyperspectral image (HSI) classification is pivotal in remote sensing applications, including land cover classification and environmental monitoring. Traditional methods face challenges due to high-dimensional data and complex spatial distributions. This paper introduces S2F2STM, a novel model integrating the Swin Transformer and an enhanced Mamba module for efficient spectral-spatial feature fusion. The Swin Transformer captures spatial features with reduced computational costs, while the Mamba module extracts long-range spectral dependencies. Experimental results across multiple datasets demonstrate S2F2STM's superior performance, achieving an overall accuracy (OA) exceeding 86% and outperforming existing methods in classification accuracy and computational efficiency.

## Repository Structure
Based on the modular design of S2F2STM, the repository is organized as follows:
* `model/`: Contains the core implementations of our proposed algorithms, including the Swin Transformer spatial feature extractor, the Mamba network for spectral dependencies, and the spectral-spatial fusion mechanisms.
* `utils/`: Includes utility scripts for dataset loading, preprocessing, dimensionality reduction (e.g., PCA), and evaluation metrics calculation.
* `train_SwinMamba.py`: The main execution script used to train the S2F2STM model, validate its performance, and output classification results.

## Prerequisites and Dependencies
To replicate the experiments, please ensure your environment meets the following requirements. 
* Python 3.8+
* PyTorch 1.10+ (CUDA enabled)
* `mamba-ssm` and `causal-conv1d` (for the Mamba network)
* `scikit-learn`, `numpy`, `scipy`, `matplotlib`

## Usage Guidelines
**1. Data Preparation**
Download the required hyperspectral datasets (e.g., Houston2013, Houston2018, Augsburg, etc.) and place them in the designated `data/` directory (you may need to create this folder).

**2. Training and Evaluation**
To train the S2F2STM model and evaluate its performance from scratch, run the following command:
```bash
python train_SwinMamba.py
