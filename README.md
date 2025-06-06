# Tackling Feature and Sample Heterogeneity in Decentralized Multi-Task Learning: A Sheaf-Theoretic Approach

This repository is the official implementation of [Tackling Feature and Sample Heterogeneity in Decentralized Multi-Task Learning: A Sheaf-Theoretic Approach](https://arxiv.org/abs/xxxx.xxxxx).

![Sheaf-FMTL Framework](docs/sheaf_framework.png)

## Abstract

We introduce a novel sheaf-theoretic approach for federated multi-task learning (FMTL) that addresses challenges arising from feature and sample heterogeneity across clients. By representing client relationships using cellular sheaves, our framework flexibly models interactions between heterogeneous client models. Our algorithm, Sheaf-FMTL, achieves substantial communication savings while maintaining competitive performance compared to state-of-the-art decentralized FMTL baselines.

## Requirements

To install requirements:

```bash
pip install -r requirements.txt
```

## Datasets

The experiments use six datasets:

- **Rotated MNIST (R-MNIST):** MNIST with different rotation angles across clients
- **Heterogeneous CIFAR-10 (H-CIFAR-10):** CIFAR-10 with heterogeneous label distribution
- **Human Activity Recognition (HAR):** Sensor data from 30 individuals
- **Vehicle Sensor:** Acoustic and seismic data from 23 sensors
- **GLEAM:** Google Glass sensor data from 38 individuals
- **School:** Exam results prediction for 139 schools
 
R-MNIST and H-CIFAR-10 datasets will be automatically downloaded when running experiments. The rest of the datasets can be found in the datasets folder.

## Training

To train Sheaf-FMTL on a specific dataset:

# Rotated MNIST
python experiments/run_rotated_mnist.py --lambda_reg 0.001 --alpha 0.0005 --eta 0.00001 --gamma 0.01 --num_rounds 200

# Heterogeneous CIFAR-10
python experiments/run_heterogeneous_cifar10.py --lambda_reg 0.001 --alpha 0.005 --eta 0.01 --gamma 0.01 --num_rounds 150

## Key Parameters
