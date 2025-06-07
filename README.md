# Tackling Feature and Sample Heterogeneity in Decentralized Multi-Task Learning: A Sheaf-Theoretic Approach

This repository is the official implementation of [Tackling Feature and Sample Heterogeneity in Decentralized Multi-Task Learning: A Sheaf-Theoretic Approach](https://arxiv.org/abs/xxxx.xxxxx).

![Sheaf-FMTL Framework](docs/sheaf_framework.png)

## Summarized Abstract

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

```bash
# Rotated MNIST
python experiments/run_rotated_mnist.py --lambda_reg 0.001 --alpha 0.0005 --eta 0.00001 --gamma 0.01 --num_rounds 200

# Heterogeneous CIFAR-10
python experiments/run_heterogeneous_cifar10.py --lambda_reg 0.001 --alpha 0.005 --eta 0.01 --gamma 0.01 --num_rounds 150
```

## Key Parameters

- `--lambda_reg`: Regularization parameter controlling task relationship strength
- `--alpha`: Learning rate for model parameters
- `--eta`: Learning rate for restriction maps
- `--gamma`: Controls the dimension of the interaction space ($d_{ij} = ⌊\gamma d⌋$)
- `--num_rounds`: Number of communication rounds

## Evaluation
To evaluate trained models:

```bash
python evaluate.py --dataset rotated_mnist --model_path results/sheaf_fmtl_rmnist.pth
```

## Results

Our model achieves the following performance:

## Rotated MNIST dataset

| Algorithm | Test Accuracy | Transmitted Bits (MB) | FLOPS (×10¹²) | Memory (MB) |
|-----------|--------------|----------------------|---------------|-------------|
| Sheaf-FMTL (γ = 0.01) | 94.3 ± 0.12 | **38.2** | 240.7 | 7563.5 |
| Sheaf-FMTL (γ = 0.03) | **94.5 ± 0.08** | 165.7 | 249.3 | 24022.2 |
| dFedU | 94.4 ± 0.06 | 3230.9 | 1184.4 | 1582.9 |
| D-PSGD | 92.22 ± 0.06 | 3230.9 | **236.8** | **1438.6** |
| FedAvg | 81.4 ± 0.16 | 27.9 | 1184.1 | 1531.0 |
| DITTO | 92.2 ± 0.11 | 27.9 | 473.8 | 1714.6 |
| pFedMe | 90.05 ± 0.14 | 27.9 | 5923.6 | 1550.7 |
| Per-FedAvg | 88.45 ± 0.11 | 27.9 | 202.0 | 1421.0 |

## Heterogeneous CIFAR-10 dataset

| Algorithm | Test Accuracy | Transmitted Bits (MB) | FLOPS (×10¹²) | Memory (MB) |
|-----------|--------------|----------------------|---------------|-------------|
| Sheaf-FMTL (γ = 0.01) | 65.5 ± 1.2 | **41.4** | 282.2 | 9582.5 |
| Sheaf-FMTL (γ = 0.03) | **65.8 ± 1.3** | 108.8 | 294.5 | 27452.2 |
| dFedU | 65.7 ± 1.8 | 3937.4 | 1376.5 | 9197.6 |
| D-PSGD | 64.0 ± 0.5 | 3937.4 | **207.5** | **8402.4** |
| FedAvg | 60.88 ± 1.4 | 25.5 | 1040.0 | 8431.6 |
| DITTO | 63.4 ± 1.1 | 25.5 | 413.4 | 8650.9 |
| pFedMe | 62.5 ± 1.3 | 25.5 | 6886.2 | 8496.6 |
| Per-FedAvg | 61.25 ± 1.4 | 25.5 | 216.0 | 3480.0 |


## Citation

If you use this code in your research, please cite:


```bash
@article{benissaid2025sheaffmtl,
  title={Tackling Feature and Sample Heterogeneity in Decentralized Multi-Task Learning: A Sheaf-Theoretic Approach},
  author={Ben Issaid, Chaouki and Vepakomma, Praneeth and Bennis, Mehdi},
  journal={Transactions on Machine Learning Research},
  year={2025}
}
```
