# GUARDIAN: Graph-based Universal Adversarial Robustness for Detection of Illicit Accounts in Networks

## Overview

GUARDIAN is a comprehensive framework for detecting illicit accounts in cryptocurrency transaction networks, providing formal guarantees for adversarial robustness and differential privacy while maintaining state-of-the-art detection performance. This implementation accompanies our paper published in Science China Information Sciences (SCIS 2025).

The framework addresses critical security challenges in cryptocurrency fraud detection by integrating certified defense mechanisms with privacy-preserving techniques, achieving 97.06% F1 score on large-scale Bitcoin networks while maintaining inference latency under 458ms for graphs containing up to 100 million nodes.

## Key Features

The GUARDIAN framework provides several distinguishing capabilities that set it apart from existing cryptocurrency fraud detection systems:

**Unified Security Framework**: The system integrates certified adversarial robustness through randomized smoothing with differential privacy guarantees via Rényi accounting, providing formal security properties essential for financial applications.

**Advanced Architecture Components**: The implementation includes temporal edge sequence encoding using graph transformers with 8-head attention mechanisms, enhanced multigraph discrepancy modules preserving behavioral differences through asymmetric message passing, and hierarchical pooling for scalable processing of networks exceeding 10 million nodes.

**Cross-Blockchain Compatibility**: The framework demonstrates effective transfer learning across heterogeneous blockchain protocols, achieving less than 15% performance degradation when transferring models between Bitcoin and Ethereum networks.

**Production-Ready Implementation**: The codebase provides comprehensive configuration management, distributed training support for multi-GPU environments, extensive logging and checkpointing mechanisms, and modular architecture facilitating component modification and extension.

## System Requirements

The implementation requires the following hardware and software configurations for optimal performance:

**Hardware Requirements**:
- GPU: NVIDIA GPU with minimum 24GB memory (RTX 3090, A100, or equivalent)
- RAM: Minimum 64GB system memory (512GB recommended for large-scale experiments)
- Storage: 500GB available disk space for datasets and checkpoints
- Multi-GPU: 4x NVIDIA RTX 3090 or similar for distributed training experiments

**Software Dependencies**:
- Python 3.8 or higher
- PyTorch 1.12.0 with CUDA 11.3 support
- PyTorch Geometric 2.1.0
- CUDA Toolkit 11.3 or higher
- Ubuntu 20.04 LTS (tested) or compatible Linux distribution

## Installation

Begin by cloning the repository and setting up the Python environment:

```bash
git clone https://github.com/anonymous/GUARDIAN-scis.git
cd GUARDIAN-scis
```

Create and activate a conda environment with the required dependencies:

```bash
conda create -n guardian python=3.8
conda activate guardian
```

Install PyTorch with CUDA support appropriate for your system:

```bash
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```

Install PyTorch Geometric and its dependencies:

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric==2.1.0
```

Install additional required packages:

```bash
pip install -r requirements.txt
```

The requirements.txt file includes essential packages such as numpy==1.21.0, scikit-learn==1.0.2, matplotlib==3.5.1, seaborn==0.11.2, tqdm==4.64.0, tensorboard==2.9.1, and pyyaml==6.0.

## Dataset Preparation

GUARDIAN utilizes four large-scale cryptocurrency datasets for comprehensive evaluation. These datasets must be downloaded and preprocessed before running experiments.

Download the datasets from the following sources:
- Ethereum-S: Available at [link to be added upon publication]
- Ethereum-P: Available at [link to be added upon publication]
- Bitcoin-M: Available at [link to be added upon publication]
- Bitcoin-L: Available at [link to be added upon publication]

Place the downloaded datasets in the data directory and run the preprocessing script:

```bash
python scripts/preprocess_data.py --data_dir ./data --output_dir ./processed_data
```

The preprocessing script performs feature extraction, graph construction, temporal ordering, and train-validation-test splitting with stratification to maintain class distributions.

## Usage

### Basic Training

To train GUARDIAN on a specific dataset with default configurations:

```bash
python main.py --config configs/standard_config.json --experiment standard
```

### Adversarial Training

For training with adversarial robustness guarantees:

```bash
python main.py --config configs/adversarial_config.json --experiment adversarial
```

### Privacy-Preserving Training

To enable differential privacy during training:

```bash
python main.py --config configs/privacy_config.json --experiment privacy
```

### Multi-GPU Distributed Training

For distributed training across multiple GPUs:

```bash
python main.py --config configs/multi_gpu_config.json --experiment all --distributed
```

### Configuration Options

The configuration system supports extensive customization through JSON configuration files. Key parameters include:

- Model architecture settings: hidden dimensions, number of layers, attention heads
- Training hyperparameters: learning rate, batch size, number of epochs
- Security parameters: adversarial epsilon, privacy budget, noise multiplier
- System settings: device allocation, distributed training options, logging frequency

Create custom configurations by modifying the template files in the configs directory.

## Reproducing Experimental Results

To reproduce the complete experimental evaluation presented in the paper, execute the comprehensive experimental pipeline:

```bash
bash scripts/run_all_experiments.sh
```

This script sequentially executes all experimental configurations including standard classification evaluation, adversarial robustness assessment, privacy-utility tradeoff analysis, scalability benchmarking, cross-blockchain transferability evaluation, and ablation studies.

Individual experiments can be reproduced using specific commands:

**Standard Classification Performance**:
```bash
python experiments.py --experiment_type standard --datasets all --num_runs 5
```

**Adversarial Robustness Evaluation**:
```bash
python experiments.py --experiment_type adversarial --epsilon_values 0.05,0.1,0.15,0.2
```

**Privacy Analysis**:
```bash
python experiments.py --experiment_type privacy --privacy_budgets 0.1,0.5,1.0,2.0,5.0
```

**Scalability Testing**:
```bash
python experiments.py --experiment_type scalability --graph_sizes 1000,10000,100000,1000000,10000000
```

**Cross-Blockchain Transfer**:
```bash
python experiments.py --experiment_type cross_blockchain --source_chains ethereum,bitcoin
```

## Model Evaluation

To evaluate a trained model on test data:

```bash
python evaluate.py --checkpoint path/to/best_model.pt --dataset ethereum_p --metrics all
```

The evaluation script generates comprehensive metrics including F1 score, AUC-ROC, precision-recall curves, false positive rates, confusion matrices, and certified robustness measurements.

## Pre-trained Models

Pre-trained models for each dataset configuration are available for download:

- GUARDIAN-Ethereum-S: [https://github.com/TommyDzh/DIAM?tab=readme-ov-file]
- GUARDIAN-Ethereum-P: [https://github.com/TommyDzh/DIAM?tab=readme-ov-file]
- GUARDIAN-Bitcoin-M: [https://github.com/TommyDzh/DIAM?tab=readme-ov-file]
- GUARDIAN-Bitcoin-L: [https://github.com/TommyDzh/DIAM?tab=readme-ov-file]

Load pre-trained models using:

```python
from models import GUARDIAN
from config import Config

config = Config.from_json('configs/standard_config.json')
model = GUARDIAN(config.model_config)
model.load_state_dict(torch.load('path/to/pretrained_model.pt'))
```

## Project Structure

The repository organization follows a modular structure facilitating component modification and extension:

```
GUARDIAN-scis/
├── config.py           # Configuration management system
├── models.py           # GUARDIAN architecture implementations
├── layers.py           # Custom neural network layers
├── experiments.py      # Experimental pipeline implementation
├── utils.py            # Utility functions and metrics
├── security.py         # Security mechanisms (not shown but referenced)
├── main.py             # Main execution framework
├── configs/            # Configuration files
│   ├── standard_config.json
│   ├── adversarial_config.json
│   ├── privacy_config.json
│   └── ...
├── data/               # Dataset storage
├── results/            # Experimental results
├── checkpoints/        # Model checkpoints
└── scripts/            # Utility scripts
```

## Results Summary

The GUARDIAN framework achieves state-of-the-art performance across multiple evaluation dimensions:

**Detection Performance**: F1 scores ranging from 93.17% to 98.42% across different cryptocurrency datasets, with false positive rates between 1.6% and 6.8%.

**Adversarial Robustness**: Maintains 92.3% accuracy under PGD attacks with ε=0.1 perturbation budget, with certified accuracy of 89.6% at ε=0.1 through randomized smoothing.

**Privacy Preservation**: Achieves over 90% F1 score with differential privacy budget ε=1.0, demonstrating practical privacy-utility tradeoffs for collaborative detection.

**Scalability**: Processes graphs with up to 100 million nodes with 458ms inference latency, maintaining throughput above 2,000 samples per second.

**Cross-Blockchain Transfer**: Limited to 15% performance degradation when transferring between Bitcoin and Ethereum networks, enabling unified detection systems.

## Disclaimer

This software is provided for research purposes only. Users should ensure compliance with applicable laws and regulations when deploying fraud detection systems in production environments. The authors assume no liability for misuse or consequences arising from the use of this software.