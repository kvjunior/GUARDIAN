"""
Centralized configuration management for GUARDIAN
Handles all experiment configurations and hyperparameters
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import yaml

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    
    # Architecture
    model_type: str = 'GUARDIAN'
    hidden_dim: int = 256
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.2
    edge_dim: int = 8
    
    # Edge2Seq parameters
    max_sequence_length: int = 100
    use_bidirectional: bool = True
    rnn_type: str = 'gru'  # 'lstm' or 'gru'
    
    # MGD parameters
    use_discrepancy: bool = True
    discrepancy_weight: float = 0.5
    attention_temperature: float = 1.0
    
    # Temporal attention
    use_temporal_attention: bool = True
    temporal_hidden_dim: int = 128
    
    # Hierarchical pooling
    pooling_ratios: list = field(default_factory=lambda: [0.5, 0.5])
    
    # Classification head
    classifier_layers: int = 3
    use_uncertainty: bool = True
    
    # Adversarial robustness
    adversarial_epsilon: float = 0.1
    adversarial_steps: int = 10
    adversarial_step_size: float = 0.01
    robustness_weight: float = 0.5
    
    # Privacy parameters
    privacy_epsilon: float = 1.0
    privacy_delta: float = 1e-5
    noise_multiplier: float = 0.1
    max_grad_norm: float = 1.0
    
    # Federated learning
    num_clients: int = 10
    client_fraction: float = 0.1
    local_epochs: int = 5
    secure_aggregation: bool = True

@dataclass
class TrainingConfig:
    """Training configuration"""
    
    # Basic training
    num_epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # Optimization
    optimizer: str = 'adamw'
    scheduler: str = 'cosine_annealing_warm_restarts'
    warmup_epochs: int = 2
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Early stopping
    patience: int = 10
    min_delta: float = 0.001
    
    # Checkpointing
    save_frequency: int = 5
    max_checkpoints: int = 5
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # Sampling
    num_neighbors: list = field(default_factory=lambda: [25, 10])
    
    # Class balancing
    use_class_weights: bool = True
    oversample_minority: bool = False

@dataclass
class ExperimentConfig:
    """Experiment-specific configuration"""
    
    # Data
    data_dir: str = './data'
    dataset_names: list = field(default_factory=lambda: [
        'ethereum_s', 'ethereum_p', 'bitcoin_m', 'bitcoin_l'
    ])
    
    # Split ratios
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    stratify: bool = True
    
    # Device settings
    device: str = 'cuda'
    rank: int = 0
    world_size: int = 1
    
    # Results
    results_dir: str = './results'
    
    # Experiment-specific
    experiment_type: str = 'standard'
    num_runs: int = 5  # For statistical significance
    
    # Visualization
    plot_attention: bool = True
    plot_embeddings: bool = True
    embedding_method: str = 'tsne'  # 'tsne' or 'pca'

class Config:
    """Main configuration class"""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        
        # Initialize with defaults
        self.model_config = ModelConfig()
        self.training_config = TrainingConfig()
        self.experiment_config = ExperimentConfig()
        
        # System configuration
        self.seed: int = 42
        self.use_distributed: bool = True
        self.dist_backend: str = 'nccl'
        self.dist_port: int = 12355
        
        # Logging
        self.log_level: str = 'INFO'
        self.log_frequency: int = 100
        self.use_tensorboard: bool = True
        self.tensorboard_dir: str = './tensorboard'
        
        # Update from provided config
        if config_dict is not None:
            self.update(config_dict)
            
    def update(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        
        # Update model config
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                if hasattr(self.model_config, key):
                    setattr(self.model_config, key, value)
                    
        # Update training config
        if 'training' in config_dict:
            for key, value in config_dict['training'].items():
                if hasattr(self.training_config, key):
                    setattr(self.training_config, key, value)
                    
        # Update experiment config
        if 'experiment' in config_dict:
            for key, value in config_dict['experiment'].items():
                if hasattr(self.experiment_config, key):
                    setattr(self.experiment_config, key, value)
                    
        # Update system config
        for key in ['seed', 'use_distributed', 'dist_backend', 'dist_port',
                    'log_level', 'log_frequency', 'use_tensorboard', 'tensorboard_dir']:
            if key in config_dict:
                setattr(self, key, config_dict[key])
                
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        
        return {
            'model': asdict(self.model_config),
            'training': asdict(self.training_config),
            'experiment': asdict(self.experiment_config),
            'seed': self.seed,
            'use_distributed': self.use_distributed,
            'dist_backend': self.dist_backend,
            'dist_port': self.dist_port,
            'log_level': self.log_level,
            'log_frequency': self.log_frequency,
            'use_tensorboard': self.use_tensorboard,
            'tensorboard_dir': self.tensorboard_dir
        }
        
    @classmethod
    def from_json(cls, json_path: str) -> 'Config':
        """Load configuration from JSON file"""
        
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
            
        return cls(config_dict)
        
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file"""
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        return cls(config_dict)
		
	def save_json(self, json_path: str):
        """Save configuration to JSON file"""
        
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
            
    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
            
    def validate(self) -> bool:
        """Validate configuration consistency"""
        
        # Check device availability
        if self.experiment_config.device == 'cuda':
            import torch
            if not torch.cuda.is_available():
                print("Warning: CUDA requested but not available. Falling back to CPU.")
                self.experiment_config.device = 'cpu'
                
        # Check data directory
        if not os.path.exists(self.experiment_config.data_dir):
            raise ValueError(f"Data directory {self.experiment_config.data_dir} does not exist")
            
        # Validate model parameters
        if self.model_config.hidden_dim % self.model_config.num_heads != 0:
            raise ValueError(
                f"Hidden dimension ({self.model_config.hidden_dim}) must be divisible "
                f"by number of heads ({self.model_config.num_heads})"
            )
            
        # Validate privacy parameters
        if self.model_config.privacy_epsilon <= 0:
            raise ValueError("Privacy epsilon must be positive")
            
        # Validate training parameters
        if self.training_config.batch_size <= 0:
            raise ValueError("Batch size must be positive")
            
        if self.training_config.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
            
        return True
        
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration as dictionary"""
        return asdict(self.model_config)
        
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration as dictionary"""
        return asdict(self.training_config)
        
    def get_experiment_config(self) -> Dict[str, Any]:
        """Get experiment configuration as dictionary"""
        return asdict(self.experiment_config)
        
    def __repr__(self) -> str:
        """String representation of configuration"""
        return json.dumps(self.to_dict(), indent=2)

# Predefined configurations for different experiment types

def get_standard_config() -> Config:
    """Standard experiment configuration"""
    
    config_dict = {
        'model': {
            'model_type': 'GUARDIAN',
            'hidden_dim': 256,
            'num_layers': 3,
            'num_heads': 8,
            'dropout': 0.2
        },
        'training': {
            'num_epochs': 30,
            'batch_size': 128,
            'learning_rate': 0.001,
            'weight_decay': 1e-4
        },
        'experiment': {
            'experiment_type': 'standard',
            'num_runs': 5
        }
    }
    
    return Config(config_dict)

def get_adversarial_config() -> Config:
    """Adversarial robustness experiment configuration"""
    
    config_dict = {
        'model': {
            'model_type': 'AdversarialGUARDIAN',
            'hidden_dim': 256,
            'num_layers': 3,
            'adversarial_epsilon': 0.1,
            'adversarial_steps': 20,
            'robustness_weight': 0.5
        },
        'training': {
            'num_epochs': 20,
            'batch_size': 64,
            'learning_rate': 0.0005
        },
        'experiment': {
            'experiment_type': 'adversarial',
            'num_runs': 3
        }
    }
    
    return Config(config_dict)

def get_privacy_config() -> Config:
    """Privacy-preserving experiment configuration"""
    
    config_dict = {
        'model': {
            'model_type': 'PrivacyPreservingGUARDIAN',
            'hidden_dim': 256,
            'num_layers': 3,
            'privacy_epsilon': 1.0,
            'privacy_delta': 1e-5,
            'noise_multiplier': 0.1,
            'max_grad_norm': 1.0
        },
        'training': {
            'num_epochs': 15,
            'batch_size': 256,
            'learning_rate': 0.005
        },
        'experiment': {
            'experiment_type': 'privacy',
            'num_runs': 3
        }
    }
    
    return Config(config_dict)

def get_scalability_config() -> Config:
    """Scalability experiment configuration"""
    
    config_dict = {
        'model': {
            'model_type': 'GUARDIAN',
            'hidden_dim': 128,  # Smaller for scalability
            'num_layers': 2,
            'num_heads': 4
        },
        'training': {
            'num_epochs': 5,
            'batch_size': 512,
            'learning_rate': 0.001
        },
        'experiment': {
            'experiment_type': 'scalability',
            'num_runs': 1
        }
    }
    
    return Config(config_dict)

def get_federated_config() -> Config:
    """Federated learning experiment configuration"""
    
    config_dict = {
        'model': {
            'model_type': 'FederatedGUARDIAN',
            'hidden_dim': 256,
            'num_layers': 3,
            'num_clients': 10,
            'client_fraction': 0.3,
            'local_epochs': 5,
            'secure_aggregation': True
        },
        'training': {
            'num_epochs': 50,  # Federated rounds
            'batch_size': 64,
            'learning_rate': 0.001
        },
        'experiment': {
            'experiment_type': 'federated',
            'num_runs': 1
        }
    }
    
    return Config(config_dict)

def get_hierarchical_config() -> Config:
    """Hierarchical model experiment configuration"""
    
    config_dict = {
        'model': {
            'model_type': 'HierarchicalGUARDIAN',
            'hidden_dim': 256,
            'num_layers': 3,
            'pooling_ratios': [0.5, 0.3, 0.2]
        },
        'training': {
            'num_epochs': 25,
            'batch_size': 128,
            'learning_rate': 0.001
        },
        'experiment': {
            'experiment_type': 'hierarchical',
            'num_runs': 3
        }
    }
    
    return Config(config_dict)

# Configuration for your specific hardware setup (4 RTX 3090s)
def get_multi_gpu_config() -> Config:
    """Configuration optimized for 4 RTX 3090 GPUs"""
    
    config_dict = {
        'model': {
            'hidden_dim': 512,  # Larger model for multi-GPU
            'num_layers': 4,
            'num_heads': 16
        },
        'training': {
            'batch_size': 512,  # 128 per GPU
            'learning_rate': 0.004,  # Scaled with batch size
            'num_workers': 16  # 4 per GPU
        },
        'use_distributed': True,
        'dist_backend': 'nccl',
        'dist_port': 12355
    }
    
    return Config(config_dict)

# Example configuration files

DEFAULT_CONFIG = """
{
    "model": {
        "model_type": "GUARDIAN",
        "hidden_dim": 256,
        "num_layers": 3,
        "num_heads": 8,
        "dropout": 0.2,
        "edge_dim": 8,
        "use_temporal_attention": true,
        "use_uncertainty": true
    },
    "training": {
        "num_epochs": 30,
        "batch_size": 128,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "optimizer": "adamw",
        "scheduler": "cosine_annealing_warm_restarts",
        "max_grad_norm": 1.0,
        "num_workers": 4
    },
    "experiment": {
        "data_dir": "./data",
        "dataset_names": ["ethereum_s", "ethereum_p", "bitcoin_m", "bitcoin_l"],
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "stratify": true,
        "results_dir": "./results"
    },
    "seed": 42,
    "use_distributed": false,
    "log_level": "INFO",
    "use_tensorboard": true
}
"""

# Save default configuration
def save_default_configs():
    """Save default configuration files"""
    
    os.makedirs('configs', exist_ok=True)
    
    # Save JSON version
    with open('configs/default_config.json', 'w') as f:
        f.write(DEFAULT_CONFIG)
        
    # Save specific experiment configs
    configs = {
        'standard': get_standard_config(),
        'adversarial': get_adversarial_config(),
        'privacy': get_privacy_config(),
        'scalability': get_scalability_config(),
        'federated': get_federated_config(),
        'hierarchical': get_hierarchical_config(),
        'multi_gpu': get_multi_gpu_config()
    }
    
    for name, config in configs.items():
        config.save_json(f'configs/{name}_config.json')
        
    print("Default configuration files saved to 'configs' directory")

if __name__ == '__main__':
    # Save default configurations when module is run directly
    save_default_configs()