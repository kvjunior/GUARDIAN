"""
GUARDIAN: Graph-based Universal Adversarial Robustness for Detection of Illicit Accounts in Networks
Main execution framework for comprehensive experiments on cryptocurrency transaction networks
IEEE Transactions on Dependable and Secure Computing (TDSC)
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import random
import json
import warnings
warnings.filterwarnings('ignore')

from config import Config, ExperimentConfig
from models import (
    GUARDIAN, 
    AdversarialGUARDIAN, 
    FederatedGUARDIAN,
    PrivacyPreservingGUARDIAN,
    HierarchicalGUARDIAN
)
from experiments import (
    StandardExperiment,
    AdversarialExperiment,
    PrivacyExperiment,
    ScalabilityExperiment,
    CrossBlockchainExperiment,
    AblationStudy
)
from utils import (
    setup_logging,
    save_checkpoint,
    load_checkpoint,
    generate_experiment_id,
    create_results_directory
)

class GUARDIANExecutor:
    """
    Unified execution framework for all GUARDIAN experiments with multi-GPU support
    """
    
    def __init__(self, config_path: str):
        self.config = Config.from_json(config_path)
        self.device_count = torch.cuda.device_count()
        self.experiment_id = generate_experiment_id()
        self.results_dir = create_results_directory(self.experiment_id)
        
        # Initialize logging with academic standards
        self.logger = setup_logging(
            log_file=os.path.join(self.results_dir, 'experiment.log'),
            level=logging.INFO
        )
        
        # Log system configuration
        self._log_system_info()
        
        # Set reproducibility
        self._set_reproducibility(self.config.seed)
        
    def _log_system_info(self):
        """Log comprehensive system information for reproducibility"""
        self.logger.info("=" * 80)
        self.logger.info("GUARDIAN: Cryptocurrency Illicit Account Detection System")
        self.logger.info(f"Experiment ID: {self.experiment_id}")
        self.logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)
        self.logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        self.logger.info(f"GPU Count: {self.device_count}")
        for i in range(self.device_count):
            self.logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        self.logger.info(f"CPU Count: {mp.cpu_count()}")
        self.logger.info(f"PyTorch Version: {torch.__version__}")
        self.logger.info("=" * 80)
        
    def _set_reproducibility(self, seed: int):
        """Ensure reproducible results across all experiments"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.logger.info(f"Random seed set to {seed} for reproducibility")
        
    def run_distributed(self, rank: int, world_size: int, experiment_type: str):
        """Execute experiment in distributed setting across multiple GPUs"""
        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://localhost:{self.config.dist_port}',
            world_size=world_size,
            rank=rank
        )
        
        # Set device for this process
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        
        # Create model based on experiment type
        model = self._create_model(experiment_type).to(device)
        
        # Wrap model with DDP
        model = DDP(model, device_ids=[rank], output_device=rank)
        
        # Create experiment instance
        experiment = self._create_experiment(experiment_type, model, device, rank)
        
        # Execute experiment
        results = experiment.run()
        
        # Save results from rank 0
        if rank == 0:
            self._save_results(results, experiment_type)
            
        # Cleanup
        dist.destroy_process_group()
        
    def _create_model(self, experiment_type: str):
        """Factory method for model creation based on experiment type"""
        model_config = self.config.model_config
        
        if experiment_type == 'standard':
            return GUARDIAN(model_config)
        elif experiment_type == 'adversarial':
            return AdversarialGUARDIAN(model_config)
        elif experiment_type == 'privacy':
            return PrivacyPreservingGUARDIAN(model_config)
        elif experiment_type == 'federated':
            return FederatedGUARDIAN(model_config)
        elif experiment_type == 'hierarchical':
            return HierarchicalGUARDIAN(model_config)
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
            
    def _create_experiment(self, experiment_type: str, model, device, rank: int):
        """Factory method for experiment creation"""
        exp_config = ExperimentConfig(
            data_dir=self.config.data_dir,
            batch_size=self.config.batch_size,
            num_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            device=device,
            rank=rank,
            world_size=self.device_count,
            results_dir=self.results_dir
        )
        
        if experiment_type == 'standard':
            return StandardExperiment(model, exp_config)
        elif experiment_type == 'adversarial':
            return AdversarialExperiment(model, exp_config)
        elif experiment_type == 'privacy':
            return PrivacyExperiment(model, exp_config)
        elif experiment_type == 'scalability':
            return ScalabilityExperiment(model, exp_config)
        elif experiment_type == 'cross_blockchain':
            return CrossBlockchainExperiment(model, exp_config)
        elif experiment_type == 'ablation':
            return AblationStudy(model, exp_config)
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
            
    def _save_results(self, results: dict, experiment_type: str):
        """Save experimental results with comprehensive metadata"""
        output = {
            'experiment_id': self.experiment_id,
            'experiment_type': experiment_type,
            'timestamp': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'results': results,
            'system_info': {
                'gpu_count': self.device_count,
                'pytorch_version': torch.__version__,
                'cuda_version': torch.version.cuda
            }
        }
        
        output_path = os.path.join(self.results_dir, f'{experiment_type}_results.json')
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=4)
            
        self.logger.info(f"Results saved to {output_path}")
        
    def execute_single_gpu(self, experiment_type: str):
        """Execute experiment on single GPU (fallback mode)"""
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = self._create_model(experiment_type).to(device)
        experiment = self._create_experiment(experiment_type, model, device, rank=0)
        results = experiment.run()
        self._save_results(results, experiment_type)
        
    def execute_all_experiments(self):
        """Execute complete experimental pipeline for TDSC submission"""
        experiments = [
            'standard',
            'adversarial',
            'privacy',
            'scalability',
            'cross_blockchain',
            'ablation'
        ]
        
        self.logger.info("Starting comprehensive GUARDIAN evaluation pipeline")
        
        for exp_type in experiments:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Starting {exp_type} experiment")
            self.logger.info(f"{'='*80}")
            
            if self.device_count > 1 and self.config.use_distributed:
                # Multi-GPU execution
                mp.spawn(
                    self.run_distributed,
                    args=(self.device_count, exp_type),
                    nprocs=self.device_count,
                    join=True
                )
            else:
                # Single GPU execution
                self.execute_single_gpu(exp_type)
                
            self.logger.info(f"Completed {exp_type} experiment")
            
        self.logger.info("\nAll experiments completed successfully")
        self._generate_summary_report()
        
    def _generate_summary_report(self):
        """Generate comprehensive summary report for paper submission"""
        self.logger.info("\nGenerating summary report...")
        
        # Aggregate results from all experiments
        summary = {
            'experiment_id': self.experiment_id,
            'total_experiments': 6,
            'completion_time': datetime.now().isoformat(),
            'aggregate_metrics': {}
        }
        
        # Process each experiment's results
        for exp_file in os.listdir(self.results_dir):
            if exp_file.endswith('_results.json'):
                with open(os.path.join(self.results_dir, exp_file), 'r') as f:
                    exp_data = json.load(f)
                    exp_type = exp_data['experiment_type']
                    summary['aggregate_metrics'][exp_type] = {
                        'best_f1': exp_data['results'].get('best_f1', 0),
                        'best_auc': exp_data['results'].get('best_auc', 0),
                        'runtime': exp_data['results'].get('total_runtime', 0)
                    }
                    
        # Save summary
        summary_path = os.path.join(self.results_dir, 'experiment_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
            
        self.logger.info(f"Summary report saved to {summary_path}")
        
def main():
    """Main entry point for GUARDIAN execution"""
    parser = argparse.ArgumentParser(
        description='GUARDIAN: Cryptocurrency Illicit Account Detection System'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.json',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['all', 'standard', 'adversarial', 'privacy', 
                 'scalability', 'cross_blockchain', 'ablation'],
        default='all',
        help='Type of experiment to run'
    )
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='Enable distributed training across multiple GPUs'
    )
    
    args = parser.parse_args()
    
    # Create executor instance
    executor = GUARDIANExecutor(args.config)
    
    # Override distributed setting if specified
    if args.distributed:
        executor.config.use_distributed = True
        
    # Execute experiments
    if args.experiment == 'all':
        executor.execute_all_experiments()
    else:
        if executor.device_count > 1 and executor.config.use_distributed:
            mp.spawn(
                executor.run_distributed,
                args=(executor.device_count, args.experiment),
                nprocs=executor.device_count,
                join=True
            )
        else:
            executor.execute_single_gpu(args.experiment)
            
if __name__ == '__main__':
    main()