"""
Comprehensive utilities for GUARDIAN
Evaluation metrics, data processing, and analysis tools
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report, f1_score
)
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
import logging
import hashlib
from collections import defaultdict
import pandas as pd
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import networkx as nx
from scipy import stats

def setup_logging(
    log_file: str,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup comprehensive logging for experiments
    """
    
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
    # Create logger
    logger = logging.getLogger('GUARDIAN')
    logger.setLevel(level)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(format_string)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def generate_experiment_id() -> str:
    """
    Generate unique experiment ID
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    random_hash = hashlib.md5(str(np.random.random()).encode()).hexdigest()[:8]
    return f"GUARDIAN_{timestamp}_{random_hash}"

def create_results_directory(experiment_id: str, base_dir: str = './results') -> str:
    """
    Create directory structure for experiment results
    """
    
    results_dir = os.path.join(base_dir, experiment_id)
    subdirs = ['checkpoints', 'logs', 'visualizations', 'metrics', 'reports']
    
    os.makedirs(results_dir, exist_ok=True)
    
    for subdir in subdirs:
        os.makedirs(os.path.join(results_dir, subdir), exist_ok=True)
        
    return results_dir

class MetricsCalculator:
    """
    Comprehensive metrics calculation for illicit account detection
    """
    
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.labels = []
        self.probabilities = []
        self.uncertainties = []
        
    def update(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
        uncertainties: Optional[torch.Tensor] = None
    ):
        """Update metrics with batch results"""
        
        self.predictions.extend(preds.cpu().numpy())
        self.labels.extend(labels.cpu().numpy())
        
        if probs is not None:
            self.probabilities.extend(probs.cpu().numpy())
            
        if uncertainties is not None:
            self.uncertainties.extend(uncertainties.cpu().numpy())
            
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics
        """
        
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = (predictions == labels).mean()
        metrics['f1_score'] = f1_score(labels, predictions, average='binary')
        
        # Per-class metrics
        cm = confusion_matrix(labels, predictions)
        
        # Handle division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = np.diag(cm) / cm.sum(axis=0)
            recall = np.diag(cm) / cm.sum(axis=1)
            
        precision = np.nan_to_num(precision, nan=0.0)
        recall = np.nan_to_num(recall, nan=0.0)
        
        metrics['precision_normal'] = precision[0]
        metrics['precision_illicit'] = precision[1]
        metrics['recall_normal'] = recall[0]
        metrics['recall_illicit'] = recall[1]
        
        # Overall precision and recall
        metrics['precision'] = precision[1]  # Focus on illicit class
        metrics['recall'] = recall[1]
        
        # Confusion matrix elements
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # False positive rate (important for financial applications)
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # If probabilities available
        if self.probabilities:
            probabilities = np.array(self.probabilities)
            
            # AUC-ROC
            try:
                metrics['auc_roc'] = roc_auc_score(labels, probabilities[:, 1])
            except:
                metrics['auc_roc'] = 0.5
                
            # AUC-PR (important for imbalanced datasets)
            try:
                metrics['auc_pr'] = average_precision_score(labels, probabilities[:, 1])
            except:
                metrics['auc_pr'] = 0.0
                
            # Brier score (calibration metric)
            metrics['brier_score'] = np.mean((probabilities[:, 1] - labels) ** 2)
            
            # Expected Calibration Error (ECE)
            metrics['ece'] = self._compute_ece(probabilities[:, 1], labels)
            
        # If uncertainties available
        if self.uncertainties:
            uncertainties = np.array(self.uncertainties)
            
            # Uncertainty metrics
            metrics['mean_uncertainty'] = uncertainties.mean()
            metrics['uncertainty_auroc'] = self._compute_uncertainty_auroc(
                predictions, labels, uncertainties
            )
            
        return metrics
        
    def _compute_ece(self, probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error
        """
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece
        
    def _compute_uncertainty_auroc(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        uncertainties: np.ndarray
    ) -> float:
        """
        Compute AUROC for uncertainty as indicator of misclassification
        """
        
        misclassified = (predictions != labels).astype(int)
        
        try:
            return roc_auc_score(misclassified, uncertainties)
        except:
            return 0.5

class DataProcessor:
    """
    Advanced data processing for cryptocurrency transaction graphs
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.scaler = None
        
    def load_data(self, data_path: str) -> Data:
        """
        Load and preprocess graph data
        """
        
        # Load data
        data = torch.load(data_path)
        
        # Validate data format
        assert hasattr(data, 'edge_index'), "Data must have edge_index"
        assert hasattr(data, 'edge_attr'), "Data must have edge_attr"
        assert hasattr(data, 'y'), "Data must have labels"
        
        # Preprocess
        data = self._preprocess_graph(data)
        
        return data
        
    def _preprocess_graph(self, data: Data) -> Data:
        """
        Comprehensive graph preprocessing
        """
        
        # Handle edge timestamps
        if not hasattr(data, 'edge_timestamps'):
            # Extract from edge attributes if available
            if data.edge_attr.size(1) > 0:
                data.edge_timestamps = data.edge_attr[:, -1]
            else:
                # Create dummy timestamps
                data.edge_timestamps = torch.arange(data.edge_index.size(1), dtype=torch.float)
                
        # Normalize edge attributes
        data.edge_attr = self._normalize_features(data.edge_attr)
        
        # Add graph-level statistics
        data = self._add_graph_statistics(data)
        
        # Handle class imbalance
        data = self._handle_imbalance(data)
        
        return data
        
    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Normalize features with robust scaling
        """
        
        # Use robust scaling (less sensitive to outliers)
        median = features.median(dim=0)[0]
        mad = (features - median).abs().median(dim=0)[0]
        
        # Avoid division by zero
        mad = torch.where(mad > 0, mad, torch.ones_like(mad))
        
        normalized = (features - median) / (1.4826 * mad)
        
        return normalized
        
    def _add_graph_statistics(self, data: Data) -> Data:
        """
        Add useful graph statistics as features
        """
        
        num_nodes = data.edge_index.max().item() + 1
        
        # Compute node degrees
        row, col = data.edge_index
        in_degree = torch.zeros(num_nodes)
        out_degree = torch.zeros(num_nodes)
        
        in_degree.scatter_add_(0, col, torch.ones_like(col, dtype=torch.float))
        out_degree.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        
        data.in_degree = in_degree
        data.out_degree = out_degree
        data.degree_ratio = (in_degree + 1) / (out_degree + 1)
        
        # Compute node-level transaction statistics
        edge_values = data.edge_attr[:, 0]  # Assuming first attribute is transaction amount
        
        # Average transaction values
        avg_in_value = torch.zeros(num_nodes)
        avg_out_value = torch.zeros(num_nodes)
        
        scatter_mean(edge_values, col, dim=0, out=avg_in_value)
        scatter_mean(edge_values, row, dim=0, out=avg_out_value)
        
        data.avg_in_value = avg_in_value
        data.avg_out_value = avg_out_value
        
        return data
        
    def _handle_imbalance(self, data: Data) -> Data:
        """
        Handle class imbalance in the dataset
        """
        
        # Compute class weights
        unique_labels, counts = torch.unique(data.y, return_counts=True)
        
        # Inverse frequency weighting
        total_samples = counts.sum()
        class_weights = total_samples / (len(unique_labels) * counts)
        
        # Normalize weights
        class_weights = class_weights / class_weights.sum() * len(unique_labels)
        
        data.class_weights = class_weights
        
        return data
        
    def create_train_val_test_split(
        self,
        data: Data,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        stratify: bool = True
    ) -> Tuple[Data, Data, Data]:
        """
        Create train/val/test splits with stratification
        """
        
        num_nodes = data.y.size(0)
        indices = torch.arange(num_nodes)
        
        if stratify:
            # Stratified split
            train_mask, val_mask, test_mask = self._stratified_split(
                indices, data.y, train_ratio, val_ratio
            )
        else:
            # Random split
            perm = torch.randperm(num_nodes)
            train_size = int(train_ratio * num_nodes)
            val_size = int(val_ratio * num_nodes)
            
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            train_mask[perm[:train_size]] = True
            val_mask[perm[train_size:train_size + val_size]] = True
            test_mask[perm[train_size + val_size:]] = True
            
        # Create split data objects
        train_data = self._create_subgraph(data, train_mask)
        val_data = self._create_subgraph(data, val_mask)
        test_data = self._create_subgraph(data, test_mask)
        
        return train_data, val_data, test_data
        
    def _stratified_split(
        self,
        indices: torch.Tensor,
        labels: torch.Tensor,
        train_ratio: float,
        val_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Stratified train/val/test split
        """
        
        train_mask = torch.zeros_like(labels, dtype=torch.bool)
        val_mask = torch.zeros_like(labels, dtype=torch.bool)
        test_mask = torch.zeros_like(labels, dtype=torch.bool)
        
        # Split each class separately
        for label in labels.unique():
            label_indices = indices[labels == label]
            n_label = len(label_indices)
            
            # Shuffle indices
            perm = torch.randperm(n_label)
            label_indices = label_indices[perm]
            
            # Compute split sizes
            train_size = int(train_ratio * n_label)
            val_size = int(val_ratio * n_label)
            
            # Assign to splits
            train_mask[label_indices[:train_size]] = True
            val_mask[label_indices[train_size:train_size + val_size]] = True
            test_mask[label_indices[train_size + val_size:]] = True
            
        return train_mask, val_mask, test_mask
        
    def _create_subgraph(self, data: Data, mask: torch.Tensor) -> Data:
        """
        Create subgraph from node mask
        """
        
        # Get subgraph edge index
        edge_mask = mask[data.edge_index[0]] & mask[data.edge_index[1]]
        
        # Create new data object
        subgraph = Data()
        
        # Copy edge data
        subgraph.edge_index = data.edge_index[:, edge_mask]
        subgraph.edge_attr = data.edge_attr[edge_mask]
        
        if hasattr(data, 'edge_timestamps'):
            subgraph.edge_timestamps = data.edge_timestamps[edge_mask]
            
        # Copy node data
        subgraph.y = data.y[mask]
        
        # Copy additional features if present
        for key in ['in_degree', 'out_degree', 'degree_ratio', 
                    'avg_in_value', 'avg_out_value', 'class_weights']:
            if hasattr(data, key):
                setattr(subgraph, key, getattr(data, key)[mask])
                
        return subgraph

class Visualizer:
    """
    Comprehensive visualization tools for analysis
    """
    
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def plot_metrics_evolution(
        self,
        metrics_history: Dict[str, List[float]],
        title: str = "Training Metrics Evolution"
    ):
        """
        Plot evolution of metrics during training
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        # Plot different metrics
        metric_groups = [
            ['train_loss', 'val_loss'],
            ['train_f1', 'val_f1'],
            ['train_auc_roc', 'val_auc_roc'],
            ['train_precision', 'train_recall']
        ]
        
        titles = ['Loss', 'F1 Score', 'AUC-ROC', 'Precision vs Recall']
        
        for idx, (metrics, title) in enumerate(zip(metric_groups, titles)):
            ax = axes[idx]
            
            for metric in metrics:
                if metric in metrics_history:
                    values = metrics_history[metric]
                    epochs = range(1, len(values) + 1)
                    
                    label = metric.replace('_', ' ').title()
                    ax.plot(epochs, values, marker='o', label=label)
                    
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'metrics_evolution.png'), dpi=300)
        plt.close()
        
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str] = ['Normal', 'Illicit']
    ):
        """
        Plot confusion matrix with detailed statistics
        """
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax1)
        ax1.set_title('Confusion Matrix (Counts)')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax2)
        ax2.set_title('Confusion Matrix (Normalized)')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
        
    def plot_roc_pr_curves(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        dataset_name: str = ""
    ):
        """
        Plot ROC and PR curves
        """
        
        from sklearn.metrics import roc_curve, precision_recall_curve
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_roc = roc_auc_score(y_true, y_scores)
        
        ax1.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {auc_roc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'ROC Curve {dataset_name}')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # PR curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        auc_pr = average_precision_score(y_true, y_scores)
        
        ax2.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AUC = {auc_pr:.3f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'Precision-Recall Curve {dataset_name}')
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'roc_pr_curves_{dataset_name}.png'), dpi=300)
        plt.close()
        
    def plot_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        node_indices: Optional[torch.Tensor] = None,
        max_nodes: int = 50
    ):
        """
        Visualize attention weights as heatmap
        """
        
        # Convert to dense format if needed
        if attention_weights.dim() == 1:
            # Assume edge-level attention
            num_nodes = int(np.sqrt(attention_weights.size(0)))
            attention_matrix = attention_weights.view(num_nodes, num_nodes)
        else:
            attention_matrix = attention_weights
            
        # Limit size for visualization
        if attention_matrix.size(0) > max_nodes:
            if node_indices is not None:
                # Use specified nodes
                attention_matrix = attention_matrix[node_indices][:, node_indices]
            else:
                # Random sample
                indices = torch.randperm(attention_matrix.size(0))[:max_nodes]
                attention_matrix = attention_matrix[indices][:, indices]
                
        # Convert to numpy
        attention_matrix = attention_matrix.cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_matrix, cmap='YlOrRd', cbar=True,
                   square=True, xticklabels=False, yticklabels=False)
        plt.title('Attention Weights Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'attention_heatmap.png'), dpi=300)
        plt.close()
        
    def plot_embedding_distribution(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        method: str = 'tsne'
    ):
        """
        Visualize embedding distribution using dimensionality reduction
        """
        
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        # Convert to numpy
        embeddings = embeddings.cpu().numpy()
        labels = labels.cpu().numpy()
        
        # Dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        # Plot
        plt.figure(figsize=(10, 8))
        
        # Plot each class
        for label in np.unique(labels):
            mask = labels == label
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       label=f'Class {label}', alpha=0.6, s=30)
                       
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.title(f'Embedding Distribution ({method.upper()})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'embedding_{method}.png'), dpi=300)
        plt.close()

class ModelCheckpointer:
    """
    Advanced model checkpointing with versioning
    """
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoint_history = []
        
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """
        Save model checkpoint with metadata
        """
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save regular checkpoint
        checkpoint_name = f'checkpoint_epoch_{epoch}.pt'
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        torch.save(checkpoint, checkpoint_path)
        
        self.checkpoint_history.append(checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            
        # Clean old checkpoints
        if len(self.checkpoint_history) > self.max_checkpoints:
            old_checkpoint = self.checkpoint_history.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
                
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict:
        """
        Load model checkpoint
        """
        
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        return checkpoint

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: str,
    is_best: bool = False
):
    """
    Wrapper function for checkpoint saving
    """
    
    checkpointer = ModelCheckpointer(checkpoint_dir)
    checkpointer.save_checkpoint(model, optimizer, epoch, metrics, is_best)

def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict:
    """
    Wrapper function for checkpoint loading
    """
    
    checkpointer = ModelCheckpointer(os.path.dirname(checkpoint_path))
    return checkpointer.load_checkpoint(checkpoint_path, model, optimizer)

class ExperimentTracker:
    """
    Track and compare multiple experiments
    """
    
    def __init__(self, tracking_file: str = 'experiments.json'):
        self.tracking_file = tracking_file
        self.experiments = self._load_experiments()
        
    def _load_experiments(self) -> Dict:
        """Load existing experiments"""
        
        if os.path.exists(self.tracking_file):
            with open(self.tracking_file, 'r') as f:
                return json.load(f)
        return {}
        
    def log_experiment(
        self,
        experiment_id: str,
        config: Dict,
        results: Dict,
        metadata: Optional[Dict] = None
    ):
        """
        Log experiment results
        """
        
        experiment_data = {
            'config': config,
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.experiments[experiment_id] = experiment_data
        
        # Save to file
        with open(self.tracking_file, 'w') as f:
            json.dump(self.experiments, f, indent=4)
            
    def compare_experiments(
        self,
        experiment_ids: List[str],
        metrics: List[str] = ['f1_score', 'auc_roc']
    ) -> pd.DataFrame:
        """
        Compare multiple experiments
        """
        
        data = []
        
        for exp_id in experiment_ids:
            if exp_id in self.experiments:
                exp = self.experiments[exp_id]
                row = {'experiment_id': exp_id}
                
                # Add metrics
                for metric in metrics:
                    if metric in exp['results']:
                        row[metric] = exp['results'][metric]
                        
                # Add config info
                row['model'] = exp['config'].get('model_type', 'unknown')
                row['dataset'] = exp['config'].get('dataset', 'unknown')
                
                data.append(row)
                
        return pd.DataFrame(data)

def compute_graph_statistics(data: Data) -> Dict[str, float]:
    """
    Compute comprehensive graph statistics
    """
    
    stats = {}
    
    # Basic statistics
    num_nodes = data.edge_index.max().item() + 1
    num_edges = data.edge_index.size(1)
    
    stats['num_nodes'] = num_nodes
    stats['num_edges'] = num_edges
    stats['edge_density'] = num_edges / (num_nodes * (num_nodes - 1))
    
    # Degree statistics
    degrees = torch.zeros(num_nodes)
    degrees.scatter_add_(0, data.edge_index[0], torch.ones(num_edges))
    
    stats['avg_degree'] = degrees.mean().item()
    stats['max_degree'] = degrees.max().item()
    stats['degree_std'] = degrees.std().item()
    
    # Class distribution
    if hasattr(data, 'y'):
        unique_labels, counts = torch.unique(data.y, return_counts=True)
        
        for label, count in zip(unique_labels, counts):
            stats[f'class_{label}_count'] = count.item()
            stats[f'class_{label}_ratio'] = (count / len(data.y)).item()
            
    # Edge attribute statistics
    if hasattr(data, 'edge_attr'):
        edge_values = data.edge_attr[:, 0]  # First attribute
        
        stats['avg_edge_value'] = edge_values.mean().item()
        stats['max_edge_value'] = edge_values.max().item()
        stats['edge_value_std'] = edge_values.std().item()
        
    return stats

def analyze_model_complexity(model: nn.Module) -> Dict[str, int]:
    """
    Analyze model complexity and parameter count
    """
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    # Compute model size in MB
    model_size = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': model_size
    }

def perform_statistical_tests(
    results_1: List[float],
    results_2: List[float],
    test_type: str = 'wilcoxon'
) -> Dict[str, float]:
    """
    Perform statistical significance tests
    """
    
    if test_type == 'wilcoxon':
        # Wilcoxon signed-rank test
        statistic, p_value = stats.wilcoxon(results_1, results_2)
    elif test_type == 'ttest':
        # Paired t-test
        statistic, p_value = stats.ttest_rel(results_1, results_2)
    else:
        raise ValueError(f"Unknown test type: {test_type}")
        
    # Compute effect size (Cohen's d)
    mean_diff = np.mean(results_1) - np.mean(results_2)
    pooled_std = np.sqrt((np.std(results_1)**2 + np.std(results_2)**2) / 2)
    cohens_d = mean_diff / pooled_std
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05
    }

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1, 
                 out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Scatter mean operation
    """
    
    dim_size = int(index.max()) + 1
    
    if out is None:
        out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
        
    # Count occurrences
    ones = torch.ones_like(index, dtype=torch.float)
    count = torch.zeros(dim_size, dtype=torch.float, device=src.device)
    count.scatter_add_(0, index, ones)
    
    # Sum values
    out.scatter_add_(dim, index.unsqueeze(-1).expand_as(src), src)
    
    # Compute mean
    count = count.clamp(min=1)
    out = out / count.unsqueeze(-1)
    
    return out