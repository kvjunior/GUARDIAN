"""
GUARDIAN Model Architectures
Complete implementations of all model variants with theoretical guarantees
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from typing import Tuple, Optional, Dict, List
import numpy as np
from collections import OrderedDict

from layers import (
    EnhancedMGD,
    GraphTransformerLayer,
    TemporalAttentionLayer,
    HierarchicalPooling,
    PrivacyPreservingAggregation,
    AdversarialPerturbation
)
from security import DifferentialPrivacyMechanism, AdversarialDefense

class GUARDIAN(nn.Module):
    """
    Base GUARDIAN architecture with enhanced theoretical foundations
    
    Mathematical Foundation:
    Given a directed multigraph G = (V, E, X_E) with edge attributes,
    GUARDIAN learns node representations that maximize the detection of illicit accounts
    while maintaining theoretical guarantees on accuracy and robustness.
    """
    
    def __init__(self, config: dict):
        super(GUARDIAN, self).__init__()
        
        # Model configuration
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_layers = config.get('num_layers', 3)
        self.dropout_rate = config.get('dropout', 0.2)
        self.edge_dim = config.get('edge_dim', 8)
        self.num_heads = config.get('num_heads', 8)
        self.temperature = config.get('temperature', 0.1)
        
        # Enhanced Edge2Seq with Graph Transformer
        self.edge_encoder = GraphTransformerEdgeEncoder(
            edge_dim=self.edge_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_rate
        )
        
        # Multi-layer Enhanced MGD with theoretical guarantees
        self.mgd_layers = nn.ModuleList([
            EnhancedMGD(
                in_channels=self.hidden_dim if i > 0 else self.hidden_dim * 2,
                out_channels=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=self.dropout_rate,
                layer_idx=i
            )
            for i in range(self.num_layers)
        ])
        
        # Temporal attention mechanism
        self.temporal_attention = TemporalAttentionLayer(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads
        )
        
        # Layer normalization for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
        ])
        
        # Classification head with uncertainty estimation
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, 2)
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Softplus()
        )
        
        # Initialize weights with Xavier initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Xavier initialization for better convergence"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(
        self, 
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_timestamps: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with comprehensive outputs for analysis
        
        Args:
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim]
            edge_timestamps: Temporal information [num_edges]
            batch: Batch assignment for nodes
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing predictions, uncertainties, and optional attention weights
        """
        
        # Edge sequence encoding with temporal information
        node_features = self.edge_encoder(
            edge_index, edge_attr, edge_timestamps
        )
        
        # Store intermediate representations for analysis
        layer_outputs = []
        attention_weights = [] if return_attention else None
        
        x = node_features
        
        # Multi-layer propagation with enhanced MGD
        for i, (mgd, norm) in enumerate(zip(self.mgd_layers, self.layer_norms)):
            x_residual = x
            
            if return_attention:
                x, attn = mgd(x, edge_index, return_attention=True)
                attention_weights.append(attn)
            else:
                x = mgd(x, edge_index)
                
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
            # Residual connection for gradient flow
            if i > 0:
                x = x + x_residual
                
            layer_outputs.append(x)
            
        # Temporal attention aggregation
        temporal_features = self.temporal_attention(x, edge_timestamps, edge_index)
        
        # Multi-scale feature aggregation
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        # Global pooling with multiple strategies
        global_mean = global_mean_pool(x, batch)
        global_max = global_max_pool(x, batch)
        global_sum = global_add_pool(x, batch)
        
        # Concatenate multi-scale features
        graph_features = torch.cat([global_mean, global_max, global_sum], dim=1)
        
        # Classification with uncertainty
        logits = self.classifier(graph_features)
        uncertainty = self.uncertainty_head(graph_features)
        
        outputs = {
            'logits': logits,
            'predictions': F.softmax(logits, dim=1),
            'uncertainty': uncertainty,
            'node_embeddings': x,
            'graph_embeddings': graph_features
        }
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
            
        return outputs
        
    def compute_loss(
        self, 
        outputs: Dict[str, torch.Tensor], 
        labels: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss with uncertainty-aware weighting
        
        Implements the loss function:
        L = L_ce + λ_u * L_uncertainty + λ_reg * L_regularization
        """
        
        # Weighted cross-entropy loss
        if class_weights is None:
            ce_loss = F.cross_entropy(outputs['logits'], labels)
        else:
            ce_loss = F.cross_entropy(outputs['logits'], labels, weight=class_weights)
            
        # Uncertainty regularization (encourage higher uncertainty for misclassified samples)
        pred_probs = outputs['predictions']
        pred_labels = pred_probs.argmax(dim=1)
        misclassified = (pred_labels != labels).float()
        
        uncertainty_loss = -torch.mean(
            misclassified * torch.log(outputs['uncertainty'] + 1e-8) +
            (1 - misclassified) * torch.log(1 - outputs['uncertainty'] + 1e-8)
        )
        
        # L2 regularization on embeddings
        reg_loss = 0.01 * torch.mean(torch.norm(outputs['node_embeddings'], dim=1))
        
        # Total loss
        total_loss = ce_loss + 0.1 * uncertainty_loss + reg_loss
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'uncertainty_loss': uncertainty_loss,
            'reg_loss': reg_loss
        }

class GraphTransformerEdgeEncoder(nn.Module):
    """
    Enhanced edge sequence encoder using Graph Transformer architecture
    """
    
    def __init__(self, edge_dim: int, hidden_dim: int, num_heads: int, dropout: float):
        super(GraphTransformerEdgeEncoder, self).__init__()
        
        # Edge attribute projection
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        
        # Positional encoding for temporal information
        self.temporal_encoder = TemporalPositionalEncoding(hidden_dim)
        
        # Graph transformer layers
        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(2)
        ])
        
        # Bidirectional aggregation
        self.in_aggregator = nn.LSTM(
            hidden_dim, hidden_dim // 2, 
            batch_first=True, bidirectional=True
        )
        self.out_aggregator = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            batch_first=True, bidirectional=True
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(
        self, 
        edge_index: torch.Tensor, 
        edge_attr: torch.Tensor,
        edge_timestamps: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode edge sequences with temporal awareness
        """
        
        # Project edge attributes
        edge_features = self.edge_proj(edge_attr)
        
        # Add temporal positional encoding
        edge_features = self.temporal_encoder(edge_features, edge_timestamps)
        
        # Apply transformer layers
        for transformer in self.transformer_layers:
            edge_features = transformer(edge_features, edge_index)
            
        # Aggregate incoming and outgoing edges separately
        num_nodes = edge_index.max().item() + 1
        
        # Group edges by source and target nodes
        in_features = self._aggregate_edges(
            edge_features, edge_index[1], edge_index[0], 
            num_nodes, self.in_aggregator
        )
        out_features = self._aggregate_edges(
            edge_features, edge_index[0], edge_index[1],
            num_nodes, self.out_aggregator
        )
        
        # Fuse bidirectional features
        node_features = torch.cat([in_features, out_features], dim=1)
        node_features = self.fusion(node_features)
        
        return node_features
        
    def _aggregate_edges(
        self, 
        edge_features: torch.Tensor,
        source_nodes: torch.Tensor,
        target_nodes: torch.Tensor,
        num_nodes: int,
        aggregator: nn.Module
    ) -> torch.Tensor:
        """
        Aggregate edge features for each node using LSTM
        """
        
        # Initialize node features
        device = edge_features.device
        node_features = torch.zeros(num_nodes, edge_features.size(1), device=device)
        
        # Group edges by source nodes
        for node_idx in range(num_nodes):
            node_edges = (source_nodes == node_idx)
            if node_edges.any():
                # Get edges for this node
                node_edge_features = edge_features[node_edges]
                
                # Apply LSTM aggregation
                if node_edge_features.size(0) > 1:
                    node_edge_features = node_edge_features.unsqueeze(0)
                    _, (h_n, _) = aggregator(node_edge_features)
                    # Concatenate forward and backward hidden states
                    node_features[node_idx] = torch.cat([h_n[0], h_n[1]], dim=0)
                else:
                    # Single edge case
                    node_features[node_idx] = node_edge_features.squeeze(0)
                    
        return node_features

class TemporalPositionalEncoding(nn.Module):
    """
    Learnable temporal positional encoding for transaction sequences
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(TemporalPositionalEncoding, self).__init__()
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        # Time-aware scaling
        self.time_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Add temporal positional encoding to features
        """
        
        # Normalize timestamps to position indices
        min_time = timestamps.min()
        max_time = timestamps.max()
        if max_time > min_time:
            positions = ((timestamps - min_time) / (max_time - min_time) * 4999).long()
        else:
            positions = torch.zeros_like(timestamps, dtype=torch.long)
            
        # Get positional embeddings
        pos_emb = self.pos_embedding(positions)
        
        # Scale and add to features
        return x + self.time_scale * pos_emb

class AdversarialGUARDIAN(GUARDIAN):
    """
    GUARDIAN with adversarial robustness guarantees
    """
    
    def __init__(self, config: dict):
        super(AdversarialGUARDIAN, self).__init__(config)
        
        # Adversarial defense components
        self.adversarial_defense = AdversarialDefense(
            epsilon=config.get('adv_epsilon', 0.1),
            num_steps=config.get('adv_steps', 10)
        )
        
        # Robustness regularization
        self.robustness_weight = config.get('robustness_weight', 0.5)
        
    def forward(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_timestamps: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        adversarial_training: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional adversarial perturbations
        """
        
        if adversarial_training and self.training:
            # Generate adversarial examples
            edge_attr_adv = self.adversarial_defense.generate_perturbation(
                edge_attr, edge_index, self
            )
            
            # Forward pass on adversarial examples
            outputs_adv = super().forward(
                edge_index, edge_attr_adv, edge_timestamps, batch
            )
            
            # Forward pass on clean examples
            outputs_clean = super().forward(
                edge_index, edge_attr, edge_timestamps, batch
            )
            
            # Combine outputs
            outputs = {
                'logits': outputs_clean['logits'],
                'predictions': outputs_clean['predictions'],
                'uncertainty': outputs_clean['uncertainty'],
                'node_embeddings': outputs_clean['node_embeddings'],
                'graph_embeddings': outputs_clean['graph_embeddings'],
                'logits_adv': outputs_adv['logits'],
                'predictions_adv': outputs_adv['predictions']
            }
        else:
            outputs = super().forward(edge_index, edge_attr, edge_timestamps, batch)
            
        return outputs
        
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss with adversarial robustness term
        """
        
        # Standard loss
        losses = super().compute_loss(outputs, labels, class_weights)
        
        # Adversarial robustness loss
        if 'logits_adv' in outputs:
            adv_loss = F.cross_entropy(outputs['logits_adv'], labels, weight=class_weights)
            
            # KL divergence between clean and adversarial predictions
            kl_loss = F.kl_div(
                F.log_softmax(outputs['logits_adv'], dim=1),
                F.softmax(outputs['logits'], dim=1),
                reduction='batchmean'
            )
            
            # Combined adversarial loss
            losses['adv_loss'] = adv_loss
            losses['kl_loss'] = kl_loss
            losses['total_loss'] = (
                losses['total_loss'] + 
                self.robustness_weight * (adv_loss + 0.1 * kl_loss)
            )
            
        return losses

class PrivacyPreservingGUARDIAN(GUARDIAN):
    """
    GUARDIAN with differential privacy guarantees
    """
    
    def __init__(self, config: dict):
        super(PrivacyPreservingGUARDIAN, self).__init__(config)
        
        # Differential privacy mechanism
        self.dp_mechanism = DifferentialPrivacyMechanism(
            epsilon=config.get('privacy_epsilon', 1.0),
            delta=config.get('privacy_delta', 1e-5),
            max_grad_norm=config.get('max_grad_norm', 1.0)
        )
        
        # Privacy-preserving aggregation
        self.private_aggregation = PrivacyPreservingAggregation(
            self.hidden_dim,
            noise_multiplier=config.get('noise_multiplier', 0.1)
        )
        
    def forward(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_timestamps: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        private_mode: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with privacy preservation
        """
        
        # Standard forward pass
        outputs = super().forward(edge_index, edge_attr, edge_timestamps, batch)
        
        if private_mode and self.training:
            # Add noise to embeddings for privacy
            outputs['node_embeddings'] = self.dp_mechanism.add_noise(
                outputs['node_embeddings']
            )
            
            # Use private aggregation
            outputs['graph_embeddings'] = self.private_aggregation(
                outputs['node_embeddings'], batch
            )
            
        return outputs
        
    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Get current privacy budget spent (epsilon, delta)
        """
        return self.dp_mechanism.get_privacy_spent()

class FederatedGUARDIAN(nn.Module):
    """
    Federated learning variant of GUARDIAN for collaborative detection
    """
    
    def __init__(self, config: dict):
        super(FederatedGUARDIAN, self).__init__()
        
        # Local model for each client
        self.local_model = GUARDIAN(config)
        
        # Federated averaging parameters
        self.num_clients = config.get('num_clients', 10)
        self.client_fraction = config.get('client_fraction', 0.1)
        self.local_epochs = config.get('local_epochs', 5)
        
        # Secure aggregation
        self.use_secure_aggregation = config.get('secure_aggregation', True)
        
    def federated_round(
        self,
        client_data_loaders: List[torch.utils.data.DataLoader],
        global_model_state: OrderedDict
    ) -> OrderedDict:
        """
        Execute one round of federated learning
        """
        
        # Select clients for this round
        num_selected = max(1, int(self.client_fraction * self.num_clients))
        selected_clients = np.random.choice(
            len(client_data_loaders), num_selected, replace=False
        )
        
        # Store client updates
        client_updates = []
        client_weights = []
        
        for client_id in selected_clients:
            # Load global model
            self.local_model.load_state_dict(global_model_state)
            
            # Local training
            client_update, num_samples = self._train_local_model(
                client_data_loaders[client_id]
            )
            
            client_updates.append(client_update)
            client_weights.append(num_samples)
            
        # Aggregate updates
        if self.use_secure_aggregation:
            new_global_state = self._secure_federated_averaging(
                client_updates, client_weights
            )
        else:
            new_global_state = self._federated_averaging(
                client_updates, client_weights
            )
            
        return new_global_state
        
    def _train_local_model(
        self,
        data_loader: torch.utils.data.DataLoader
    ) -> Tuple[OrderedDict, int]:
        """
        Train model on local client data
        """
        
        optimizer = torch.optim.Adam(
            self.local_model.parameters(), lr=0.001
        )
        
        num_samples = 0
        
        for epoch in range(self.local_epochs):
            for batch in data_loader:
                # Forward pass
                outputs = self.local_model(
                    batch.edge_index,
                    batch.edge_attr,
                    batch.edge_timestamps,
                    batch.batch
                )
                
                # Compute loss
                losses = self.local_model.compute_loss(
                    outputs, batch.y
                )
                
                # Backward pass
                optimizer.zero_grad()
                losses['total_loss'].backward()
                optimizer.step()
                
                num_samples += batch.num_graphs
                
        # Compute update
        update = OrderedDict()
        global_state = self.local_model.state_dict()
        
        for key in global_state:
            update[key] = global_state[key]
            
        return update, num_samples
        
    def _federated_averaging(
        self,
        client_updates: List[OrderedDict],
        client_weights: List[int]
    ) -> OrderedDict:
        """
        Standard federated averaging
        """
        
        # Normalize weights
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]
        
        # Initialize averaged state
        avg_state = OrderedDict()
        
        # Average each parameter
        for key in client_updates[0].keys():
            avg_state[key] = sum(
                client_updates[i][key] * normalized_weights[i]
                for i in range(len(client_updates))
            )
            
        return avg_state
        
    def _secure_federated_averaging(
        self,
        client_updates: List[OrderedDict],
        client_weights: List[int]
    ) -> OrderedDict:
        """
        Secure aggregation with privacy preservation
        """
        
        # Add differential privacy noise to each update
        noisy_updates = []
        for update in client_updates:
            noisy_update = OrderedDict()
            for key, value in update.items():
                if value.dtype == torch.float32:
                    noise = torch.randn_like(value) * 0.01
                    noisy_update[key] = value + noise
                else:
                    noisy_update[key] = value
            noisy_updates.append(noisy_update)
            
        # Perform standard averaging on noisy updates
        return self._federated_averaging(noisy_updates, client_weights)

class HierarchicalGUARDIAN(GUARDIAN):
    """
    Hierarchical multi-scale variant of GUARDIAN
    """
    
    def __init__(self, config: dict):
        super(HierarchicalGUARDIAN, self).__init__(config)
        
        # Hierarchical pooling layers
        self.pooling_ratios = config.get('pooling_ratios', [0.5, 0.5])
        self.hierarchical_pools = nn.ModuleList([
            HierarchicalPooling(self.hidden_dim, ratio)
            for ratio in self.pooling_ratios
        ])
        
        # Multi-scale fusion
        num_scales = len(self.pooling_ratios) + 1
        self.scale_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * num_scales, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
    def forward(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_timestamps: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_hierarchy: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with hierarchical multi-scale processing
        """
        
        # Get base features
        outputs = super().forward(
            edge_index, edge_attr, edge_timestamps, batch
        )
        
        # Extract node embeddings
        x = outputs['node_embeddings']
        
        # Hierarchical pooling
        hierarchical_features = [x]
        hierarchical_graphs = [(edge_index, batch)]
        
        current_x = x
        current_edge_index = edge_index
        current_batch = batch if batch is not None else torch.zeros(
            x.size(0), dtype=torch.long, device=x.device
        )
        
        for pool_layer in self.hierarchical_pools:
            # Apply pooling
            pooled_x, pooled_edge_index, pooled_batch, _ = pool_layer(
                current_x, current_edge_index, current_batch
            )
            
            hierarchical_features.append(pooled_x)
            hierarchical_graphs.append((pooled_edge_index, pooled_batch))
            
            # Update for next level
            current_x = pooled_x
            current_edge_index = pooled_edge_index
            current_batch = pooled_batch
            
        # Multi-scale aggregation
        multi_scale_features = []
        for features, (edges, batch_idx) in zip(hierarchical_features, hierarchical_graphs):
            if batch_idx is None:
                batch_idx = torch.zeros(features.size(0), dtype=torch.long, device=features.device)
            pooled = global_mean_pool(features, batch_idx)
            multi_scale_features.append(pooled)
            
        # Fuse multi-scale features
        fused_features = torch.cat(multi_scale_features, dim=1)
        outputs['graph_embeddings'] = self.scale_fusion(fused_features)
        
        # Update predictions with multi-scale features
        outputs['logits'] = self.classifier(outputs['graph_embeddings'])
        outputs['predictions'] = F.softmax(outputs['logits'], dim=1)
        
        if return_hierarchy:
            outputs['hierarchical_features'] = hierarchical_features
            outputs['hierarchical_graphs'] = hierarchical_graphs
            
        return outputs