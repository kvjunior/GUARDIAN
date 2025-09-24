"""
Advanced neural network layers for GUARDIAN
Implementation of theoretically-grounded graph neural network components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import softmax, degree, add_self_loops
from torch_scatter import scatter_add, scatter_mean
from typing import Optional, Tuple, Union
import math

class EnhancedMGD(MessagePassing):
    """
    Enhanced Multigraph Discrepancy layer with theoretical guarantees
    
    Mathematical Foundation:
    For node v with neighbors N(v), the discrepancy-aware message passing is:
    h_v^(l+1) = σ(W_self * h_v^(l) + Σ_{u∈N_in(v)} α_vu * W_in * [h_u^(l) || (h_v^(l) - h_u^(l))]
                                    + Σ_{u∈N_out(v)} β_vu * W_out * [h_u^(l) || (h_v^(l) - h_u^(l))])
    
    where α_vu and β_vu are learned attention weights, and || denotes concatenation.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 8,
        dropout: float = 0.2,
        layer_idx: int = 0,
        negative_slope: float = 0.2
    ):
        super(EnhancedMGD, self).__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.layer_idx = layer_idx
        self.negative_slope = negative_slope
        
        # Multi-head attention mechanism
        self.head_dim = out_channels // num_heads
        assert self.head_dim * num_heads == out_channels
        
        # Separate transformations for incoming and outgoing edges
        self.W_in = nn.Linear(in_channels * 2, out_channels, bias=False)
        self.W_out = nn.Linear(in_channels * 2, out_channels, bias=False)
        self.W_self = nn.Linear(in_channels, out_channels, bias=False)
        
        # Attention parameters
        self.att_in = nn.Parameter(torch.Tensor(1, num_heads, 2 * self.head_dim))
        self.att_out = nn.Parameter(torch.Tensor(1, num_heads, 2 * self.head_dim))
        
        # Learnable temperature for attention
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Layer-specific parameters for adaptive weighting
        self.layer_weight = nn.Parameter(torch.ones(3))
        
        # Bias terms
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        # Batch normalization for stability
        self.batch_norm = nn.BatchNorm1d(out_channels)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters with careful consideration for stability"""
        # Xavier initialization for linear layers
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_out.weight)
        nn.init.xavier_uniform_(self.W_self.weight)
        
        # Initialize attention parameters
        nn.init.xavier_uniform_(self.att_in)
        nn.init.xavier_uniform_(self.att_out)
        
        # Initialize bias to zero
        nn.init.zeros_(self.bias)
        
        # Initialize layer weights uniformly
        nn.init.ones_(self.layer_weight)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with discrepancy-aware message passing
        """
        
        # Add self-loops for stability
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Separate incoming and outgoing edges
        row, col = edge_index
        
        # Create reverse edges for incoming direction
        edge_index_reverse = torch.stack([col, row], dim=0)
        
        # Transform node features
        x_self = self.W_self(x)
        
        # Message passing for incoming edges
        out_in = self.propagate(
            edge_index_reverse, x=x, 
            transformer=self.W_in, attention=self.att_in,
            direction='incoming'
        )
        
        # Message passing for outgoing edges
        out_out = self.propagate(
            edge_index, x=x,
            transformer=self.W_out, attention=self.att_out,
            direction='outgoing'
        )
        
        # Adaptive layer weighting
        weights = F.softmax(self.layer_weight, dim=0)
        out = weights[0] * x_self + weights[1] * out_in + weights[2] * out_out
        
        # Add bias and apply batch normalization
        out = out + self.bias
        out = self.batch_norm(out)
        
        if return_attention:
            # Compute attention weights for visualization
            with torch.no_grad():
                alpha_in = self._compute_attention_weights(
                    x, edge_index_reverse, self.att_in
                )
                alpha_out = self._compute_attention_weights(
                    x, edge_index, self.att_out
                )
            return out, (alpha_in, alpha_out)
        
        return out
        
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        transformer: nn.Module,
        attention: torch.Tensor,
        direction: str,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int]
    ) -> torch.Tensor:
        """
        Compute discrepancy-aware messages with attention
        """
        
        # Compute discrepancy features
        discrepancy = x_i - x_j
        
        # Concatenate neighbor features with discrepancy
        message_input = torch.cat([x_j, discrepancy], dim=-1)
        
        # Transform messages
        messages = transformer(message_input)
        
        # Reshape for multi-head attention
        messages = messages.view(-1, self.num_heads, self.head_dim)
        x_i_heads = x_i.view(-1, self.num_heads, self.head_dim)
        
        # Compute attention scores
        alpha = (torch.cat([x_i_heads, messages], dim=-1) * attention).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha / self.temperature, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Apply attention to messages
        messages = messages * alpha.unsqueeze(-1)
        
        return messages.view(-1, self.out_channels)
        
    def _compute_attention_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        attention: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention weights for analysis
        """
        row, col = edge_index
        x_i = x[row]
        x_j = x[col]
        
        # Compute discrepancy
        discrepancy = x_i - x_j
        message_input = torch.cat([x_j, discrepancy], dim=-1)
        
        # Get attention scores
        x_i_heads = x_i.view(-1, self.num_heads, self.head_dim)
        messages = message_input.view(-1, self.num_heads, self.head_dim * 2)
        
        alpha = (torch.cat([x_i_heads, messages[:, :, :self.head_dim]], dim=-1) * attention).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha / self.temperature, row)
        
        return alpha.mean(dim=1)  # Average over heads

class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer layer with edge feature incorporation
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5
    ):
        super(GraphTransformerLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Multi-head attention components
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
        # Edge feature projection
        self.W_e = nn.Linear(hidden_dim, num_heads)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of graph transformer layer
        """
        
        # Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_out = self._attention(x_norm, edge_index, edge_attr)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual connection
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)
        
        return x
        
    def _attention(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Multi-head attention computation
        """
        
        # Linear transformations
        Q = self.W_q(x).view(-1, self.num_heads, self.head_dim)
        K = self.W_k(x).view(-1, self.num_heads, self.head_dim)
        V = self.W_v(x).view(-1, self.num_heads, self.head_dim)
        
        # Compute attention scores
        row, col = edge_index
        Q_i = Q[row]  # [num_edges, num_heads, head_dim]
        K_j = K[col]  # [num_edges, num_heads, head_dim]
        V_j = V[col]  # [num_edges, num_heads, head_dim]
        
        # Scaled dot-product attention
        scores = (Q_i * K_j).sum(dim=-1) / math.sqrt(self.head_dim)
        
        # Include edge features if available
        if edge_attr is not None:
            edge_scores = self.W_e(edge_attr)  # [num_edges, num_heads]
            scores = scores + edge_scores
            
        # Apply softmax
        alpha = softmax(scores, row, num_nodes=x.size(0))
        alpha = self.dropout(alpha)
        
        # Weighted aggregation
        out = V_j * alpha.unsqueeze(-1)
        out = scatter_add(out, row, dim=0, dim_size=x.size(0))
        
        # Reshape and final projection
        out = out.view(-1, self.hidden_dim)
        out = self.W_o(out)
        
        return out

class TemporalAttentionLayer(nn.Module):
    """
    Temporal attention mechanism for transaction sequences
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super(TemporalAttentionLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Temporal encoding
        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention components
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply temporal attention based on transaction timestamps
        """
        
        # Encode temporal information
        time_features = self.time_encoder(timestamps.unsqueeze(-1))
        
        # Combine with node features
        x_temporal = x + time_features
        
        # Compute attention
        Q = self.W_q(x_temporal).view(-1, self.num_heads, self.head_dim)
        K = self.W_k(x_temporal).view(-1, self.num_heads, self.head_dim)
        V = self.W_v(x_temporal).view(-1, self.num_heads, self.head_dim)
        
        # Self-attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Temporal masking (optional: mask future transactions)
        # This can be implemented based on specific requirements
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Weighted aggregation
        out = torch.matmul(attn_weights, V)
        out = out.view(-1, self.hidden_dim)
        out = self.W_o(out)
        
        return out

class HierarchicalPooling(nn.Module):
    """
    Hierarchical graph pooling with learnable scoring
    """
    
    def __init__(self, hidden_dim: int, ratio: float = 0.5):
        super(HierarchicalPooling, self).__init__()
        
        self.ratio = ratio
        
        # Scoring network
        self.score_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Feature transformation
        self.feature_transform = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform hierarchical pooling
        """
        
        # Compute scores for each node
        scores = self.score_layer(x).squeeze(-1)
        
        # Determine number of nodes to keep
        num_nodes = x.size(0)
        num_kept = max(1, int(num_nodes * self.ratio))
        
        # Select top-k nodes
        _, perm = torch.topk(scores, num_kept, sorted=True)
        
        # Transform features
        x_pooled = self.feature_transform(x[perm])
        
        # Update edge index
        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)
        node_mask[perm] = True
        
        # Map old indices to new indices
        new_index = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
        new_index[perm] = torch.arange(num_kept, device=x.device)
        
        # Filter edges
        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        edge_index_pooled = new_index[edge_index[:, edge_mask]]
        
        # Update batch assignment
        batch_pooled = batch[perm]
        
        return x_pooled, edge_index_pooled, batch_pooled, perm

class PrivacyPreservingAggregation(nn.Module):
    """
    Privacy-preserving aggregation with noise injection
    """
    
    def __init__(self, hidden_dim: int, noise_multiplier: float = 0.1):
        super(PrivacyPreservingAggregation, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.noise_multiplier = noise_multiplier
        
        # Learnable aggregation weights
        self.aggregation_weights = nn.Parameter(torch.ones(3))
        
        # Privacy-aware transformation
        self.privacy_transform = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate with privacy preservation
        """
        
        # Multiple aggregation strategies
        mean_pool = global_mean_pool(x, batch)
        max_pool = scatter_max(x, batch, dim=0)[0]
        sum_pool = scatter_add(x, batch, dim=0)
        
        # Add calibrated noise for privacy
        if self.training:
            noise_scale = self.noise_multiplier
            mean_pool = mean_pool + torch.randn_like(mean_pool) * noise_scale
            max_pool = max_pool + torch.randn_like(max_pool) * noise_scale
            sum_pool = sum_pool + torch.randn_like(sum_pool) * noise_scale
            
        # Weighted combination
        weights = F.softmax(self.aggregation_weights, dim=0)
        pooled = torch.cat([
            weights[0] * mean_pool,
            weights[1] * max_pool,
            weights[2] * sum_pool
        ], dim=1)
        
        # Final transformation
        output = self.privacy_transform(pooled)
        
        return output

class AdversarialPerturbation(nn.Module):
    """
    Learnable adversarial perturbation generation
    """
    
    def __init__(self, input_dim: int, epsilon: float = 0.1):
        super(AdversarialPerturbation, self).__init__()
        
        self.epsilon = epsilon
        
        # Perturbation generator network
        self.generator = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate bounded adversarial perturbations
        """
        
        # Generate perturbation
        delta = self.generator(x)
        
        # Scale to epsilon ball
        delta = self.epsilon * delta
        
        # Apply perturbation
        x_perturbed = x + delta
        
        return x_perturbed

def scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scatter max operation for pooling
    """
    size = list(src.size())
    size[dim] = int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    out = out.scatter_reduce(dim, index.unsqueeze(-1).expand_as(src), src, reduce='amax')
    return out, index