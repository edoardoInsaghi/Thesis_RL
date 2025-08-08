import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data

class GNNPolicy(nn.Module):
    def __init__(self, temp_memory, num_actions, device, 
                 node_dim=128, edge_dim=64, num_passes=3):
        super().__init__()

        self.device = device
        self.temp_memory = temp_memory
        self.num_actions = num_actions
        self.num_passes = num_passes
        
        self.node_encoder = nn.Sequential(
            nn.Linear(temp_memory * 3, node_dim*2),
            nn.ReLU(),
            nn.LayerNorm(node_dim*2),
            nn.Linear(node_dim*2, node_dim),
            nn.ReLU(),
            nn.LayerNorm(node_dim)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(2, edge_dim),
            nn.ReLU(),
            nn.LayerNorm(edge_dim),
            nn.Linear(edge_dim, node_dim),
            nn.ReLU(), 
            nn.LayerNorm(node_dim)
        )

        self.norm_layers = nn.ModuleList(
            [nn.LayerNorm(node_dim) for _ in range(num_passes)]
        ) 
        self.gnn_layers = nn.ModuleList(
            [GATv2Conv(node_dim, node_dim, edge_dim=node_dim) for _ in range(num_passes)]
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(node_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(node_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, node_features, edge_index, edge_attr):

        node_emb = self.node_encoder(node_features)
        edge_emb = self.edge_encoder(edge_attr)

        data = Data(x=node_emb, edge_index=edge_index, edge_attr=edge_emb)

        x = data.x
        for i, gnn_layer in enumerate(self.gnn_layers):
            residual = x
            x = gnn_layer(x, data.edge_index, data.edge_attr)
            x = F.relu(x)
            x = self.norm_layers[i](x)
            x = x + residual
        
        policies = F.softmax(self.policy_head(x), dim=-1)
        values = self.value_head(x)
        
        return policies, values