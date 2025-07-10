import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data

class GNNPolicy(nn.Module):
    def __init__(self, temp_memory, num_actions, device, 
                 node_dim=128, edge_dim=2, num_passes=3):
        super().__init__()
        self.device = device
        self.temp_memory = temp_memory
        self.num_actions = num_actions
        self.num_passes = num_passes
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(temp_memory * 3, 256),
            nn.ReLU(),
            nn.Linear(256, node_dim),
            nn.ReLU()
        )
        
        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, node_dim),
            nn.ReLU()
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_passes):
            self.gnn_layers.append(GATv2Conv(node_dim, node_dim, edge_dim=node_dim))
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(node_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(node_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, node_features, edge_index, edge_attr):
        # Encode node features
        node_emb = self.node_encoder(node_features)
        
        # Encode edge features
        edge_emb = self.edge_encoder(edge_attr)
        
        # Create PyG data object
        data = Data(x=node_emb, edge_index=edge_index, edge_attr=edge_emb)
        
        # Process through GNN layers
        x = data.x
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, data.edge_index, data.edge_attr)
            x = F.relu(x)
        
        # Generate outputs
        policies = F.softmax(self.policy_head(x), dim=-1)
        values = self.value_head(x)
        
        return policies, values