import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, temp_memory, num_actions, n_agents, device):
        super(MLP, self).__init__()

        self.device = device
        self.temp_memory = temp_memory
        self.num_actions = num_actions
        self.n_agents = n_agents
        
        # Batched memory: (n_agents, temp_memory * 3)
        self.register_buffer('memory_buffer', torch.zeros((n_agents, temp_memory * 3), device=device))
        
        self.backbone = nn.Sequential(
            nn.Linear(temp_memory * 3, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU()
        )
        self.policy_head = nn.Linear(128, num_actions)
        self.value_head = nn.Linear(128, 1)
        self.softmax = nn.Softmax(dim=-1)

    def update_memory(self, action_idxs: torch.Tensor, rewards: torch.Tensor):
        # memory_buffer: (n_agents, 3*temp_memory)
        # action_idxs: (n_agents, 2)
        # rewards: (n_agents,)
        
        self.memory_buffer[:, :-3] = self.memory_buffer[:, 3:].clone()
        self.memory_buffer[:,-3] = action_idxs[:,0]
        self.memory_buffer[:,-2] = action_idxs[:,1]
        self.memory_buffer[:,-1] = rewards

    def reset_memory(self):
        self.memory_buffer.zero_()

    def forward(self, x=None):
        if x is None:
            x = self.memory_buffer
        x = self.backbone(x)
        policy = self.softmax(self.policy_head(x))
        value = self.value_head(x)
        return policy, value


class Transformer(nn.Module):
    def __init__(self, temp_memory, num_actions, n_agents, device):
        super(Transformer, self).__init__()
        self.device = device
        self.temp_memory = temp_memory
        self.num_actions = num_actions
        self.n_agents = n_agents
        
        # Batched memory: (n_agents, temp_memory, 2)
        self.register_buffer('memory_buffer', 
                           torch.zeros((n_agents, temp_memory, 2), device=device))
        
        # Transformer backbone
        self.embedding = nn.Linear(2, 32)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=128),
            num_layers=3
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Heads
        self.policy_head = nn.Linear(32, num_actions)
        self.value_head = nn.Linear(32, 1)
        self.softmax = nn.Softmax(dim=-1)

    def update_memory(self, action_idxs: torch.Tensor, rewards: torch.Tensor):
        # Shift memory
        self.memory_buffer[:, :-1] = self.memory_buffer[:, 1:].clone()
        
        # Update new experiences
        self.memory_buffer[:, -1, 0] = action_idxs.float()
        self.memory_buffer[:, -1, 1] = rewards

    def reset_memory(self):
        self.memory_buffer.zero_()

    def forward(self, x=None):
        if x is None:
            x = self.memory_buffer
        
        # x shape: (batch_size, seq_len, 2)
        batch_size, seq_len, _ = x.shape
        x = self.embedding(x)  # (batch_size, seq_len, 32)
        
        # Transformer expects (seq_len, batch_size, features)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, 32)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, 32)
        
        # Global average pooling
        x = self.pool(x.permute(0, 2, 1))  # (batch_size, 32, 1)
        x = x.squeeze(-1)  # (batch_size, 32)
        
        policy = self.softmax(self.policy_head(x))
        value = self.value_head(x)
        return policy, value


NNDICT = {
    "mlp": MLP,
    "transformer": Transformer
}