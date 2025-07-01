import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, temp_memory, num_actions, device):
        super(MLP, self).__init__()

        self.device = device
        self.temp_memory = temp_memory * 2  # (action, reward) pairs
        self.num_actions = num_actions
        
        self.memory_buffer = torch.zeros(self.temp_memory).to(self.device)
        
        self.backbone = nn.Sequential(
            nn.Linear(self.temp_memory, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU()
        )
        self.policy_head = nn.Linear(128, num_actions)
        self.value_head = nn.Linear(128, 1)
        self.softmax = nn.Softmax(dim=-1)

    def update_memory(self, action_idx: torch.Tensor, reward: torch.Tensor):
        self.memory_buffer[:-2] = self.memory_buffer[2:].clone()
        self.memory_buffer[-2] = action_idx
        self.memory_buffer[-1] = reward

    def reset_memory(self):
        self.memory_buffer = torch.zeros(self.temp_memory).to(self.device)

    def forward(self, x):
        x = self.backbone(x)
        policy = self.softmax(self.policy_head(x))
        value = self.value_head(x)
        return policy, value


class Transformer(nn.Module):
    def __init__(self, temp_memory, num_actions, device):
        super(Transformer, self).__init__()
        self.device = device
        self.temp_memory = temp_memory
        self.num_actions = num_actions
        
        # Initialize memory buffer
        self.memory_buffer = torch.zeros((temp_memory, 2)).to(self.device)
        
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

    def update_memory(self, action_idx: torch.Tensor, reward: torch.Tensor):
        # Shift memory and add new (action, reward) pair
        self.memory_buffer[:-1] = self.memory_buffer[1:].clone()
        self.memory_buffer[-1, 0] = action_idx
        self.memory_buffer[-1, 1] = reward

    def reset_memory(self):
        self.memory_buffer = torch.zeros((self.temp_memory, 2)).to(self.device)

    def forward(self, x):
        # x shape: (seq_len, 2)
        x = self.embedding(x)  # (seq_len, 32)
        x = x.permute(1, 0, 2)  # (seq_len, batch, features) -> (batch, seq_len, features)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # (batch, seq_len, features) -> (seq_len, batch, features)
        x = self.pool(x.permute(1, 2, 0)).squeeze(-1)  # Global average pooling
        
        policy = self.softmax(self.policy_head(x))
        value = self.value_head(x)
        return policy, value


NNDICT = {
    "mlp": MLP,
    "transformer": Transformer
}