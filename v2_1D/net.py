import torch
import torch.nn as nn

# input is sequence of temp_memory (action, reward) pairs, shape is (temp_memory * 2)
class MLP(nn.Module):
    def __init__(self, temp_memory, device):
        super(MLP, self).__init__()

        self.device = device
        self.temp_memory = temp_memory * 2
        self.memory_buffer = torch.zeros(self.temp_memory).to(self.device)

        self.backbone = nn.Sequential(
            nn.Linear(self.temp_memory, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU()
        )
        self.policy_head = nn.Linear(128, 3)
        self.value_head = nn.Linear(128, 1)
        self.softmax = nn.Softmax(dim=-1)

    def update_memory(self, action: torch.Tensor, reward: torch.Tensor):
        self.memory_buffer[:-2] = self.memory_buffer[2:].clone()
        self.memory_buffer[-2] = action
        self.memory_buffer[-1] = reward

    def reset_memory(self):
        self.memory_buffer = torch.zeros(self.temp_memory).to(self.device)

    def forward(self, x):

        x = self.backbone(x)
        policy = self.softmax(self.policy_head(x))
        value = self.value_head(x)
        return policy, value
    

# input is sequence of temp_memory (action, reward) pairs, shape is (temp_memory, 2)
class Transformer(nn.Module):
    def __init__(self, temp_memory, device):
        super(Transformer, self).__init__()

        self.device=device
        self.temp_memory = temp_memory
        self.memory_buffer = torch.zeros(temp_memory, 2).to(self.device)

        self.backbone = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=2, nhead=2), num_layers=2
        )
        self.policy_head = nn.Linear(2, 3)
        self.value_head = nn.Linear(2, 1)
        self.softmax = nn.Softmax(dim=-1)

    def update_memory(self, action: torch.Tensor, reward: torch.Tensor):
        self.memory_buffer[:-1] = self.memory_buffer[1:].clone()
        self.memory_buffer[-1] = torch.tensor([action, reward])

    def reset_memory(self):
        self.memory_buffer = torch.zeros(self.temp_memory, 2).to(self.device)

    def forward(self, x):
        x = self.backbone(x.unsqueeze(0)).squeeze(0)
        policy = self.softmax(self.policy_head(x))
        value = self.value_head(x)
        return policy, value
    

NNDICT = {
    "mlp": MLP,
    "transformer": Transformer
}
