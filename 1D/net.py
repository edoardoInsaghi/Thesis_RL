import torch
import torch.nn as nn

class Net_MLP(nn.Module):
    
    def __init__(self, n_dim: int, n_hidden: int = 128, temp_memory: int = 4, device: torch.device = torch.device("cpu")):
        super(Net_MLP, self).__init__()

        self.temp_memory = temp_memory
        self.memory_buffer = torch.zeros(temp_memory).to(device)

        self.n_dim = n_dim

        self.linear1 = nn.Linear(n_dim + temp_memory, n_hidden)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.act2 = nn.ReLU()
        self.linear3 = nn.Linear(n_hidden, n_hidden)
        self.act3 = nn.ReLU()

        self.policy_head = nn.Linear(n_hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.value_head = nn.Linear(n_hidden, 1)
        
        self.device = device
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.to(device)


    def forward(self, x: torch.Tensor):
        x = torch.cat((x, self.memory_buffer), dim=-1)
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        x = self.linear3(x)
        x = self.act3(x)

        policy = self.policy_head(x)
        policy = self.softmax(policy)

        value = self.value_head(x)

        return policy, value