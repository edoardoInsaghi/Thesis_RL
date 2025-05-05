import torch.nn as nn
import torch

class PPO_Buffer:
    def __init__(self, 
                 buffer_size: int, 
                 gamma: float=0.99,
                 l: float=0.95,
                 epsilon_clamp: float=0.2, 
                 critic_loss_coeff: float=0.5,
                 entropy_loss_coeff: float=0.1,
                 ppo_training_steps: int=4):

        self.buffer_size = buffer_size
        self.gamma = gamma
        self.l = l
        self.epsilon_clamp = epsilon_clamp
        self.critic_loss_coeff = critic_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.ppo_training_steps = ppo_training_steps
        self.states = []
        self.actions = []
        self.log_policies = []
        self.rewards = []
        self.values = []
        self.advantages = []
        self.returns = []
        self.dones = []

    def add(self, 
            state: torch.Tensor, 
            action_idx: torch.Tensor,
            log_policy: torch.Tensor, 
            reward: torch.Tensor, 
            value: torch.Tensor, 
            done: bool):

        if len(self.states) >= self.buffer_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.log_policies.pop(0)
            self.rewards.pop(0)
            self.values.pop(0)
            self.dones.pop(0)

        self.states.append(state.detach())
        self.actions.append(action_idx.detach())
        self.log_policies.append(log_policy.detach())
        self.rewards.append(reward.detach())
        self.values.append(value.detach())
        self.dones.append(done)


    def compute_returns(self, last_value: torch.Tensor):

        rewards = torch.stack(self.rewards)
        values = torch.stack(self.values + [last_value])
        dones = self.dones

        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.l * gae * (1 - dones[t])
            advantages.insert(0, gae)

        advantages = torch.stack(advantages)
        self.returns = (advantages + values[:-1]).detach()
        self.advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-8).detach()
        

    def sample_minibatches(self, 
                           batch_size: int,
                           shuffle: bool=True,
                           device=torch.device("cpu")):

        if len(self.states) < batch_size:
            raise ValueError(f"Not enough samples in buffer. Current size: {len(self.states)}, required: {batch_size}")

        states_tensor = torch.stack(self.states).to(device)
        actions_tensor = torch.stack(self.actions).to(device)
        log_policies_tensor = torch.stack(self.log_policies).to(device)
        advantages_tensor = self.advantages.to(device)
        returns_tensor = self.returns.to(device)

        idxs = torch.randperm(len(self.states)) if shuffle else torch.arange(len(self.states))

        states_tensor = states_tensor[idxs]
        actions_tensor = actions_tensor[idxs]
        log_policies_tensor = log_policies_tensor[idxs]
        advantages_tensor = advantages_tensor[idxs]
        returns_tensor = returns_tensor[idxs]

        batches = []
        for i in range(0, len(self.states), batch_size):
            batch = {
                "states": states_tensor[i:i + batch_size],
                "actions": actions_tensor[i:i + batch_size],
                "log_policies": log_policies_tensor[i:i + batch_size],
                "advantages": advantages_tensor[i:i + batch_size],
                "returns": returns_tensor[i:i + batch_size]
            }
            batches.append(batch)

        return batches
    

    def clear(self):
        self.states = []
        self.actions = []
        self.log_policies = []
        self.rewards = []
        self.values = []
        self.advantages = []
        self.returns = []
        self.dones = []



class Agent1D(nn.Module):

    def __init__(self, n_dim: int, n_hidden: int = 128, temp_memory: int = 4, device: torch.device = torch.device("cpu")):
        super(Agent1D, self).__init__()

        self.temp_memory = temp_memory * 2
        self.memory_buffer = torch.zeros(self.temp_memory, device=device)

        self.n_dim = n_dim

        self.linear1 = nn.Linear(self.temp_memory, n_hidden)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.act2 = nn.ReLU()
        self.linear3 = nn.Linear(n_hidden, n_hidden)
        self.act3 = nn.ReLU()

        self.policy_head = nn.Linear(n_hidden, 3)
        self.softmax = nn.Softmax(dim=-1)

        self.value_head = nn.Linear(n_hidden, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.device = device
        self.to(device)

    

    def forward(self, x):

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


    @torch.no_grad()
    def act(self):
        self.eval()
        return self.forward(self.memory_buffer)
    
    def reset_memory(self):
        self.memory_buffer = torch.zeros(self.temp_memory, device=self.device)


    def update(self, buffer: PPO_Buffer, last_value: torch.Tensor, batch_size: int=32):
        
        self.train()
        buffer.compute_returns(last_value)

        avg_critic_loss = 0
        avg_actor_loss = 0
        avg_entropy = 0

        for _ in range(buffer.ppo_training_steps):
            for batch in buffer.sample_minibatches(batch_size=batch_size, shuffle=True, device=self.device):

                states = batch["states"]
                actions = batch["actions"]
                log_policies = batch["log_policies"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                new_policy, new_value = self.forward(states)
                new_log_policies = torch.log(new_policy.gather(1, actions))

                ratios = torch.exp(new_log_policies - log_policies)
                clipped_ratios = torch.clamp(ratios, 1-buffer.epsilon_clamp, 1+buffer.epsilon_clamp)

                actor_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()
                critic_loss = torch.nn.functional.mse_loss(new_value.squeeze(), returns.squeeze())
                entropy_loss = torch.sum(new_policy * torch.log(new_policy), dim=1).mean()

                loss = actor_loss + buffer.critic_loss_coeff * critic_loss + buffer.entropy_loss_coeff * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                self.optimizer.step()

                avg_critic_loss += critic_loss.item()
                avg_actor_loss += actor_loss.item()
                avg_entropy += -entropy_loss.item()

        buffer.advantages = []
        buffer.returns = []

        avg_critic_loss /= (len(buffer.states) * buffer.ppo_training_steps)
        avg_actor_loss /= (len(buffer.states) * buffer.ppo_training_steps)
        avg_entropy /= (len(buffer.states) * buffer.ppo_training_steps)

        return avg_critic_loss, avg_actor_loss, avg_entropy