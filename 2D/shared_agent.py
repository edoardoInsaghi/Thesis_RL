import torch
import torch.nn as nn
import torch.nn.functional as F
from shared_net import NNDICT

class PPO_Buffer:
    def __init__(self, 
                 gamma: float = 0.99,
                 l: float = 0.95,
                 epsilon_clamp: float = 0.2, 
                 critic_loss_coeff: float = 0.5,
                 entropy_loss_coeff: float = 0.1,
                 ppo_training_steps: int = 4,
                 n_agents: int=5):
        
        self.gamma = gamma
        self.l = l
        self.epsilon_clamp = epsilon_clamp
        self.critic_loss_coeff = critic_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.ppo_training_steps = ppo_training_steps
        self.n_agents = n_agents
        
        self.states = []
        self.actions = []
        self.log_policies = []
        self.rewards = []
        self.values = []
        self.advantages = []
        self.returns = []
        self.dones = []

    def add_batch(self, 
                 states: torch.Tensor, 
                 action_idxs: torch.Tensor,
                 log_policies: torch.Tensor, 
                 rewards: torch.Tensor, 
                 values: torch.Tensor, 
                 dones: torch.Tensor):
        
        # Add batch of experiences from all agents
        self.states.extend(states.detach().unbind(0))
        self.actions.extend(action_idxs.detach().unbind(0))
        self.log_policies.extend(log_policies.detach().unbind(0))
        self.rewards.extend(rewards.detach().unbind(0))
        self.values.extend(values.detach().unbind(0))
        self.dones.extend(dones.detach().unbind(0))

    def compute_returns_2(self, last_values: torch.Tensor):
        # Stack all experiences
        rewards = torch.stack(self.rewards)
        values = torch.stack(self.values + last_values)
        dones = torch.tensor(self.dones, dtype=torch.float32)
        
        advantages = []
        gae = 0
        
        # Compute GAE
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.l * gae * (1 - dones[t])
            advantages.insert(0, gae)
        
        advantages = torch.stack(advantages)
        self.returns = (advantages + values[:-5]).detach()
        self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8).detach()
    
    def compute_returns(self, last_values: torch.Tensor):

        # Convert to tensors
        rewards = torch.stack(self.rewards)
        values = torch.stack(self.values)
        dones = torch.tensor(self.dones, dtype=torch.float32)
        
        # Group by agent and time
        T = len(self.rewards) // self.n_agents
        rewards = rewards.view(T, self.n_agents)
        values = values.view(T, self.n_agents)
        dones = dones.view(T, self.n_agents)
        
        next_values = torch.cat((values[1:], last_values))
        
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute per-agent GAE
        for i in range(self.n_agents):
            agent_advantages = []
            gae = 0
            
            # Process in reverse time order
            for t in reversed(range(T)):
                done_mask = 1 - dones[t, i]
                delta = rewards[t, i] + self.gamma * next_values[t, i] * done_mask - values[t, i]
                gae = delta + self.gamma * self.l * gae * done_mask
                agent_advantages.insert(0, gae)
            
            advantages[:, i] = torch.stack(agent_advantages)
        
        # Normalize advantages across all agents
        advantages_flat = advantages.view(-1)
        advantages_normalized = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
        
        # Compute returns
        returns = advantages + values.to("cpu")
        
        # Store results
        self.advantages = advantages_normalized.detach()
        self.returns = returns.view(-1).detach()
        

    def sample_minibatches(self, batch_size: int, shuffle: bool = True, device=torch.device("cpu")):
        
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
                "returns": returns_tensor[i:i + batch_size],
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


class Agent(nn.Module):
    def __init__(self,
                 net: str, 
                 temp_memory: int, 
                 num_actions: int,
                 n_agents: int,
                 device: torch.device = torch.device("cpu"),
                 weights: str = None):
        super(Agent, self).__init__()
        
        self.temp_memory = temp_memory
        self.num_actions = num_actions
        self.n_agents = n_agents
        self.device = device
        
        self.network = NNDICT[net](
            temp_memory=temp_memory, 
            num_actions=num_actions,
            n_agents=n_agents,
            device=device
        )
        
        if weights is not None:
            self.load_state_dict(torch.load(weights, map_location=device))
            print(f"Weights loaded from {weights}")
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.to(device)
    
    @torch.no_grad()
    def act(self):
        self.network.eval()
        policy, value = self.network.forward()
        return policy.detach(), value.detach()
    
    def reset_memory(self):
        self.network.reset_memory()
    
    def update_memory(self, action_idxs: torch.Tensor, rewards: torch.Tensor):
        self.network.update_memory(action_idxs, rewards)
    
    def update(self, buffer: PPO_Buffer, last_value: torch.Tensor, batch_size: int=32, shuffle=True):
        self.network.train()
        buffer.compute_returns(last_value)
        
        avg_critic_loss = 0
        avg_actor_loss = 0
        avg_entropy = 0
        total_steps = 0
        
        for _ in range(buffer.ppo_training_steps):
            for batch in buffer.sample_minibatches(batch_size=batch_size, shuffle=shuffle, device=self.device):

                states = batch["states"]
                actions = batch["actions"]
                old_log_policies = batch["log_policies"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                
                # Forward pass
                new_policy, new_value = self.network.forward(states)
                new_log_policy = torch.log(new_policy.gather(1, actions.unsqueeze(1)).squeeze())
                
                ratio = torch.exp(new_log_policy - old_log_policies)
                clipped_ratio = torch.clamp(ratio, 1-buffer.epsilon_clamp, 1+buffer.epsilon_clamp)
                
                actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
                critic_loss = F.mse_loss(new_value.squeeze(), returns)
                entropy_loss = torch.sum(new_policy * torch.log(new_policy), dim=1).mean()
                
                loss = actor_loss + buffer.critic_loss_coeff * critic_loss + buffer.entropy_loss_coeff * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                self.optimizer.step()
                
                avg_critic_loss += critic_loss.item()
                avg_actor_loss += actor_loss.item()
                avg_entropy += -entropy_loss.item()
                total_steps += 1
        
        if total_steps > 0:
            avg_critic_loss /= total_steps
            avg_actor_loss /= total_steps
            avg_entropy /= total_steps
        
        buffer.clear()
        
        return avg_critic_loss, avg_actor_loss, avg_entropy
    
    def save_model(self, path: str):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")