import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from graph_net import GNNPolicy
from torch_geometric.data import Data

class PPO_Buffer:
    def __init__(self, gamma=0.99, l=0.95, epsilon_clamp=0.2, 
                 critic_loss_coeff=0.5, entropy_loss_coeff=0.1, ppo_training_steps=4):
        self.gamma = gamma
        self.l = l
        self.epsilon_clamp = epsilon_clamp
        self.critic_loss_coeff = critic_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.ppo_training_steps = ppo_training_steps
        
        # Store graph states
        self.node_features = []    # Agent memories
        self.edge_indices = []     # Graph connectivity
        self.edge_attrs = []       # Edge features
        self.actions = []          # Actions per agent
        self.log_policies = []     # Log probs per agent
        self.rewards = []          # Rewards per agent
        self.values = []           # Values per agent
        self.dones = []            # Done flags
        
        self.advantages = []
        self.returns = []

    def add_timestep(self, 
                     node_features, 
                     edge_index,
                     edge_attr,
                     actions,
                     log_policies,
                     rewards,
                     values,
                     done):
        
        self.node_features.append(node_features.detach().clone())
        self.edge_indices.append(edge_index.detach().clone())
        self.edge_attrs.append(edge_attr.detach().clone())
        self.actions.append(actions.detach().clone())
        self.log_policies.append(log_policies.detach().clone())
        self.rewards.append(rewards.detach().clone())
        self.values.append(values.detach().clone())
        self.dones.append(done)

    def compute_returns(self, last_values):
        if not self.rewards:
            return
            
        # Convert to tensors
        rewards = torch.stack(self.rewards)           # [T, n_agents]
        values = torch.stack(self.values)             # [T, n_agents]
        dones = torch.tensor(self.dones, dtype=torch.float32)  # [T]
        
        # Add last_values for final step
        next_values = torch.cat([values[1:], last_values.unsqueeze(0)])
        
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        n_agents = rewards.shape[1]
        
        # Compute per-agent GAE
        for i in range(n_agents):
            agent_advantages = []
            gae = 0
            
            # Process in reverse time order
            for t in reversed(range(len(rewards))):
                done_mask = 1 - dones[t]
                delta = rewards[t, i] + self.gamma * next_values[t, i] * done_mask - values[t, i]
                gae = delta + self.gamma * self.l * gae * done_mask
                agent_advantages.insert(0, gae)
            
            advantages[:, i] = torch.stack(agent_advantages)
        
        # Normalize advantages
        advantages_flat = advantages.view(-1)
        advantages_normalized = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
        
        # Compute returns
        returns = advantages + values.to("cpu")
        
        # Store results
        self.advantages = advantages_normalized.detach()
        self.returns = returns.view(-1).detach()

    def sample_minibatches(self, batch_size, device):
        if len(self.node_features) < batch_size:
            raise ValueError("Not enough samples in buffer")
        
        # Create random indices
        idxs = torch.randperm(len(self.node_features))
        
        batches = []
        for i in range(0, len(self.node_features), batch_size):
            batch_idxs = idxs[i:i + batch_size]
            
            # Prepare graph batch
            data_list = []
            for idx in batch_idxs:
                data = Data(
                    x=self.node_features[idx],
                    edge_index=self.edge_indices[idx],
                    edge_attr=self.edge_attrs[idx]
                )
                data_list.append(data)
            graph_batch = Batch.from_data_list(data_list)
            
            # Prepare other components
            actions_batch = torch.cat([self.actions[idx] for idx in batch_idxs])
            log_policies_batch = torch.cat([self.log_policies[idx] for idx in batch_idxs])
            advantages_batch = self.advantages.view(len(self.node_features), -1)[batch_idxs].view(-1)
            returns_batch = self.returns.view(len(self.node_features), -1)[batch_idxs].view(-1)
            
            batch = {
                "graph_batch": graph_batch.to(device),
                "actions": actions_batch.to(device),
                "log_policies": log_policies_batch.to(device),
                "advantages": advantages_batch.to(device),
                "returns": returns_batch.to(device),
            }
            batches.append(batch)
            
        return batches
    
    def clear(self):
        self.node_features = []
        self.edge_indices = []
        self.edge_attrs = []
        self.actions = []
        self.log_policies = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []


class Agent(nn.Module):
    def __init__(self, temp_memory, num_actions, device, weights=None):
        super().__init__()
        self.device = device
        self.temp_memory = temp_memory
        self.num_actions = num_actions
        
        # Network
        self.network = GNNPolicy(
            temp_memory=temp_memory,
            num_actions=num_actions,
            device=device
        )
        
        # Agent memory buffer
        self.memory_buffer = None
        
        if weights:
            self.load_state_dict(torch.load(weights, map_location=device))
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.to(device)
    
    def init_memory(self, n_agents):
        """Initialize memory for n_agents"""
        self.memory_buffer = torch.zeros((n_agents, self.temp_memory * 3), device=self.device)
    
    @torch.no_grad()
    def act(self, edge_index, edge_attr):
        self.network.eval()
        policy, value = self.network.forward(self.memory_buffer, edge_index, edge_attr)
        return policy.detach(), value.detach()
    
    def reset_memory(self):
        if self.memory_buffer is not None:
            self.memory_buffer.zero_()
    
    def update_memory(self, action_reprs, rewards):
        
        self.memory_buffer[:, :-3] = self.memory_buffer[:, 3:].clone()
        self.memory_buffer[:,-3] = action_reprs[:,0]
        self.memory_buffer[:,-2] = action_reprs[:,1]
        self.memory_buffer[:,-1] = rewards
    
    def update(self, buffer, batch_size=32):
        self.network.train()
        
        critic_loss_total = 0
        actor_loss_total = 0
        entropy_total = 0
        total_steps = 0
        
        for _ in range(buffer.ppo_training_steps):
            for batch in buffer.sample_minibatches(batch_size, self.device):

                graph_batch = batch["graph_batch"]
                actions = batch["actions"]
                old_log_policies = batch["log_policies"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                
                new_policy, new_value = self.network(
                    graph_batch.x, 
                    graph_batch.edge_index, 
                    graph_batch.edge_attr
                )
                
                new_log_policy = torch.log(new_policy.gather(1, actions.unsqueeze(1))).squeeze()
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
                
                critic_loss_total += critic_loss.item()
                actor_loss_total += actor_loss.item()
                entropy_total += -entropy_loss.item()
                total_steps += 1
        
        critic_loss_avg = critic_loss_total / total_steps
        actor_loss_avg = actor_loss_total / total_steps
        entropy_avg = entropy_total / total_steps
        
        buffer.clear()
        return critic_loss_avg, actor_loss_avg, entropy_avg
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")