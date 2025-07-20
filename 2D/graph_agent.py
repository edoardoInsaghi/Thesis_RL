import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.data import Batch
from graph_net import GNNPolicy
from torch_geometric.data import Data
from environment import EnvArgs2D, Environment2D

class PPO_Buffer:
    def __init__(self, gamma=0.99, l=0.95, epsilon_clamp=0.2, 
                 critic_loss_coeff=0.5, entropy_loss_coeff=0.1, ppo_training_steps=4):
        self.gamma = gamma
        self.l = l
        self.epsilon_clamp = epsilon_clamp
        self.critic_loss_coeff = critic_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.ppo_training_steps = ppo_training_steps
        
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
        
        idxs = torch.randperm(len(self.node_features))
        
        batches = []
        for i in range(0, len(self.node_features), batch_size):
            batch_idxs = idxs[i:i + batch_size]
            
            data_list = []
            for idx in batch_idxs:
                data = Data(
                    x=self.node_features[idx],
                    edge_index=self.edge_indices[idx],
                    edge_attr=self.edge_attrs[idx]
                )
                data_list.append(data.to(device))
            graph_batch = Batch.from_data_list(data_list)
            
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




import torch
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from environment import EnvArgs2D, Environment2D

class MultiAgentController:
    def __init__(self, env, strategy="sticky_random", comm_radius=50.0):
        self.env = env
        self.n_agents = env.n_actors
        self.action_vectors = env.action_vectors
        self.comm_radius = comm_radius
        self.strategy = strategy
        self.reset()
        
    def reset(self):
        """Reset all agent states between episodes"""
        # Common state for all agents
        self.last_vectors = torch.zeros((self.n_agents, 2))
        self.last_rewards = torch.zeros(self.n_agents)
        self.displacements = torch.zeros((self.n_agents, 2))  # From start position
        
        # Cognitive maps: {position_hash: (reward, displacement)}
        self.cognitive_maps = [{} for _ in range(self.n_agents)]
        
        # Strategy-specific states
        if self.strategy == "sticky_random":
            self.agent_states = [
                {
                    "last_action": 0, 
                    "last_reward": -float('inf'),
                    "best_reward": -float('inf'),
                    "best_displacement": torch.zeros(2),
                    "value_memory": torch.zeros(len(self.action_vectors))
                } 
                for _ in range(self.n_agents)
            ]
        elif self.strategy == "pso":
            self.agent_states = [
                {
                    "velocity": torch.zeros(2),
                    "personal_best_reward": -float('inf'),
                    "personal_best_displacement": torch.zeros(2),
                    "global_best_reward": -float('inf'),
                    "global_best_displacement": torch.zeros(2),
                    "value_memory": torch.zeros(len(self.action_vectors))
                } 
                for _ in range(self.n_agents)
            ]
    
    def update_displacement(self, movements):
        """Update displacement from start position"""
        self.displacements += movements
    
    def position_hash(self, displacement, precision=0.5):
        """Create discrete position hash from continuous displacement"""
        x = round(displacement[0].item() / precision) * precision
        y = round(displacement[1].item() / precision) * precision
        return f"{x:.2f},{y:.2f}"
    
    def update_cognitive_map(self, agent_idx, reward):
        """Remember this relative position if it's valuable"""
        pos_hash = self.position_hash(self.displacements[agent_idx])
        current_best = self.cognitive_maps[agent_idx].get(pos_hash, (-float('inf'), None))[0]
        
        if reward > current_best:
            self.cognitive_maps[agent_idx][pos_hash] = (
                reward.item(),
                self.displacements[agent_idx].clone()
            )
    
    def compute_relative_positions(self, positions):
        """Compute relative positions between all agents"""
        relative_positions = [[None] * self.n_agents for _ in range(self.n_agents)]
        
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i == j:
                    continue
                diff = positions[j] - positions[i]
                distance = torch.norm(diff).item()
                angle = torch.atan2(diff[1], diff[0]).item()
                relative_positions[i][j] = (distance, angle)
        
        return relative_positions
    
    def compute_actions(self, positions, movements):
        """Compute actions for all agents using cognitive mapping"""
        # Update displacement tracking
        self.update_displacement(movements)
        
        actions = torch.zeros(self.n_agents, dtype=torch.long)
        relative_positions = self.compute_relative_positions(positions)
        
        # Update movement vectors from last actions
        for i in range(self.n_agents):
            if "last_action" in self.agent_states[i]:
                action_idx = self.agent_states[i]["last_action"]
                if action_idx < len(self.action_vectors):
                    self.last_vectors[i] = self.action_vectors[action_idx]
        
        # Find global best for PSO
        if self.strategy == "pso":
            best_reward, best_idx = torch.max(self.last_rewards, dim=0)
            best_reward = best_reward.item()
            best_idx = best_idx.item()
            
            for i in range(self.n_agents):
                if best_reward > self.agent_states[i]["global_best_reward"]:
                    self.agent_states[i]["global_best_reward"] = best_reward
                    self.agent_states[i]["global_best_displacement"] = self.displacements[best_idx].clone()
        
        for i in range(self.n_agents):
            # Update cognitive map with current position
            self.update_cognitive_map(i, self.last_rewards[i])
            
            # Get neighbors within communication radius
            neighbors = []
            neighbor_rewards = []
            
            for j in range(self.n_agents):
                if i == j:
                    continue
                if relative_positions[i][j] and relative_positions[i][j][0] < self.comm_radius:
                    neighbors.append(relative_positions[i][j])
                    neighbor_rewards.append(self.last_rewards[j].item())
            
            if self.strategy == "sticky_random":
                actions[i] = self._sticky_random_action(i, neighbors, self.last_rewards[i])
            elif self.strategy == "pso":
                actions[i] = self._pso_action(i, neighbors, neighbor_rewards, self.last_rewards[i])
        
        return actions
    
    def update_state(self, actions, rewards, movements):
        """Update controller state after environment step"""
        # Update movement vectors
        for i in range(self.n_agents):
            if actions[i] < len(self.action_vectors):
                self.last_vectors[i] = self.action_vectors[actions[i]]
        
        # Update rewards
        self.last_rewards = rewards.clone()
        
        # Update agent-specific states
        for i in range(self.n_agents):
            # Update value memory (Q-learning like)
            if "value_memory" in self.agent_states[i]:
                action_idx = actions[i].item()
                reward_diff = rewards[i] - self.agent_states[i].get("last_reward", 0)
                self.agent_states[i]["value_memory"][action_idx] += 0.1 * reward_diff.item()
            
            if self.strategy == "sticky_random":
                # Update personal best
                if rewards[i] > self.agent_states[i]["best_reward"]:
                    self.agent_states[i]["best_reward"] = rewards[i].item()
                    self.agent_states[i]["best_displacement"] = self.displacements[i].clone()
                
                self.agent_states[i]["last_reward"] = rewards[i].item()
                self.agent_states[i]["last_action"] = actions[i].item()
                
            elif self.strategy == "pso":
                # Update personal best
                if rewards[i] > self.agent_states[i]["personal_best_reward"]:
                    self.agent_states[i]["personal_best_reward"] = rewards[i].item()
                    self.agent_states[i]["personal_best_displacement"] = self.displacements[i].clone()
    
    def _sticky_random_action(self, agent_idx, neighbors, current_reward):
        """Compute action with cognitive mapping and value-based selection"""
        state = self.agent_states[agent_idx]
        current_disp = self.displacements[agent_idx]
        
        # 1. Calculate attraction to personal best position (relative)
        to_best = torch.zeros(2)
        if state["best_reward"] > -float('inf'):
            to_best = state["best_displacement"] - current_disp
            dist = torch.norm(to_best)
            if dist > 1e-5:
                to_best = to_best / dist * 0.3 * min(1.0, state["best_reward"])
        
        # 2. Calculate swarm center attraction (only if not in good position)
        swarm_vector = torch.zeros(2)
        swarm_weight = 0.4
        if neighbors and current_reward < state["best_reward"]:
            neighbor_positions = [
                self.relative_to_absolute(current_disp, d, a) 
                for d, a in neighbors
            ]
            swarm_center = torch.stack(neighbor_positions).mean(dim=0)
            to_center = swarm_center - current_disp
            dist = torch.norm(to_center)
            if dist > 1e-5:
                swarm_vector = to_center / dist * swarm_weight
        
        # 3. Value-based action selection
        value_bonus = state["value_memory"] / (state["value_memory"].max() + 1e-8)
        value_probs = F.softmax(value_bonus, dim=0)
        
        # Decision process
        if current_reward > state["last_reward"]:
            # Repeat last successful action
            action_idx = state["last_action"]
        else:
            # Explore with value bias
            if torch.rand(1).item() < 0.7:  # 70% value-based selection
                action_idx = torch.multinomial(value_probs, 1).item()
            else:
                action_idx = torch.randint(0, len(self.action_vectors), (1,)).item()
        
        # 4. Apply attractions to base action
        base_vector = self.action_vectors[action_idx]
        combined_vector = base_vector + to_best + swarm_vector
        
        # 5. Convert to discrete action
        if torch.norm(combined_vector) < 1e-5:
            return len(self.action_vectors) - 1  # Stand still
        
        # Find action with closest direction
        dot_products = torch.mv(self.action_vectors, combined_vector)
        action_idx = torch.argmax(dot_products).item()
        
        return action_idx
    
    def _pso_action(self, agent_idx, neighbors, neighbor_rewards, current_reward):
        """PSO with cognitive mapping and position-aware swarm attraction"""
        state = self.agent_states[agent_idx]
        current_disp = self.displacements[agent_idx]
        
        # 1. Cognitive component (personal best)
        cognitive = torch.zeros(2)
        if state["personal_best_reward"] > -float('inf'):
            to_personal_best = state["personal_best_displacement"] - current_disp
            dist = torch.norm(to_personal_best)
            if dist > 1e-5:
                cognitive = (0.8 * torch.rand(1).item() * 
                            to_personal_best / dist * 
                            min(1.0, state["personal_best_reward"]))
        
        # 2. Social component (global best)
        social = torch.zeros(2)
        if state["global_best_reward"] > -float('inf'):
            to_global_best = state["global_best_displacement"] - current_disp
            dist = torch.norm(to_global_best)
            if dist > 1e-5:
                social = (1.5 * torch.rand(1).item() * 
                         to_global_best / dist * 
                         min(1.0, state["global_best_reward"]))
        
        # 3. Swarm component (only if not in good position)
        swarm_vector = torch.zeros(2)
        if neighbors and current_reward < state["personal_best_reward"]:
            neighbor_positions = [
                self.relative_to_absolute(current_disp, d, a) 
                for d, a in neighbors
            ]
            swarm_center = torch.stack(neighbor_positions).mean(dim=0)
            to_center = swarm_center - current_disp
            dist = torch.norm(to_center)
            if dist > 1e-5:
                swarm_vector = to_center / dist * 0.4
        
        # 4. Combine components
        state["velocity"] = 0.6 * state["velocity"] + cognitive + social + swarm_vector
        
        # 5. Convert to discrete action
        if torch.norm(state["velocity"]) < 1e-5:
            return len(self.action_vectors) - 1  # Stand still
        
        # Find action with closest direction
        dot_products = torch.mv(self.action_vectors, state["velocity"])
        action_idx = torch.argmax(dot_products).item()
        
        return action_idx
    
    def relative_to_absolute(self, reference, distance, angle):
        """Convert relative position to absolute displacement"""
        dx = distance * math.cos(angle)
        dy = distance * math.sin(angle)
        return reference + torch.tensor([dx, dy])

def main_heuristic(strategy="sticky_random"):
    # Environment setup
    n_agents = 5
    comm_radius = 50.0
    max_steps = 1000
    render_interval = 0.1
    
    env_args = EnvArgs2D(
        n_actors=n_agents,
        velocity=0.15,
        angular_velocity=45,
        movement_noise=0.005,
        max_steps=max_steps,
        starting_position_mean=(0, 0),
        starting_position_var=10,
        num_actions=24  # 24 directions + stand still
    )
    env = Environment2D(env_args)
    
    # Initialize multi-agent controller
    controller = MultiAgentController(env, strategy=strategy, comm_radius=comm_radius)
    
    # Main loop
    for episode in range(10):  # Run 10 episodes
        positions = env.reset()
        controller.reset()
        done = False
        step_count = 0
        cumulative_rewards = torch.zeros(n_agents)
        last_positions = positions.clone()
        
        # For visualization
        best_position_markers = [None] * n_agents
        cognitive_map_markers = [None] * n_agents
        
        while not done:
            start_time = time.time()
            
            # Compute movements since last step
            movements = positions - last_positions
            last_positions = positions.clone()
            
            # Compute graph for visualization
            edge_index, _ = compute_graph(positions, comm_radius, 'cpu')
            
            # Get actions for all agents
            actions = controller.compute_actions(positions, movements)
            
            # Step environment
            positions, rewards, done = env.step(actions)
            cumulative_rewards += rewards
            
            # Update controller state
            controller.update_state(actions, rewards, movements)
            
            # Update visualization
            if env.fig is not None:
                # Visualize cognitive maps
                for i, marker_group in enumerate(cognitive_map_markers):
                    if marker_group is not None:
                        for marker in marker_group:
                            marker.remove()
                
                cognitive_map_markers = []
                for i in range(n_agents):
                    markers = []
                    for pos_hash, (reward, disp) in controller.cognitive_maps[i].items():
                        if reward > 0.5:  # Only show valuable positions
                            marker = env.ax.scatter(
                                disp[0].item(), disp[1].item(),
                                s=30 + 50 * reward,
                                alpha=0.6,
                                color=env.agent_colors[i],
                                marker="o",
                                edgecolors='white' if reward > 0.7 else 'none'
                            )
                            markers.append(marker)
                    cognitive_map_markers.append(markers)
            
            # Render environment
            env.render(rewards, edge_index=edge_index)
            
            # Control rendering speed
            elapsed = time.time() - start_time
            if elapsed < render_interval:
                time.sleep(render_interval - elapsed)
            
            step_count += 1
            if step_count >= max_steps:
                break
        
        print(f"Episode {episode+1} ({strategy}) complete. "
              f"Avg cumulative reward: {cumulative_rewards.mean().item():.2f}")
    
    plt.ioff()
    plt.show()

def compute_graph(positions, radius, device='cpu'):
    n_agents = positions.size(0)
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    dist = torch.norm(diff, dim=-1)
    mask = (dist < radius) & (dist > 0)
    edge_index = torch.nonzero(mask, as_tuple=False).t().contiguous()
    
    if edge_index.size(1) > 0:
        vec = positions[edge_index[1]] - positions[edge_index[0]]
        distances = torch.norm(vec, dim=1, keepdim=True)
        angles = torch.atan2(vec[:,1], vec[:,0]).unsqueeze(1)
        edge_attr = torch.cat([distances, angles], dim=1)
    else:
        edge_attr = torch.zeros((0, 2), device=device)
    
    return edge_index, edge_attr

if __name__ == "__main__":
    strategy = "sticky_random"  # "sticky_random" or "pso"
    main_heuristic(strategy)