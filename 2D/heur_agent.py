import torch
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from environment import EnvArgs2D, Environment2D

class RelativePSOMultiAgent:
    def __init__(self, env, comm_radius=50.0, memory_size=5, cognitive_decay=0.95):
        """
        Multi-agent PSO with relative position memory
        
        :param env: Environment instance
        :param comm_radius: Communication radius for neighbor detection
        :param memory_size: Number of best positions to remember
        :param cognitive_decay: Decay rate for cognitive memory importance
        """
        self.env = env
        self.n_agents = env.n_actors
        self.action_vectors = env.action_vectors
        self.comm_radius = comm_radius
        self.memory_size = memory_size
        self.cognitive_decay = cognitive_decay
        self.reset()
    
    def reset(self):
        """Reset all agent states between episodes"""
        # Current displacement from start position
        self.current_displacement = torch.zeros((self.n_agents, 2))
        
        # Cognitive memory (best positions relative to agent's position when recorded)
        self.cognitive_memory = [{
            'positions': [],
            'rewards': [],
            'displacements': []  # From start when recorded
        } for _ in range(self.n_agents)]
        
        # Social memory (best neighbor information)
        self.social_best = [{
            'relative_position': torch.zeros(2),
            'reward': -float('inf')
        } for _ in range(self.n_agents)]
        
        # Movement history for odometry
        self.movement_history = [[] for _ in range(self.n_agents)]
    
    def update_cognitive_memory(self, agent_idx, reward, current_position, displacement):
        """Update cognitive memory with current position info"""
        mem = self.cognitive_memory[agent_idx]
        
        # Add new entry
        mem['positions'].append(current_position.clone())
        mem['rewards'].append(reward)
        mem['displacements'].append(displacement.clone())
        
        # Trim to memory size
        if len(mem['positions']) > self.memory_size:
            mem['positions'] = mem['positions'][-self.memory_size:]
            mem['rewards'] = mem['rewards'][-self.memory_size:]
            mem['displacements'] = mem['displacements'][-self.memory_size:]
    
    def get_best_cognitive_direction(self, agent_idx, current_position, current_displacement):
        """Get direction to best cognitive position relative to current position"""
        mem = self.cognitive_memory[agent_idx]
        if not mem['positions']:
            return torch.zeros(2)
        
        # Find best position in memory
        best_idx = torch.argmax(torch.tensor(mem['rewards'])).item()
        best_global_displacement = mem['displacements'][best_idx]
        
        # Convert to relative position from current location
        relative_position = best_global_displacement - current_displacement
        
        return relative_position
    
    def update_social_memory(self, positions, rewards):
        """Update social memory based on neighbors' best cognitive positions"""
        # Compute distance matrix
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)
        dist_matrix = torch.norm(diff, dim=-1)
        
        for i in range(self.n_agents):
            # Find neighbors within communication radius
            neighbors = []
            for j in range(self.n_agents):
                if i != j and dist_matrix[i, j] < self.comm_radius:
                    neighbors.append(j)
            
            if not neighbors:
                # No neighbors, reset social memory
                self.social_best[i]['reward'] = -float('inf')
                continue
            
            # Find best neighbor based on cognitive memory
            best_reward = -float('inf')
            best_agent = -1
            for j in neighbors:
                if self.cognitive_memory[j]['rewards']:
                    neighbor_best_reward = max(self.cognitive_memory[j]['rewards'])
                    if neighbor_best_reward > best_reward:
                        best_reward = neighbor_best_reward
                        best_agent = j
            
            if best_agent == -1:
                continue
            
            # Get best cognitive position from best neighbor
            best_idx = torch.argmax(torch.tensor(
                self.cognitive_memory[best_agent]['rewards'])).item()
            best_position = self.cognitive_memory[best_agent]['positions'][best_idx]
            
            # Calculate relative position to current agent
            relative_position = best_position - positions[i]
            
            # Update social memory
            self.social_best[i]['relative_position'] = relative_position.clone()
            self.social_best[i]['reward'] = best_reward
    
    def compute_actions(self, positions, rewards):
        """Compute PSO-inspired actions for all agents"""
        actions = torch.zeros(self.n_agents, dtype=torch.long)
        
        for i in range(self.n_agents):
            # Get cognitive direction (to best remembered position)
            cognitive_dir = self.get_best_cognitive_direction(
                i, positions[i], self.current_displacement[i])
            
            # Get social direction (to best neighbor's best position)
            social_dir = self.social_best[i]['relative_position']
            
            # Skip if no valid directions
            if torch.norm(cognitive_dir) < 1e-5 and torch.norm(social_dir) < 1e-5:
                actions[i] = len(self.action_vectors) - 1  # Stand still
                continue
            
            # Calculate adaptive weights
            cognitive_reward = max(self.cognitive_memory[i]['rewards']) if self.cognitive_memory[i]['rewards'] else 1e-5
            social_reward = self.social_best[i]['reward']
            
            if cognitive_reward > 0 and social_reward > 0:
                cognitive_weight = math.exp(cognitive_reward / (social_reward + 1e-8))
                social_weight = math.exp(social_reward / (cognitive_reward + 1e-8))
                
                # Normalize weights
                total_weight = cognitive_weight + social_weight
                cognitive_weight /= total_weight
                social_weight /= total_weight
            else:
                # Default equal weights if rewards not positive
                cognitive_weight = 0.5
                social_weight = 0.5
            
            # Normalize directions
            if torch.norm(cognitive_dir) > 1e-5:
                cognitive_dir = cognitive_dir / torch.norm(cognitive_dir)
            if torch.norm(social_dir) > 1e-5:
                social_dir = social_dir / torch.norm(social_dir)
            
            # Combine directions
            desired_direction = (cognitive_weight * cognitive_dir + 
                                 social_weight * social_dir)
            
            # Find closest action
            if torch.norm(desired_direction) < 1e-5:
                actions[i] = len(self.action_vectors) - 1 
            else:
                # Normalize desired direction
                desired_direction = desired_direction / torch.norm(desired_direction)
                
                # Find action with closest direction
                dot_products = torch.mv(self.action_vectors[:-1], desired_direction)  # Exclude stand still
                action_idx = torch.argmax(dot_products).item()
                actions[i] = action_idx
        
        return actions

def main_relative_pso():

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
        num_actions=9  # 24 directions + stand still
    )
    env = Environment2D(env_args)
    
    # Initialize multi-agent controller
    controller = RelativePSOMultiAgent(env, comm_radius=comm_radius, memory_size=50)
    
    # Main loop
    for episode in range(10):  # Run 10 episodes
        positions = env.reset()
        controller.reset()
        done = False
        step_count = 0
        cumulative_rewards = torch.zeros(n_agents)
        
        # Initialize displacements
        start_positions = positions.clone()
        current_displacements = torch.zeros((n_agents, 2))
        
        while not done:
            start_time = time.time()
            
            # Compute graph for visualization
            edge_index, _ = compute_graph(positions, comm_radius, 'cpu')
            
            # Get rewards from environment (landscape rewards)
            rewards = torch.zeros(n_agents)  # Will be updated
            
            # Update cognitive memory with current positions
            for i in range(n_agents):
                controller.update_cognitive_memory(
                    i, rewards[i], positions[i], current_displacements[i]
                )
            
            # Update social memory
            controller.update_social_memory(positions, rewards)
            
            # Get actions for all agents
            actions = controller.compute_actions(positions, rewards)
            
            # Step environment
            new_positions, env_rewards, done = env.step(actions)
            cumulative_rewards += env_rewards
            
            # Update displacements (using intended movement without noise)
            for i in range(n_agents):
                movement = env.action_vectors[actions[i]]
                current_displacements[i] += movement
            
            # Update rewards for next step
            rewards = env_rewards.clone()
            
            # Update cognitive memory with actual rewards
            for i in range(n_agents):
                controller.update_cognitive_memory(
                    i, rewards[i], positions[i], current_displacements[i]
                )
            
            # Update positions
            positions = new_positions.clone()
            
            # Render environment
            env.render(env_rewards, edge_index=edge_index)
            
            # Control rendering speed
            elapsed = time.time() - start_time
            if elapsed < render_interval:
                time.sleep(render_interval - elapsed)
            
            step_count += 1
            if step_count >= max_steps:
                break
        
        print(f"Episode {episode+1} complete. "
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
    main_relative_pso()
