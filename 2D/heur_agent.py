import torch
import time
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from environment import EnvArgs2D, Environment2D

class RelativePSOMultiAgent:
    def __init__(self, env, comm_radius=50.0, memory_size=5):
        self.env = env
        self.n_agents = env.n_actors
        self.action_vectors = env.action_vectors
        self.comm_radius = comm_radius
        self.memory_size = memory_size
        self.momentum_coeff = 0.7
        self.reset()
    
    def reset(self):
        self.current_displacement = torch.zeros((self.n_agents, 2))
        
        self.cognitive_memory = [{
            'positions': [],
            'rewards': [],
            'displacements': []
        } for _ in range(self.n_agents)]
        
        self.social_best = [{
            'relative_position': torch.zeros(2),
            'reward': -float('inf')
        } for _ in range(self.n_agents)]
        
        self.movement_history = [[] for _ in range(self.n_agents)]
    

    def update_cognitive_memory(self, agent_idx, reward, current_position, displacement):
        mem = self.cognitive_memory[agent_idx]
        
        mem['positions'].append(current_position.clone())
        mem['rewards'].append(reward)
        mem['displacements'].append(displacement.clone())
        
        if len(mem['positions']) > self.memory_size:
            mem['positions'] = mem['positions'][-self.memory_size:]
            mem['rewards'] = mem['rewards'][-self.memory_size:]
            mem['displacements'] = mem['displacements'][-self.memory_size:]
    

    def get_best_cognitive_direction(self, agent_idx):
        mem = self.cognitive_memory[agent_idx]
        
        best_idx = torch.argmax(torch.tensor(mem['rewards'])).item()
        best_global_displacement = mem['displacements'][best_idx]
        current_displacement = self.cognitive_memory[agent_idx]['displacements'][-1]
        relative_position = best_global_displacement - current_displacement

        #if agent_idx == 0:
        #    print(f"Agent {agent_idx} \nbest global displacement: {best_global_displacement}, \
        #                              \ncurrent displacement: {current_displacement}")  
        
        return relative_position
    
    def update_social_memory(self, positions):
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)
        dist_matrix = torch.norm(diff, dim=-1)
        
        for i in range(self.n_agents):
            neighbors = []
            for j in range(self.n_agents):
                if i != j and dist_matrix[i, j] < self.comm_radius:
                    neighbors.append(j)
            
            best_reward = -float('inf')
            best_agent = -1
            for j in neighbors:
                if self.cognitive_memory[j]['rewards']:
                    #neighbor_best_reward = max(self.cognitive_memory[j]['rewards'])
                    neighbor_best_reward = self.cognitive_memory[j]['rewards'][-1]

                    if neighbor_best_reward > best_reward:
                        best_reward = neighbor_best_reward
                        best_agent = j
            
            if best_agent == -1:
                continue
            
            best_idx = torch.argmax(torch.tensor(self.cognitive_memory[best_agent]['rewards'])).item()
            #best_position = self.cognitive_memory[best_agent]['positions'][best_idx]
            best_position = self.cognitive_memory[best_agent]['positions'][-1]
            
            relative_position = best_position - positions[i]
            
            self.social_best[i]['relative_position'] = relative_position.clone()
            self.social_best[i]['reward'] = best_reward
    

    def compute_actions(self):
        actions = torch.zeros(self.n_agents, dtype=torch.long)
        
        for i in range(self.n_agents):
            
            cognitive_dir = self.get_best_cognitive_direction(i)
            social_dir = self.social_best[i]['relative_position']
            momentum_dir = self.current_displacement[i]
            
            if torch.norm(cognitive_dir) < 1e-5 and torch.norm(social_dir) < 1e-5:
                actions[i] = len(self.action_vectors) - 1  # Stand still
                actions[i] = torch.randint(0, len(self.action_vectors) - 1, (1,)).item()
                continue
            
            # Calculate adaptive weights
            cognitive_reward = max(self.cognitive_memory[i]['rewards']) if self.cognitive_memory[i]['rewards'] else 1e-5
            social_reward = self.social_best[i]['reward']
            
            if social_reward > 0.0:
                cognitive_weight = math.exp(cognitive_reward / (social_reward + cognitive_reward + 1e-8))
                social_weight = math.exp(social_reward / (social_reward + cognitive_reward + 1e-8))
            else:
                cognitive_weight = 1.0
                social_weight = 0.0

            if torch.norm(cognitive_dir) > 1e-5:
                cognitive_dir = cognitive_dir / torch.norm(cognitive_dir)
            if torch.norm(social_dir) > 1e-5:
                social_dir = social_dir / torch.norm(social_dir)
            if torch.norm(momentum_dir) > 1e-5:
                momentum_dir = momentum_dir / torch.norm(momentum_dir)
            
            desired_direction = cognitive_weight * cognitive_dir + social_weight * social_dir
            # desired_direction = social_weight * social_dir # No cognitivie component
            # desired_direction = self.momentum_coeff * momentum_dir + random.uniform(0, 1) * cognitive_dir + random.uniform(0, 1) * social_dir # Actual PSO

            if torch.norm(desired_direction) < 1e-5:
                actions[i] = len(self.action_vectors) - 1 
            else:
                desired_direction = desired_direction / torch.norm(desired_direction)
                
                dot_products = torch.mv(self.action_vectors[:-1], desired_direction)
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
