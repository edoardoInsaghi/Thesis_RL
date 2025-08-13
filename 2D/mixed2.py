import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from environment import EnvArgs2D, Environment2D
from shared_agent import Agent
from shared_net import NNDICT
from heur_agent import RelativePSOMultiAgent, compute_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

action_idx_to_repr = {
    0: [0,1],
    1: [1,1],
    2: [1,0],
    3: [1,-1],
    4: [0,-1],
    5: [-1,-1],
    6: [-1,0],
    7: [-1,1],
    8: [0,0]
}

class HybridAgent:
    def __init__(self, env, rl_weights, comm_radius=30.0):
        self.env = env
        self.comm_radius = comm_radius
        self.n_agents = env.n_actors
        
        self.rl_agent = Agent(
            net='mlp',
            temp_memory=10,
            num_actions=env.args.num_actions,
            n_agents=env.n_actors,
            device=device,
            weights=rl_weights
        )
        self.rl_agent.network.eval()
        
        self.heur_agent = RelativePSOMultiAgent(
            env, 
            comm_radius=comm_radius,
            memory_size=50,
            cognitive_decay=0.95
        )
        self.current_displacements = torch.zeros((env.n_actors, 2))
        self.last_rewards = torch.zeros(env.n_actors)
        self.has_neighbor = torch.zeros(env.n_actors, dtype=torch.bool)
        self.is_best = torch.zeros(env.n_actors, dtype=torch.bool)

    def reset(self):
        positions = self.env.reset()
        self.rl_agent.reset_memory()
        self.heur_agent.reset()
        self.current_displacements = torch.zeros((self.n_agents, 2))
        self.last_rewards = torch.zeros(self.n_agents)
        self.has_neighbor = torch.zeros(self.n_agents, dtype=torch.bool)
        self.is_best = torch.zeros(self.n_agents, dtype=torch.bool)
        return positions

    def step(self, positions):
        # Compute neighbor information
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)
        dist = torch.norm(diff, dim=-1)
        self.has_neighbor = (dist < self.comm_radius) & (dist > 0)
        
        # Identify best agents in their neighborhoods
        self.is_best[:] = False
        for i in range(self.n_agents):
            # Find neighbors including self
            neighbors = torch.where(self.has_neighbor[i])[0]
            
            # Always include self in the group
            if neighbors.numel() == 0:
                # No neighbors - agent is alone
                group = torch.tensor([i])
            else:
                group = torch.cat([torch.tensor([i]), neighbors])
                
            group_rewards = self.last_rewards[group]
            max_reward = torch.max(group_rewards)
            
            # Check if current agent has max reward
            self.is_best[i] = (self.last_rewards[i] >= max_reward - 1e-5)

        # Update heuristic memory with current state
        for i in range(self.n_agents):
            self.heur_agent.update_cognitive_memory(
                i, 
                self.last_rewards[i].item(), 
                positions[i], 
                self.current_displacements[i]
            )
        
        # Update social memory and get heuristic actions
        self.heur_agent.update_social_memory(positions)
        heur_actions = self.heur_agent.compute_actions(positions)

        # Get RL actions
        with torch.no_grad():
            policies, _ = self.rl_agent.act()
            rl_actions = torch.multinomial(policies, 1).squeeze().cpu()

        # Combine actions: use RL when alone OR when best in neighborhood
        use_rl = (~self.has_neighbor.any(dim=1)) | self.is_best
        actions = torch.where(use_rl, rl_actions, heur_actions)
        
        # Execute environment step
        new_positions, rewards, done = self.env.step(actions)
        
        # Update displacements
        movements = self.env.action_vectors[actions]
        self.current_displacements += movements
        
        # Update RL agent memory
        action_reprs = torch.tensor([action_idx_to_repr[idx.item()] for idx in actions])
        self.rl_agent.update_memory(action_reprs, rewards)
        
        # Update rewards for next step
        self.last_rewards = rewards.clone()
        
        return new_positions, rewards, done, use_rl

def main_hybrid_evaluation(rl_weights_path):
    n_agents = 10
    comm_radius = 10.0
    max_steps = 1000
    render_interval = 0.05
    
    env_args = EnvArgs2D(
        n_actors=n_agents,
        velocity=0.5,
        angular_velocity=45,
        movement_noise=0.005,
        max_steps=max_steps,
        starting_position_mean=(0, 0),
        starting_position_var=10,
        num_actions=9
    )
    env = Environment2D(env_args)
    hybrid_agent = HybridAgent(env, rl_weights_path, comm_radius)
    
    positions = hybrid_agent.reset()
    cumulative_rewards = torch.zeros(n_agents)
    done = False
    step_count = 0
    
    while not done:
        start_time = time.time()
        
        # Compute graph for visualization
        edge_index, _ = compute_graph(positions, comm_radius, device)
        
        # Agent step - returns new positions from environment.step()
        new_positions, rewards, done, use_rl = hybrid_agent.step(positions)
        cumulative_rewards += rewards
        
        # Update visualization colors based on strategy
        agent_colors = []
        for i in range(n_agents):
            if use_rl[i]:
                # RL agent - red if alone, green if best in group
                if hybrid_agent.has_neighbor[i].any():
                    agent_colors.append([0.2, 0.8, 0.2, 1])  # Green: RL as best
                else:
                    agent_colors.append([1, 0, 0, 1])  # Red: RL alone
            else:
                agent_colors.append([0, 0, 1, 1])  # Blue: Heuristic
        env.agent_colors = np.array(agent_colors)
        
        # Render environment with current state
        env.render(rewards, edge_index=edge_index)
        
        # Prepare for next iteration
        positions = new_positions.clone()
        
        # Control rendering speed
        elapsed = time.time() - start_time
        if elapsed < render_interval:
            time.sleep(render_interval - elapsed)
        
        step_count += 1
        if step_count >= max_steps:
            break
    
    print(f"Evaluation complete. Avg cumulative reward: {cumulative_rewards.mean().item():.2f}")

if __name__ == "__main__":
    main_hybrid_evaluation("weights/shared_agent_shared_network_10.pth")