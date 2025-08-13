import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from environment import EnvArgs2D, Environment2D
from shared_agent import Agent
from shared_net import NNDICT
from heur_agent import RelativePSOMultiAgent
from shared_main import compute_graph

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
    def __init__(self, env, rl_weights, comm_radius=5.0):
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
        self.has_neighbor = torch.zeros(env.n_actors, dtype=torch.bool)

    def reset(self):
        positions = self.env.reset()
        self.rl_agent.reset_memory()
        self.heur_agent.reset()
        self.current_displacements = torch.zeros((self.n_agents, 2))
        self.has_neighbor = torch.zeros(self.n_agents, dtype=torch.bool)
        return positions

    def step(self, positions, last_rewards):

        diff = positions.unsqueeze(1) - positions.unsqueeze(0)
        dist = torch.norm(diff, dim=-1)
        mask = (dist < self.comm_radius) & (dist > 0)
        self.has_neighbor = mask.any(dim=1)

        for i in range(self.n_agents):
            self.heur_agent.update_cognitive_memory(
                i, 
                last_rewards[i].item(), 
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

        # Combine actions based on neighbor presence
        actions = torch.where(self.has_neighbor, heur_actions, rl_actions)
        
        # Execute environment step
        new_positions, rewards, done = self.env.step(actions)
        
        # Update displacements
        movements = self.env.action_vectors[actions]
        self.current_displacements += movements
        
        # Update RL agent memory
        action_reprs = torch.tensor([action_idx_to_repr[idx.item()] for idx in actions])
        self.rl_agent.update_memory(action_reprs, rewards)
        
        return new_positions, rewards, done

def main_hybrid_evaluation(rl_weights_path):
    n_agents = 20
    comm_radius = 5.0
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
    rewards = torch.zeros(n_agents)
    
    while not done:
        start_time = time.time()
        
        # Compute graph for visualization
        edge_index, _, _ = compute_graph(positions, comm_radius, device)
        
        # Agent step - returns new positions from environment.step()
        new_positions, rewards, done = hybrid_agent.step(positions, rewards)
        cumulative_rewards += rewards
        
        # Update visualization colors based on strategy
        env.agent_colors = np.array([
            [0, 0, 1, 1] if hybrid_agent.has_neighbor[i] else [1, 0, 0, 1] 
            for i in range(n_agents)
        ])
        
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