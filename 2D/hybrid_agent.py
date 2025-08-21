import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from pathlib import Path
from environment import EnvArgs2D, Environment2D
from shared_agent import Agent
from shared_net import NNDICT
from heur_agent import RelativePSOMultiAgent, compute_graph

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

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
    def __init__(self, env, rl_weights, comm_radius=50.0, sample_noise=0.005):
        self.env = env
        self.comm_radius = comm_radius
        self.n_agents = env.n_actors
        self.sample_noise = sample_noise
        
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
            memory_size=50
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
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)
        dist = torch.norm(diff, dim=-1)
        self.has_neighbor = (dist < self.comm_radius) & (dist > 0)
        
        self.is_best[:] = False
        for i in range(self.n_agents):
            neighbors = torch.where(self.has_neighbor[i])[0]
            
            if neighbors.numel() == 0:
                group = torch.tensor([i])
            else:
                group = torch.cat([torch.tensor([i]), neighbors])
                
            group_rewards = self.last_rewards[group]
            max_reward = torch.max(group_rewards)
            
            self.is_best[i] = (self.last_rewards[i] >= max_reward - 1e-5)
            # self.is_best[i] = False # All agents are heuristic if this is on

        # Update heuristic agent memory
        for i in range(self.n_agents):
            self.heur_agent.update_cognitive_memory(
                i, 
                self.last_rewards[i].item(), 
                positions[i], 
                self.current_displacements[i]
            )
        self.heur_agent.update_social_memory(positions)

        # Get heuristic actions
        heur_actions = self.heur_agent.compute_actions(positions)

        # Get RL actions
        with torch.no_grad():
            policies, _ = self.rl_agent.act()
            rl_actions = torch.multinomial(policies, 1).squeeze().cpu()

        # Combine actions: use RL when alone OR when best in neighborhood
        use_rl = (~self.has_neighbor.any(dim=1)) | self.is_best
        actions = torch.where(use_rl, rl_actions, heur_actions)
        
        new_positions, rewards, done = self.env.step(actions)
        sampled_rewards = rewards.clone() + torch.randn_like(rewards) * self.sample_noise
        
        movements = self.env.action_vectors[actions]
        self.current_displacements += movements
        
        action_reprs = torch.tensor([action_idx_to_repr[idx.item()] for idx in actions])
        self.rl_agent.update_memory(action_reprs, sampled_rewards)
        
        self.last_rewards = sampled_rewards.clone()
        
        return new_positions, rewards, sampled_rewards, done, use_rl





def main_hybrid_evaluation(rl_weights_path,
                           n_agents = 5,
                           comm_radius = 10.0,
                           movement_noise = 0.01,
                           sample_noise = 0.01,
                           render = False,
                           model_name = None, 
                           max_steps = 250,
                           num_episodes = 100):
                    
    n_agents = n_agents
    comm_radius = comm_radius
    max_steps = max_steps
    num_episodes = num_episodes
    render_interval = 0.05
    csv_filename = f"{model_name}_results.csv" if model_name else None
    
    env_args = EnvArgs2D(
        n_actors=n_agents,
        velocity=0.5,
        angular_velocity=45,
        movement_noise=movement_noise,
        max_steps=max_steps,
        starting_position_mean=(0, 0),
        starting_position_var=10,
        num_actions=9
    )
    env = Environment2D(env_args)
    hybrid_agent = HybridAgent(env, rl_weights_path, comm_radius, sample_noise)
    
    if not render:
        assert model_name is not None, "Model name must be provided for non-rendering mode"
        results_path = Path(f"{model_name}_results.csv")
        assert results_path.exists(), f"Results file {results_path} does not exist"
    
    for episode in range(num_episodes):
        positions = hybrid_agent.reset()
        cumulative_rewards = torch.zeros(n_agents)
        best_reward = -float('inf')
        done = False
        step_count = 0
        
        while not done:
            if render:
                start_time = time.time()
                edge_index, _ = compute_graph(positions, comm_radius, device)
            
            new_positions, rewards, sampled_rewards, done, use_rl = hybrid_agent.step(positions)
            cumulative_rewards += rewards
            
            current_max = torch.max(rewards).item()
            if current_max > best_reward:
                best_reward = current_max
            
            if render:
                agent_colors = []
                for i in range(n_agents):
                    if use_rl[i]:
                        # RL agent: red if alone, green if best in group
                        if hybrid_agent.has_neighbor[i].any():
                            agent_colors.append([0.2, 0.8, 0.2, 1])  # Green: RL as best
                        else:
                            agent_colors.append([1, 0, 0, 1])  # Red: RL alone
                    else:
                        agent_colors.append([0, 0, 1, 1])  # Blue: Heuristic
                env.agent_colors = np.array(agent_colors)
                
                env.render(sampled_rewards, edge_index=edge_index)
                
                elapsed = time.time() - start_time
                if elapsed < render_interval:
                    time.sleep(render_interval - elapsed)
            
            positions = new_positions.clone()
            step_count += 1
            if step_count >= max_steps:
                break
        
        normalized_cumulative = cumulative_rewards / max_steps
        
        # Log results to csv
        if not render:
            with open(csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    episode,
                    normalized_cumulative.mean().item(),
                    rewards.mean().item(),
                    best_reward,
                    n_agents,
                    comm_radius,
                    movement_noise,
                    sample_noise,
                    model_name,
                ])
        
        print(f"Episode {episode+1}/{num_episodes} complete. "
              f"Avg cumulative reward: {normalized_cumulative.mean().item():.2f}, "
              f"Final reward: {rewards.mean().item():.2f}, "
              f"Best reward: {best_reward:.2f}")
    
    if not render:
        print(f"Evaluation complete. Results saved to {csv_filename}")



def main_evaluation_experiment(rl_weights_path):

    model_name = "Best_RL"
    results_path = Path(f"{model_name}_results.csv")
    if not results_path.exists():
        with open(results_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['episode', 
                             'normalized_cumulative_reward', 
                             'final_rewards', 
                             'best_reward', 
                             'n_agents',
                             'comm_radius',
                             'movement_noise',
                             'sample_noise',
                             'model'])

    for n_agents in [5, 10, 20]:
        for comm_radius in [0.0, 10.0, 20.0, 50.0]:
            for movement_noise in [0.01, 0.05, 0.1]:
                for sample_noise in [0.01, 0.05, 0.1]:

                    print(f"Running evaluation with n_agents={n_agents}, "
                          f"comm_radius={comm_radius}, "
                          f"movement_noise={movement_noise}, "
                          f"sample_noise={sample_noise}")

                    main_hybrid_evaluation(
                        rl_weights_path=rl_weights_path,
                        n_agents=n_agents,
                        comm_radius=comm_radius,
                        movement_noise=movement_noise,
                        sample_noise=sample_noise,
                        render=False,
                        model_name=f"Best_RL",
                    )


if __name__ == "__main__":

    simple_play = False
    if simple_play:
        main_hybrid_evaluation("weights/shared_agent_shared_network_10.pth", 
                            n_agents = 10,
                            comm_radius = 10.0,
                            movement_noise = 0.01,
                            sample_noise = 0.01,
                            render = True,
                            model_name = None, 
                            max_steps = 250,
                            num_episodes = 100)
    else:
        main_evaluation_experiment("weights/shared_agent_shared_network_10.pth")
