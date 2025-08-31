import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import os
from tqdm import trange
from environment import EnvArgs2D, Environment2D
from shared_agent import Agent, PPO_Buffer

# Device selection
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



def compute_graph(positions, radius, device):
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    dist = torch.norm(diff, dim=-1)
    mask = (dist < radius) & (dist > 0)
    edge_index = torch.nonzero(mask, as_tuple=False).t().contiguous()

    if edge_index.size(1) > 0:
        vec = positions[edge_index[1]] - positions[edge_index[0]]
        distances = torch.norm(vec, dim=1, keepdim=True)
        angles = torch.atan2(vec[:, 1], vec[:, 0]).unsqueeze(1)
        edge_attr = torch.cat([distances, angles], dim=1)
    else:
        edge_attr = torch.zeros((0, 2), device=device)

    return edge_index, edge_attr, dist


def main_training_loop(simple_play=False):

    n_agents = 5
    local_steps = 64
    n_episodes = 20000
    batch_size = 128
    temp_memory = 10
    agent_colors = plt.cm.tab10(np.linspace(0, 1, n_agents))

    args = EnvArgs2D(
        n_actors=n_agents,
        velocity=0.5,
        angular_velocity=45,
        movement_noise=0.005,
        max_steps=256,
        starting_position_mean=(0, 0),
        starting_position_var=10,
        num_actions=9                     )
    env = Environment2D(args)

    shared_agent = Agent(
        net="mlp",
        temp_memory=temp_memory,
        num_actions=args.num_actions,
        n_agents=n_agents,
        device=device,
    )
    shared_buffer = PPO_Buffer(
        gamma=0.95,
        entropy_loss_coeff=0.05,
        critic_loss_coeff=0.5,
        n_agents=n_agents
    )

    if not simple_play:
        csv_file = f"{temp_memory}_temp_memory_RL_training_metrics.csv"
        write_header = not os.path.exists(csv_file)
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "episode",
                    "normalized_cumulative_reward",
                    "final_rewards",
                    "best_reward",
                    "mean_normalized_cumulative_reward",
                    "mean_final_rewards",
                    "mean_best_reward",
                    "memory_size"
                ])


    pbar = trange(n_episodes, desc=f"Running {n_episodes} episodes", unit="episode")
    for episode in pbar:

        positions = env.reset()
        done = False
        cumulative_rewards = torch.zeros(n_agents)
        best_rewards = torch.zeros(n_agents)
        total_steps = 0

        while not done:

            policies, values = shared_agent.act()
            action_idxs = torch.multinomial(policies, 1).squeeze()
            action_reprs = torch.tensor([action_idx_to_repr[idx.item()] for idx in action_idxs]) 
            log_policies = torch.log(policies.gather(1, action_idxs.unsqueeze(1))).squeeze()
            positions, rewards, done = env.step(action_idxs.to("cpu"))
            shared_agent.update_memory(action_reprs, rewards)

            cumulative_rewards += rewards
            best_rewards = torch.maximum(best_rewards, rewards)

            if not simple_play:
                shared_buffer.add_batch(
                    shared_agent.network.memory_buffer.clone(),
                    action_idxs,
                    log_policies,
                    rewards,
                    values.squeeze(),
                    torch.tensor([done] * n_agents)
                )

            if simple_play:
                env.render(rewards)

            total_steps += 1

            if not simple_play and (done or (env.time_elapsed % local_steps == 0 and env.time_elapsed > 0)):

                _, last_values = shared_agent.act()
                if done: 
                    last_values = torch.zeros_like(last_values)

                critic_loss, actor_loss, entropy_loss = shared_agent.update(
                    shared_buffer, last_values.detach().T, batch_size=batch_size
                )

        normalized_cumulative_reward = (cumulative_rewards / total_steps)

        if not simple_play:
            with open(csv_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode,
                    torch.max(normalized_cumulative_reward).item(),
                    torch.max(rewards).item(),
                    torch.max(best_rewards).item(),
                    normalized_cumulative_reward.mean().item(),
                    rewards.mean().item(),
                    best_rewards.mean().item(),
                    temp_memory
                ])


if __name__ == "__main__":

    main_training_loop(simple_play=False)
