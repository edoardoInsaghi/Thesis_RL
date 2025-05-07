import torch
from environment import EnvArgs1D, Environment1D
from agent import Agent1D, PPO_Buffer
import matplotlib.pyplot as plt
from matplotlib import cm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="runs/1D_ppo_4")

device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")
print(f'Using device {device}')


def main_training_loop():

    n_agents = 5
    n_observations = 2 # number of chemicals sampled by each agent
    local_steps = 128 # number of steps before updating the policy
    n_episodes = 2000
    batch_size = 64
    agent_colors = cm.tab10([i/n_agents for i in range(n_agents)])

    render = False

    env_args = EnvArgs1D(
        n_actors=n_agents,
        n_observations=n_observations,
        velocity=0.05,
        max_steps=500, # number of steps per episode
        starting_position_mean=0,
        starting_position_var=7,
        movement_noise=0.0
    )
    env = Environment1D(env_args)

    agents = [Agent1D(n_dim=n_observations, 
                      temp_memory=40,
                      n_hidden=256, 
                      device=device,
                      weights=None)
                      #weights=f"weights/agent_{i}.pth" if i < n_agents else None)
              for i in range(n_agents)]
    
    buffers = [PPO_Buffer(entropy_loss_coeff=0.5) for _ in range(n_agents)]


    # # Plotting setup # #
    if render:
        
        # Rewards and Value Estimates
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
        value_lines = [ax1.plot([], [], color=c, label=f'Agent {i+1}')[0] for i, c in enumerate(agent_colors)]
        reward_lines = [ax2.plot([], [], color=c, label=f'Agent {i+1}')[0] for i, c in enumerate(agent_colors)]
        ax1.set(title='Value Estimates', xlabel='Time Step', ylabel='Value')
        ax2.set(title='Episode Cumulative Reward', xlabel='Time Step', ylabel='Reward')
        for ax in (ax1, ax2):
            ax.grid(True)
            ax.legend()
        plt.tight_layout()
        plt.show(block=False)

        # Policies
        policy_fig, policy_ax = plt.subplots(1, n_agents, figsize=(8, 6))
        plt.show(block=False)
    # # # # # # # # # #


    updates = 0
    for episode in range(n_episodes):

        state = env.reset()
        done = False
        episode_values = []
        cumulative_rewards = torch.zeros(n_agents, requires_grad=False)
        rewards = torch.zeros(n_agents, requires_grad=False)
        reward_history = []

        while not done:
            
            for _ in range(local_steps):

                actions, values, log_policies, action_idxs, policies = [], [], [], [], []
                for i, agent in enumerate(agents):

                    policy, value = agent.act() # does not retain grad
                    log_policy = torch.log(policy)
                    action_idx = torch.multinomial(policy, 1)

                    action_idxs.append(action_idx)
                    actions.append((action_idx - 1).float()) # 0 -> -1 (left), 1 -> 0 (stand), 2 -> 1 (right)
                    values.append(value)
                    log_policies.append(log_policy[action_idx])
                    policies.append(policy)

                    agent.memory_buffer = torch.cat((agent.memory_buffer[1:], actions[-1]))

                state, rewards, info, done = env.step(torch.stack(actions).cpu().squeeze(1)) # joint transition step
                rewards = rewards.float()

                for i, agent in enumerate(agents):
                    internal_state = agent.memory_buffer.clone().detach()
                    agent.memory_buffer = torch.cat((agent.memory_buffer[1:], rewards[i].unsqueeze(0).to(device)))
                    buffers[i].add(internal_state, action_idxs[i], log_policies[i], rewards[i], values[i], done)

                cumulative_rewards += rewards
                episode_values.append(torch.stack(values).detach().cpu())
                reward_history.append(cumulative_rewards.clone().cpu())            

                ## Plotting stuff ##
                if render:

                    # Rewards and Value Estimates
                    window = 100
                    values_to_plot = torch.stack(episode_values[-window:])
                    for i, line in enumerate(value_lines):
                        y_data = values_to_plot[:, i].numpy()
                        x_data = torch.arange(len(y_data))
                        line.set_data(x_data, y_data)
                    ax1.relim()
                    ax1.autoscale_view()

                    rewards_to_plot = torch.stack(reward_history[-window*10:])
                    for i, line in enumerate(reward_lines):
                        y_data = rewards_to_plot[:, i].numpy()
                        x_data = torch.arange(len(y_data))
                        line.set_data(x_data, y_data)
                    ax2.relim()
                    ax2.autoscale_view()

                    # Policies
                    for i, (policy, ax) in enumerate(zip(policies, policy_ax)):
                        ax.clear()
                        ax.set_title(f'Agent {i+1}')
                        if i == 0:
                            ax.set_ylabel('Probability')
                        ax.set_ylim(0, 1)
                        ax.grid(True)
                        colors = ["red" if j == action_idxs[i] else agent_colors[i] for j in range(len(policy))]
                        ax.bar(torch.arange(len(policy)), policy.cpu().numpy(), color=colors)

                    env.render(rewards)
                # # # # # # # # # #


            avg_critic_loss_tot, avg_actor_loss_tot, avg_entropy_tot = 0, 0, 0
            for i, agent in enumerate(agents):
                last_value = agent.act()[1] if not done else torch.tensor([0], device=device)
                avg_critic_loss, avg_actor_loss, avg_entropy = agent.update(buffers[i], last_value, batch_size=batch_size)
                avg_critic_loss_tot += avg_critic_loss
                avg_actor_loss_tot += avg_actor_loss
                avg_entropy_tot += avg_entropy
                writer.add_scalar(f'Loss/Critic_{i+1}', avg_critic_loss, updates)
                writer.add_scalar(f'Loss/Actor_{i+1}', avg_actor_loss, updates)
                writer.add_scalar(f'Loss/Entropy_{i+1}', avg_entropy, updates)

            updates += 1

            avg_critic_loss_tot /= n_agents
            avg_actor_loss_tot /= n_agents
            avg_entropy_tot /= n_agents

            writer.add_scalar('Loss/Critic_avg', avg_critic_loss_tot, updates)
            writer.add_scalar('Loss/Actor_avg', avg_actor_loss_tot, updates)
            writer.add_scalar('Loss/Entropy_avg', avg_entropy_tot, updates)

        print(f"Episode {episode+1}/{n_episodes} | Avg Reward: {cumulative_rewards.mean().item():.2f}")

        for i, agent in enumerate(agents):
            agent.reset_memory()
            # agent.save_model(f"weights/agent_{i}.pth")

    for i in range(n_agents):
        agents[i].save_model(f"weights/agent_{i}.pth")
        print(f"Agent {i} model saved.")


if __name__ == "__main__":
    main_training_loop()