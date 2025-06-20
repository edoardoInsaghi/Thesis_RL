import torch
from environment import EnvArgs1D, Environment1D
from agent import Agent1D, PPO_Buffer, DummyAgent
import matplotlib.pyplot as plt
from matplotlib import cm
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")
print(f'Using device {device}')


def dummy_agent_eval():
    
    n_agents = 5
    n_observations = 2
    n_episodes = 1000

    render = True

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

    agents = [DummyAgent() for _ in range(n_agents)]

    all_rewards = []

    for episode in range(1, n_episodes+1):

        state = env.reset()
        done = False
        cumulative_rewards = torch.zeros(n_agents, requires_grad=False)
        rewards = torch.zeros(n_agents, requires_grad=False)

        while not done:

            actions = []
            for i, agent in enumerate(agents):

                action = agent.act(rewards[i])
                actions.append(action)

            state, rewards, info, done = env.step(torch.stack(actions).cpu())
            rewards = rewards.float()
            cumulative_rewards += rewards 

            if render:
                env.render(rewards)

        print(f"Episode {episode}/{n_episodes} | Avg Reward: {cumulative_rewards.mean().item():.2f}")
        all_rewards.append(cumulative_rewards.clone().mean().cpu())
        print(f"Episode {episode}/{n_episodes} | Avg Reward: {torch.mean(torch.stack(all_rewards))}")
        



def main_training_loop():

    n_agents = 5
    local_steps = 256 # number of steps before updating the policy
    n_episodes = 200000
    save_every_episodes = 250
    batch_size = 64
    agent_colors = cm.tab10([i/n_agents for i in range(n_agents)])

    render = False
    save_params = True
    log_results = True

    record_params = {
        'id': "PPO_1",
        'temp_memory': 80,
        'max_steps': 1537,
        'velocity': 0.05,
        'entropy_loss_coeff': 0.05,
        'critic_loss_coeff': 0.1,
        'gamma': 0.7,
        'movement_noise': 0.0,
        'net': 'mlp'
    }

    if log_results:
        writer = SummaryWriter(log_dir=f"runs/{record_params['id']}")
        with open("training_params.txt", "a") as f: f.write(str({k: v for k, v in record_params.items()}))


    env_args = EnvArgs1D(
        cosines=20,
        n_actors=n_agents,
        velocity=record_params['velocity'],
        max_steps=record_params['max_steps'], # number of steps per episode
        starting_position_mean=0,
        starting_position_var=7,
        movement_noise=record_params['movement_noise'],
    )
    env = Environment1D(env_args)

    agents = [Agent1D(net=record_params['net'],
                      temp_memory=record_params['temp_memory'],
                      device=device,
                      weights=None,
                      #weights=f"weights/agent_{i}_{record_params[id]}" if i < n_agents else None
                      )
              for i in range(n_agents)]
    
    buffers = [PPO_Buffer(entropy_loss_coeff=record_params['entropy_loss_coeff'], 
                          critic_loss_coeff=record_params['critic_loss_coeff'],
                          gamma=record_params['gamma']) 
               for _ in range(n_agents)]


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
    for episode in range(1, n_episodes+1):

        state = env.reset()
        done = False
        episode_values = []
        cumulative_rewards = torch.zeros(n_agents, requires_grad=False)
        rewards = torch.zeros(n_agents, requires_grad=False)
        reward_history = []
        best_rewards = torch.zeros(n_agents, requires_grad=False)
        final_rewards = torch.zeros(n_agents, requires_grad=False)

        while not done:
            
            for _ in range(local_steps):

                action_idxs, actions, values, log_policies, policies, hidden_states = [], [], [], [], [], []
                for i, agent in enumerate(agents):

                    policy, value = agent.act() # does not retain grad
                    log_policy = torch.log(policy)
                    action_idx = torch.multinomial(policy, 1)

                    action_idxs.append(action_idx)
                    actions.append((action_idx - 1).float())   # 0 -> -1 (left), 1 -> 0 (stand), 2 -> 1 (right)
                    values.append(value)
                    log_policies.append(log_policy[action_idx])
                    policies.append(policy)

                joint_actions = torch.tensor(actions).to("cpu")
                state, rewards, done = env.step(joint_actions) # joint transition step

                for i, agent in enumerate(agents):

                    internal_state = agent.network.memory_buffer.clone().detach()
                    buffers[i].add(internal_state, action_idxs[i], 
                                    log_policies[i], rewards[i], 
                                    values[i], done)
                    agent.update_memory(actions[i], rewards[i])

                cumulative_rewards += rewards
                episode_values.append(torch.stack(values).detach().cpu())
                reward_history.append(cumulative_rewards.clone().cpu())
                best_rewards = torch.maximum(best_rewards, rewards)


                # # Plotting stuff # #
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
                    
            # End of local steps, updating network and log losses  
            critic_losses = []
            actor_losses = []
            entropy_losses = []
            
            for i, agent in enumerate(agents):
                last_value = agent.act()[1] if not done else torch.zeros_like(agent.act()[1])
                avg_critic_loss, avg_actor_loss, avg_entropy = agent.update(
                    buffers[i], last_value, batch_size=batch_size, shuffle=False
                )
                critic_losses.append(avg_critic_loss)
                actor_losses.append(avg_actor_loss)
                entropy_losses.append(avg_entropy)

            mean_critic = sum(critic_losses) / n_agents
            mean_actor = sum(actor_losses) / n_agents
            mean_entropy = sum(entropy_losses) / n_agents
            updates += 1

            if log_results:
                writer.add_scalar('Loss/Critic_avg', mean_critic, updates)
                writer.add_scalar('Loss/Actor_avg', mean_actor, updates)
                writer.add_scalar('Loss/Entropy_avg', mean_entropy, updates)

        # End of episode, logging results
        final_rewards = rewards.clone()
        mean_cumulative = cumulative_rewards.mean().item()
        mean_best = best_rewards.mean().item()
        mean_final = final_rewards.mean().item()
        
        if log_results:
            writer.add_scalar('Rewards/Cumulative_avg', mean_cumulative, episode)
            writer.add_scalar('Rewards/Best_avg', mean_best, episode)
            writer.add_scalar('Rewards/Final_avg', mean_final, episode)
            
        print(
            f"Episode {episode} | "
            f"Cumulative: {mean_cumulative:.2f} | "
            f"Best: {mean_best:.2f} | "
            f"Final: {mean_final:.2f} | "
        )

        for i, agent in enumerate(agents):
            agent.reset_memory()
            if episode % save_every_episodes == 0 and save_params:
                agent.save_model(f"weights/agent_{i}_{record_params['id']}.pth")

    if save_params:
        for i in range(n_agents):
            agents[i].save_model(f"weights/agent_{i}_{record_params['id']}.pth")


if __name__ == "__main__":
    #dummy_agent_eval()
    main_training_loop()
