import torch
import numpy as np
import matplotlib.pyplot as plt
from environment import EnvArgs2D, Environment2D
from agent import Agent, PPO_Buffer
from torch.utils.tensorboard import SummaryWriter
import time
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def main_training_loop():
    # Training parameters
    n_agents = 5
    local_steps = 64
    n_episodes = 10000
    save_every_episodes = 100
    batch_size = 32
    agent_colors = plt.cm.tab10(np.linspace(0, 1, n_agents))

    eval_train = True
    
    # Environment and agent parameters
    record_params = {
        'id': "2D_PPO_80_speed001",
        'temp_memory': 40,
        'max_steps': 128,
        'velocity': 0.1,
        'angular_velocity': 45,  # degrees per action
        'entropy_loss_coeff': 0.01,
        'critic_loss_coeff': 0.5,
        'gamma': 0.8,
        'movement_noise': 0.005,
        'net': 'mlp',  # 'mlp' or 'transformer'
        'num_actions': 8 + 1  # 24 directions + stand still
    }
    
    # Create weights directory if not exists
    os.makedirs("weights", exist_ok=True)
    
    # Initialize logging
    writer = SummaryWriter(log_dir=f"runs/{record_params['id']}")
    
    # Environment setup
    env_args = EnvArgs2D(
        n_actors=n_agents,
        velocity=record_params['velocity'],
        angular_velocity=record_params['angular_velocity'],
        movement_noise=record_params['movement_noise'],
        max_steps=record_params['max_steps'],
        starting_position_mean=(0, 0),
        starting_position_var=10,
        num_actions=record_params['num_actions']
    )
    env = Environment2D(env_args)
    
    # Initialize agents
    agents = [
        Agent(
            net=record_params['net'],
            temp_memory=record_params['temp_memory'],
            num_actions=record_params['num_actions'],
            device=device
        ) 
        for _ in range(n_agents)
    ]
    
    # Initialize buffers
    buffers = [
        PPO_Buffer(
            gamma=record_params['gamma'],
            entropy_loss_coeff=record_params['entropy_loss_coeff'],
            critic_loss_coeff=record_params['critic_loss_coeff']
        )
        for _ in range(n_agents)
    ]
    
    # ==============================================
    # Setup for visualizations
    # ==============================================
    if eval_train:
        plt.ion()
        
        # Create separate figures for metrics and policies
        metrics_fig = plt.figure(figsize=(12, 6))
        metrics_fig.suptitle("Agent Performance Metrics", fontsize=16)
        
        # Metrics figure layout (value and cumulative rewards)
        ax_value = plt.subplot(121)
        ax_value.set_title("Value Function Estimates (Last 100 Steps)")
        ax_value.set_xlabel("Step")
        ax_value.set_ylabel("Value")
        ax_value.grid(True)
        
        ax_cumulative = plt.subplot(122)
        ax_cumulative.set_title("Cumulative Rewards (Full Episode)")
        ax_cumulative.set_xlabel("Step")
        ax_cumulative.set_ylabel("Reward")
        ax_cumulative.grid(True)
        
        # Create lines for each agent
        value_lines = []
        cum_lines = []
        for i in range(n_agents):
            line1, = ax_value.plot([], [], color=agent_colors[i], label=f"Agent {i}")
            line2, = ax_cumulative.plot([], [], color=agent_colors[i], label=f"Agent {i}")
            value_lines.append(line1)
            cum_lines.append(line2)
        
        ax_value.legend()
        ax_cumulative.legend()
        
        # Create separate figure for policy visualizations
        policy_fig = plt.figure(figsize=(16, 4))
        policy_fig.suptitle("Agent Policies", fontsize=16)
        
        # Create horizontal layout for policy plots
        policy_axes = []
        for i in range(n_agents):
            ax = policy_fig.add_subplot(1, n_agents, i+1, projection='polar')
            ax.set_title(f"Agent {i}", pad=15)
            policy_axes.append(ax)
        
        # Initialize policy visualization data
        num_directions = record_params['num_actions'] - 1
        theta = np.linspace(0, 2*np.pi, num_directions, endpoint=False)
        radii = np.zeros(num_directions)
        width = 2 * np.pi / num_directions
        
        # Create policy bars with agent colors
        policy_bars = []
        for i, ax in enumerate(policy_axes):
            # Create bars with agent-specific color and inner radius
            bars = ax.bar(theta, radii, width=width, bottom=0.2, 
                        color=agent_colors[i], alpha=0.7, edgecolor='k')
            policy_bars.append(bars)
            
            # Set radial limits to 0-1 for consistent scaling
            ax.set_ylim(0, 1)
            ax.set_yticklabels([])
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            
            # Clear center area
            ax.add_patch(plt.Circle((0, 0), 0.2, color='white', zorder=10))
        
        metrics_fig.tight_layout()
        
        # Data storage for visualizations
        value_history = [[] for _ in range(n_agents)]
        cum_reward_history = [[] for _ in range(n_agents)]
        last_policy = [None] * n_agents
        last_actions = [None] * n_agents
    
    # ==============================================
    # Training loop
    # ==============================================
    start_time = time.time()
    last_render_time = time.time()
    render_interval = 0.1  # seconds
    
    for episode in range(n_episodes):
        positions = env.reset()
        done = False
        cumulative_rewards = torch.zeros(n_agents)
        best_rewards = torch.zeros(n_agents)
        
        # Reset visualization data for new episode
        if eval_train:
            for i in range(n_agents):
                value_history[i] = []
                cum_reward_history[i] = []
        
        step_count = 0
        
        while not done:
            current_time = time.time()
            action_idxs, actions, values, log_policies, policies = [], [], [], [], []
            
            # Get actions from agents
            for i, agent in enumerate(agents):
                policy, value = agent.act()
                action_idx = torch.multinomial(policy, 1).item()
                
                action_idxs.append(action_idx)
                actions.append(env.action_vectors[action_idx])
                values.append(value)
                log_policy = torch.log(policy[action_idx])
                log_policies.append(log_policy)
                policies.append(policy)
                
                # Store policy and action for visualization
                if eval_train:
                    last_policy[i] = policy.detach().cpu().numpy()
                    last_actions[i] = action_idx
            
            # Convert to tensors
            action_idxs_tensor = torch.tensor(action_idxs)
            actions_tensor = torch.stack(actions)
            
            # Environment step
            positions, rewards, done = env.step(action_idxs_tensor)
            
            # Update agent memories and buffers
            for i, agent in enumerate(agents):
                internal_state = agent.network.memory_buffer.clone().detach()
                buffers[i].add(
                    internal_state,
                    action_idxs_tensor[i],
                    log_policies[i],
                    rewards[i],
                    values[i],
                    done
                )
                agent.update_memory(action_idxs_tensor[i], rewards[i])
            
            # Update rewards
            cumulative_rewards += rewards
            best_rewards = torch.maximum(best_rewards, rewards)
            
            # viz
            if eval_train:
                for i in range(n_agents):
                    value_history[i].append(values[i].item())
                    cum_reward_history[i].append(cumulative_rewards[i].item())

                    if len(value_history[i]) > 100:
                        value_history[i] = value_history[i][-100:]
            
            # Update networks periodically
            if done or (env.time_elapsed % local_steps == 0 and env.time_elapsed > 0):
                critic_losses, actor_losses, entropy_losses = [], [], []
                
                for i, agent in enumerate(agents):
                    last_value = agent.act()[1] if not done else torch.tensor([0.0]).to(device)
                    critic_loss, actor_loss, entropy_loss = agent.update(
                        buffers[i], last_value, batch_size=batch_size
                    )
                    
                    critic_losses.append(critic_loss)
                    actor_losses.append(actor_loss)
                    entropy_losses.append(entropy_loss)
                
                # Log losses
                if not eval_train:
                    writer.add_scalar('Loss/Critic', np.mean(critic_losses), episode)
                    writer.add_scalar('Loss/Actor', np.mean(actor_losses), episode)
                    writer.add_scalar('Loss/Entropy', np.mean(entropy_losses), episode)
            
            # Update visualizations at appropriate intervals
            if eval_train:
                if current_time - last_render_time > render_interval:
                    # Update value function plot
                    for i, line in enumerate(value_lines):
                        if value_history[i]:
                            x_data = np.arange(len(value_history[i]))
                            line.set_data(x_data, value_history[i])
                            ax_value.relim()
                            ax_value.autoscale_view()
                    
                    # Update cumulative reward plot (full episode)
                    for i, line in enumerate(cum_lines):
                        if cum_reward_history[i]:
                            x_data = np.arange(len(cum_reward_history[i]))
                            line.set_data(x_data, cum_reward_history[i])
                            ax_cumulative.relim()
                            ax_cumulative.autoscale_view()
                    
                    # Update metrics figure
                    metrics_fig.canvas.draw()
                    metrics_fig.canvas.flush_events()
                    
                    # Update policy visualization
                    for i, bars in enumerate(policy_bars):
                        if last_policy[i] is not None:
                            policy = last_policy[i]
                            action_idx = last_actions[i]
                            
                            # Update directional probabilities
                            for j, bar in enumerate(bars):
                                # Set height scaled for visibility
                                height = policy[j] * 5
                                bar.set_height(height)
                                
                                # Highlight chosen action in red
                                if j == action_idx:
                                    bar.set_facecolor('red')
                                    bar.set_edgecolor('darkred')
                                    bar.set_alpha(1.0)
                                else:
                                    bar.set_facecolor(agent_colors[i])
                                    bar.set_edgecolor('black')
                                    bar.set_alpha(0.7)
                    
                    # Update policy figure
                    policy_fig.canvas.draw()
                    policy_fig.canvas.flush_events()
                    
                    # Update environment rendering
                    env.render(rewards)
                    
                    last_render_time = current_time
            
            step_count += 1
        
        # End of episode
        mean_cumulative = cumulative_rewards.mean().item()
        mean_best = best_rewards.mean().item()
        mean_final = rewards.mean().item()
        
        # Log rewards
        if not eval_train:
            writer.add_scalar('Reward/Cumulative', mean_cumulative, episode)
            writer.add_scalar('Reward/Best', mean_best, episode)
            writer.add_scalar('Reward/Final', mean_final, episode)
        
        print(f"Episode {episode:4d} | "
              f"Cum: {mean_cumulative:6.2f} | "
              f"Best: {mean_best:5.2f} | "
              f"Final: {mean_final:5.2f} | "
              f"Steps: {step_count}")
        
        # Reset agent memories
        for agent in agents:
            agent.reset_memory()
        
        # Save models periodically
        if episode % save_every_episodes == 0 and episode > 0 and not eval_train:
            for i, agent in enumerate(agents):
                agent.save_model(f"weights/agent_{i}_{record_params['id']}_ep{episode}.pth")
    
    # Final save
    for i, agent in enumerate(agents):
        agent.save_model(f"weights/agent_{i}_{record_params['id']}_final.pth")
    
    writer.close()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main_training_loop()