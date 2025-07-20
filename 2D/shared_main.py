import torch
import numpy as np
import matplotlib.pyplot as plt
from environment import EnvArgs2D, Environment2D
from shared_agent import Agent, PPO_Buffer
from torch.utils.tensorboard import SummaryWriter
import time
import os

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

def main_training_loop():
    n_agents = 5
    local_steps = 256
    n_episodes = 100000
    save_every_episodes = 250
    batch_size = 128
    agent_colors = plt.cm.tab10(np.linspace(0, 1, n_agents))

    eval_train = True
    
    record_params = {
        'id': "shared_network_fast",
        'temp_memory': 20,
        'max_steps': 1024,
        'velocity': 0.75,
        'angular_velocity': 45,
        'entropy_loss_coeff': 0.05,
        'critic_loss_coeff': 0.5,
        'gamma': 0.95,
        'movement_noise': 0.005,
        'net': 'mlp',
        'num_actions': 8 + 1
    }
    
    os.makedirs("weights", exist_ok=True)
    writer = SummaryWriter(log_dir=f"runs/{record_params['id']}")
    
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
    
    shared_agent = Agent(
        net=record_params['net'],
        temp_memory=record_params['temp_memory'],
        num_actions=record_params['num_actions'],
        n_agents=n_agents,
        device=device,
        #weights=f"weights/shared_agent_{record_params['id']}.pth",
        weights=None 
    )

    shared_buffer = PPO_Buffer(
        gamma=record_params['gamma'],
        entropy_loss_coeff=record_params['entropy_loss_coeff'],
        critic_loss_coeff=record_params['critic_loss_coeff'],
        n_agents=n_agents
    )
    
    # ==============================================
    # Setup for visualizations
    # ==============================================
    if eval_train:
        plt.ion()
        
        metrics_fig = plt.figure(figsize=(12, 6))
        metrics_fig.suptitle("Agent Performance Metrics", fontsize=16)
        
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
        
        value_lines = []
        cum_lines = []
        for i in range(n_agents):
            line1, = ax_value.plot([], [], color=agent_colors[i], label=f"Agent {i}")
            line2, = ax_cumulative.plot([], [], color=agent_colors[i], label=f"Agent {i}")
            value_lines.append(line1)
            cum_lines.append(line2)
        
        ax_value.legend()
        ax_cumulative.legend()
        
        policy_fig = plt.figure(figsize=(16, 4))
        policy_fig.suptitle("Agent Policies", fontsize=16)
        
        policy_axes = []
        for i in range(n_agents):
            ax = policy_fig.add_subplot(1, n_agents, i+1, projection='polar')
            ax.set_title(f"Agent {i}", pad=15)
            policy_axes.append(ax)
        
        num_directions = record_params['num_actions'] - 1
        theta = np.linspace(0, 2*np.pi, num_directions, endpoint=False)
        radii = np.zeros(num_directions)
        width = 2 * np.pi / num_directions
        
        policy_bars = []
        for i, ax in enumerate(policy_axes):
            bars = ax.bar(theta, radii, width=width, bottom=0.2, 
                        color=agent_colors[i], alpha=0.7, edgecolor='k')
            policy_bars.append(bars)
            
            ax.set_ylim(0, 1)
            ax.set_yticklabels([])
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            
            ax.add_patch(plt.Circle((0, 0), 0.2, color='white', zorder=10))
        
        metrics_fig.tight_layout()
        
        value_history = [[] for _ in range(n_agents)]
        cum_reward_history = [[] for _ in range(n_agents)]
        last_policy = [None] * n_agents
        last_actions = [None] * n_agents
    
    # ==============================================
    # Training loop
    # ==============================================
    start_time = time.time()
    last_render_time = time.time()
    render_interval = 0.1
    updates = 0
    
    for episode in range(n_episodes):
        positions = env.reset()
        done = False
        cumulative_rewards = torch.zeros(n_agents)
        best_rewards = torch.zeros(n_agents)
        step_count = 0
        
        # Reset visualization data for new episode
        if eval_train:
            for i in range(n_agents):
                value_history[i] = []
                cum_reward_history[i] = []
        
        while not done:
            current_time = time.time()
            
            # Batched act
            policies, values = shared_agent.act()
            action_idxs = torch.multinomial(policies, 1).squeeze()
            
            # Convert action indices to representations
            action_reprs = torch.tensor([action_idx_to_repr[idx.item()] for idx in action_idxs])
            
            # Environment step
            positions, rewards, done = env.step(action_idxs.to("cpu"))
            
            # Calculate log probabilities
            log_policies = torch.log(policies.gather(1, action_idxs.unsqueeze(1))).squeeze()
            
            # Add batch to buffer
            shared_buffer.add_batch(
                shared_agent.network.memory_buffer.clone(),
                action_idxs,
                log_policies,
                rewards,
                values.squeeze(),
                torch.tensor([done] * n_agents)
            )
            
            # Update memory
            shared_agent.update_memory(action_reprs, rewards)
            
            # Update rewards
            cumulative_rewards += rewards
            best_rewards = torch.maximum(best_rewards, rewards)
            
            # Collect visualization data
            if eval_train:
                for i in range(n_agents):
                    value_history[i].append(values[i].item())
                    cum_reward_history[i].append(cumulative_rewards[i].item())
                    last_policy[i] = policies[i].detach().cpu().numpy()
                    last_actions[i] = action_idxs[i].item()
                    
                    if len(value_history[i]) > 100:
                        value_history[i] = value_history[i][-100:]
            
            if done or (env.time_elapsed % local_steps == 0 and env.time_elapsed > 0):
                updates += 1
            
                _, last_values = shared_agent.act()
                if done: 
                    last_values = torch.zeros_like(last_values)

                critic_loss, actor_loss, entropy_loss = shared_agent.update(
                    shared_buffer, last_values.detach().T, batch_size=batch_size
                )
                
                if not eval_train:
                    writer.add_scalar('Loss/Critic', critic_loss, updates)
                    writer.add_scalar('Loss/Actor', actor_loss, updates)
                    writer.add_scalar('Loss/Entropy', entropy_loss, updates)
            
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
        shared_agent.reset_memory()
        
        # Save model periodically
        if episode % save_every_episodes == 0 and episode > 0 and not eval_train:
            shared_agent.save_model(f"weights/shared_agent_{record_params['id']}.pth")
    
    # Final save
    shared_agent.save_model(f"weights/shared_agent_{record_params['id']}.pth")
    writer.close()
    
    if eval_train:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main_training_loop()