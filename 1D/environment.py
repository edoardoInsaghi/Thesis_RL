import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import inspect

def f1(x):
    return 0.5 * x**2 + 2 * torch.sin(3*x) + torch.cos(5*x)

def f2(x):
    return 0.3 * x**2 - 4 * torch.cos(2*x) + torch.abs(torch.sin(x))**3


class EnvArgs1D:

    def __init__(self, 
                 n_actors: int = 5, 
                 n_observations: int = 2, 
                 observations=None, 
                 reward_function_args=None,
                 velocity: float = 0.01, 
                 movement_noise: float = 0.001, 
                 max_steps: int = 100000,
                 starting_position_mean: float = 0, 
                 starting_position_var: float = 5):
        
        self.n_actors = n_actors
        self.n_observations = n_observations
        self.observations = observations if observations is not None else [f1, f2]
        self.reward_function_args = reward_function_args if reward_function_args is not None else [
            {"low": 5.0, "high": 40.0, "sharpness": .5} for _ in range(n_observations)
        ]
        self.starting_position_mean = starting_position_mean
        self.starting_position_var = starting_position_var
        self.velocity = velocity
        self.movement_noise = movement_noise
        self.max_steps = max_steps


class Environment1D:

    def __init__(self, args: EnvArgs1D):

        self.args = args
        self.n_actors = args.n_actors
        self.observations = args.observations
        self.n_observations = args.n_observations
        self.reward_function = self.smooth_window
        self.reward_function_args = args.reward_function_args
        self.starting_position_mean = args.starting_position_mean
        self.starting_position_var = args.starting_position_var
        self.positions = torch.normal(mean=self.starting_position_mean, std=self.starting_position_var, size=(self.n_actors,))
        self.coeff_velocity = args.velocity
        self.coeff_movement_noise = args.movement_noise
        self.time_elapsed = 0
        self.done = False

        # Rendering
        plt.ion() 
        self.fig = None
        self.axs = None
        self.scatters = [] 
        self.lines = [] 
        self.agent_colors = cm.tab10([i/args.n_actors for i in range(args.n_actors)]) 

        assert len(self.observations) == self.n_observations


    def step(self, action: torch.Tensor): # (n_actors, n_dim = 1) -> (n_actors, n_observations), (n_actors, 1), (n_actors, n_dim = 1), (1, )

        self.positions = self.positions + self.coeff_velocity * action + torch.normal(mean=0.0, std=self.coeff_movement_noise, size=(self.n_actors,))

        observations = torch.zeros((self.n_actors, self.n_observations))
        rewards = torch.zeros((self.n_actors, self.n_observations))
        for j in range(self.n_actors): # TODO: vectorize this
            for i in range(self.n_observations):
                observations[j, i] = self.observations[i](self.positions[j])
                rewards[j, i] = self.reward_function(observations[j, i], **self.reward_function_args[i])

        rewards = rewards.mean(dim=1)
        
        self.time_elapsed += 1
        if self.time_elapsed >= self.args.max_steps:
            self.done = True

        return observations, rewards, self.positions, self.done
    

    def smooth_window(self, x, **kwargs):
        low = kwargs.get("low", 5.0)
        high = kwargs.get("high", 10.0)
        sharpness = kwargs.get("sharpness", 0.1)
        return torch.sigmoid(sharpness * (x - low)) * torch.sigmoid(sharpness * (high - x))

    
    def _init_plot(self):
        if self.fig is None:
            self.fig = plt.figure(figsize=(8, 6))
            self.ax = self.fig.add_subplot(111)
            self.ax.grid(True)
            self.ax.set_xlabel('Position')
            self.ax.set_ylabel('Reward')
            
            self.scatter = self.ax.scatter(
                self.positions.numpy(), 
                torch.zeros_like(self.positions).numpy(),
                c=self.agent_colors,
                edgecolors='black', 
                s=100, 
                zorder=10
            )
            self.line, = self.ax.plot([], [], lw=2)
            
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.001)

    def render(self, rewards=None):
        self._init_plot()
        
        min_pos = self.positions.min().item()
        max_pos = self.positions.max().item()
        padding = 0.2 * (max_pos - min_pos)
        x_start = min_pos - padding
        x_end = max_pos + padding
        
        x_vals = torch.linspace(x_start, x_end, 1000)
        
        # Compute average reward across all observation functions
        average_rewards = torch.zeros_like(x_vals)
        for i in range(self.n_observations):
            obs_i = self.observations[i](x_vals)
            reward_i = self.reward_function(obs_i, **self.reward_function_args[i])
            average_rewards += reward_i
        average_rewards /= self.n_observations
        
        # Update the line plot
        self.line.set_data(x_vals.numpy(), average_rewards.numpy())
        
        # Update scatter positions with current rewards
        if rewards is not None:
            y_positions = rewards
        else:
            y_positions = torch.zeros_like(self.positions)
        
        self.scatter.set_offsets(
            torch.stack([self.positions, y_positions], dim=1).numpy()
        )
        
        # Adjust axis limits
        self.ax.set_xlim(x_start, x_end)
        y_min = average_rewards.min().item()
        y_max = average_rewards.max().item()
        padding_y = 0.1 * (y_max - y_min)
        self.ax.set_ylim(y_min - padding_y, y_max + padding_y)
        
        self.fig.canvas.draw_idle()
        plt.pause(0.0001)


    def reset(self):

        self.positions = torch.normal(mean=self.starting_position_mean, std=self.starting_position_var, size=(self.n_actors,))
        self.time_elapsed = 0
        self.done = False

        return torch.zeros((self.n_actors, self.n_observations))
