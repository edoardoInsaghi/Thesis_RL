import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import random


def reward_function(amplitudes: torch.Tensor, 
                    freqs: torch.Tensor, 
                    phases: torch.Tensor,
                    landscape_center: torch.Tensor,
                    landscape_width: torch.Tensor,
                    x: torch.Tensor):
    
    cos_terms = amplitudes * torch.cos(freqs * x.unsqueeze(-1) + phases)
    envelope = torch.exp(-0.5 * ((x - landscape_center) / landscape_width)**2)

    return torch.sigmoid(torch.sum(cos_terms, dim=-1)) * envelope


class EnvArgs1D:

    def __init__(self, 
                 n_actors: int = 5, 
                 velocity: float = 0.01, 
                 movement_noise: float = 0.001, 
                 max_steps: int = 1000,
                 starting_position_mean: float = 0, 
                 starting_position_var: float = 10,
                 cosines: int = 10):
        
        self.n_actors = n_actors
        self.starting_position_mean = starting_position_mean
        self.starting_position_var = starting_position_var
        self.velocity = velocity
        self.movement_noise = movement_noise
        self.max_steps = max_steps
        self.cosines = cosines


class Environment1D:

    def __init__(self, args: EnvArgs1D):

        self.args = args
        self.n_actors = args.n_actors
        self.starting_position_mean = args.starting_position_mean
        self.starting_position_var = args.starting_position_var
        self.positions = torch.normal(mean=self.starting_position_mean, std=self.starting_position_var, size=(self.n_actors,))
        self.coeff_velocity = args.velocity
        self.coeff_movement_noise = args.movement_noise
        self.cosines = args.cosines
        self.amplitudes = torch.rand((self.cosines))
        self.freqs = torch.rand((self.cosines)) * 2
        self.phases = torch.rand((self.cosines))
        self.landscape_center = torch.rand((1)) * 20 - 5
        self.landscape_width = torch.rand((1)) * 10 + 10
        self.done = False

        self.time_elapsed = 0

        # Rendering
        plt.ion() 
        self.fig = None
        self.axs = None
        self.scatters = [] 
        self.lines = [] 
        self.agent_colors = cm.tab10([i/args.n_actors for i in range(args.n_actors)]) 



    def step(self, action: torch.Tensor): # (n_actors, n_dim = 1) -> (n_actors, 1), (n_actors, 1)

        self.positions = self.positions + self.coeff_velocity * action + torch.normal(mean=0.0, std=self.coeff_movement_noise, size=(self.n_actors,))
        rewards = reward_function(
            self.amplitudes, 
            self.freqs, 
            self.phases, 
            self.landscape_center, 
            self.landscape_width,
            self.positions
        )
        self.time_elapsed += 1
        if self.time_elapsed >= self.args.max_steps:
            self.done = True

        return self.positions, rewards, self.done
    

    def _init_plot(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
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

    
    def render(self, rewards: torch.Tensor):
        self._init_plot()
        
        min_pos = self.positions.min().item()
        max_pos = self.positions.max().item()
        padding = max(1.0, 0.2 * (max_pos - min_pos))
        x_start = min_pos - padding
        x_end = max_pos + padding
        
        x_vals = torch.linspace(x_start, x_end, 1000)
        
        landscape = reward_function(
            self.amplitudes, 
            self.freqs, 
            self.phases, 
            self.landscape_center, 
            self.landscape_width,
            x_vals
        )
        self.line.set_data(x_vals.numpy(), landscape.numpy())
        
        self.scatter.set_offsets(torch.stack([
            self.positions, 
            rewards
        ], dim=1).numpy())
        
        self.ax.set_xlim(x_start, x_end)
        y_min = landscape.min().item()
        y_max = landscape.max().item()
        padding_y = max(0.1, 0.1 * (y_max - y_min))
        self.ax.set_ylim(y_min - padding_y, y_max + padding_y)
        
        self.fig.canvas.draw_idle()
        plt.pause(0.001)


    def reset(self):

        self.positions = torch.normal(mean=self.starting_position_mean, std=self.starting_position_var, size=(self.n_actors,))
        self.time_elapsed = 0
        self.done = False
        self.amplitudes = torch.rand((self.cosines))
        self.freqs = torch.rand((self.cosines)) * 2
        self.phases = torch.rand((self.cosines))
        self.landscape_center = torch.rand((1)) * 20 - 5
        self.landscape_width = torch.rand((1)) * 10 + 10

        return self.positions
    


if __name__ == "__main__":

    args = EnvArgs1D()
    env = Environment1D(args)
    
    for episode in range(10):
        while True:
            actions = torch.tensor([random.choice([-1, 0, 1]) for _ in range(args.n_actors)])
            positions, rewards, done = env.step(actions)
            env.render(rewards)
            if done:
                positions = env.reset()
                break

    plt.ioff()
    plt.show()

