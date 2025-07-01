import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math

class EnvArgs2D:
    def __init__(self, 
                 n_actors: int = 5,
                 velocity: float = 0.01,
                 angular_velocity: float = 15,  # degrees per action
                 movement_noise: float = 0.001,
                 max_steps: int = 1000,
                 starting_position_mean: tuple = (0, 0),
                 starting_position_var: float = 10,
                 cosines: int = 10,
                 num_actions: int = 24):  # 24 directions + stand still
        
        self.n_actors = n_actors
        self.starting_position_mean = starting_position_mean
        self.starting_position_var = starting_position_var
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.movement_noise = movement_noise
        self.max_steps = max_steps
        self.cosines = cosines
        self.num_actions = num_actions


def reward_function_2d(amplitudes, freqs_x, freqs_y, phases, 
                       landscape_center, landscape_width, positions):
    # Calculate dot product for each position and cosine component
    x = positions[:, 0].unsqueeze(1)
    y = positions[:, 1].unsqueeze(1)
    
    # Calculate cosine terms
    dot_products = freqs_x * x + freqs_y * y
    cos_terms = amplitudes * torch.cos(dot_products + phases)
    
    # Sum over all cosine components
    reward_sum = torch.sum(cos_terms, dim=1)
    
    # Apply Gaussian envelope
    dist_sq = (x.squeeze() - landscape_center[0])**2 + (y.squeeze() - landscape_center[1])**2
    envelope = torch.exp(-0.5 * dist_sq / (landscape_width**2))
    
    return torch.sigmoid(reward_sum) * envelope


class Environment2D:
    def __init__(self, args: EnvArgs2D):
        self.args = args
        self.n_actors = args.n_actors
        self.positions = torch.zeros((self.n_actors, 2))
        self.directions = torch.zeros((self.n_actors, 2))
        self.time_elapsed = 0
        self.done = False
        
        # Create action vectors (directions)
        self.action_vectors = self._create_action_vectors(
            args.num_actions, 
            args.velocity,
            args.angular_velocity
        )
        
        # Initialize landscape parameters
        self._init_landscape()
        
        # Visualization
        plt.ion()
        self.fig = None
        self.ax = None
        self.contour = None
        self.cbar = None
        self.quiver = None
        self.reward_texts = None
        self.agent_colors = cm.tab10(np.linspace(0, 1, self.n_actors))
        self.landscape_changed = True  # Track landscape changes

    def _create_action_vectors(self, num_actions, velocity, angular_velocity):
        vectors = []
        # Create directional vectors (except stand still)
        for i in range(num_actions - 1):
            angle_deg = i * angular_velocity
            angle_rad = np.deg2rad(angle_deg)
            x = velocity * np.cos(angle_rad)
            y = velocity * np.sin(angle_rad)
            vectors.append([x, y])
        
        # Add stand still action (0, 0)
        vectors.append([0.0, 0.0])
        
        return torch.tensor(vectors, dtype=torch.float32)

    def _init_landscape(self):
        self.amplitudes = torch.rand((self.args.cosines))
        self.freqs_x = (torch.rand((self.args.cosines)) - 0.5) * 2
        self.freqs_y = (torch.rand((self.args.cosines)) - 0.5) * 2
        self.phases = torch.rand((self.args.cosines)) * 2 * math.pi
        
        # Landscape center and width
        self.landscape_center = torch.tensor([
            (torch.rand(1) - 0.5) * 20,
            (torch.rand(1) - 0.5) * 20
        ])
        self.landscape_width = torch.rand(1) * 10 + 10

    def reset(self):
        # Reset positions
        mean_x, mean_y = self.args.starting_position_mean
        self.positions = torch.normal(
            mean=mean_x, 
            std=self.args.starting_position_var,
            size=(self.n_actors, 2)
        )
        
        # Reset directions to random orientations
        angles = torch.rand(self.n_actors) * 2 * math.pi
        self.directions = torch.stack([
            torch.cos(angles),
            torch.sin(angles)
        ], dim=1)
        
        # Reset time and landscape
        self.time_elapsed = 0
        self.done = False
        self._init_landscape()
        self.landscape_changed = True  # Flag landscape change
        
        return self.positions.clone()

    def step(self, action_indices: torch.Tensor):
        # Convert action indices to movement vectors
        movements = self.action_vectors[action_indices]
        
        # Update directions (only if moving)
        moving_mask = (action_indices < self.args.num_actions - 1)
        if moving_mask.any():
            self.directions[moving_mask] = movements[moving_mask] / self.args.velocity
        
        # Update positions with noise
        noise = torch.normal(
            mean=0.0, 
            std=self.args.movement_noise, 
            size=(self.n_actors, 2))
        self.positions += movements + noise
        
        # Calculate rewards
        rewards = reward_function_2d(
            self.amplitudes, 
            self.freqs_x, 
            self.freqs_y,
            self.phases,
            self.landscape_center,
            self.landscape_width,
            self.positions
        )
        
        # Update time
        self.time_elapsed += 1
        if self.time_elapsed >= self.args.max_steps:
            self.done = True
        
        return self.positions.clone(), rewards, self.done

    def render(self, rewards: torch.Tensor):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            self.ax.set_title("2D Environment")
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.grid(True)
            
            # Create grid for contour plot
            self.x_grid = torch.linspace(-30, 30, 100)
            self.y_grid = torch.linspace(-30, 30, 100)
            X, Y = torch.meshgrid(self.x_grid, self.y_grid, indexing='xy')
            self.grid_positions = torch.stack([X.flatten(), Y.flatten()], dim=1)
            
            # Create initial contour plot with fixed color range (0-1)
            self.contour = self.ax.contourf(
                X.numpy(), Y.numpy(), 
                np.zeros((100, 100)),  # Start with zeros
                levels=20, 
                cmap='viridis', 
                alpha=0.6,
                vmin=0, 
                vmax=1
            )
            
            # Create quiver plot for agents
            self.quiver = self.ax.quiver(
                self.positions[:, 0].numpy(),
                self.positions[:, 1].numpy(),
                self.directions[:, 0].numpy(),
                self.directions[:, 1].numpy(),
                color=self.agent_colors,
                scale=12,
                scale_units='inches',
                edgecolors='black',
                linewidths=0.5,
                headwidth=6,
                headlength=6,
                headaxislength=5
            )
            
            # Add rewards text
            self.reward_texts = []
            for i in range(self.n_actors):
                text = self.ax.text(
                    self.positions[i, 0].item() + 1,
                    self.positions[i, 1].item() + 1,
                    f"{rewards[i]:.2f}",
                    color='white' if rewards[i] < 0.5 else 'black',
                    fontsize=9,
                    bbox=dict(facecolor=self.agent_colors[i], alpha=0.7, boxstyle='round,pad=0.2')
                )
                self.reward_texts.append(text)
            
            plt.tight_layout()
            plt.show(block=False)
            self.landscape_changed = True  # Force update on first render
        
        # Always update landscape if changed
        if self.landscape_changed:
            # Calculate new landscape values
            Z = reward_function_2d(
                self.amplitudes,
                self.freqs_x,
                self.freqs_y,
                self.phases,
                self.landscape_center,
                self.landscape_width,
                self.grid_positions
            ).reshape(100, 100).numpy()
            
            # Robust contour removal for all Matplotlib versions
            if self.contour:
                try:
                    # Try the standard removal method
                    for coll in self.contour.collections:
                        coll.remove()
                except AttributeError:
                    # Fallback to simple remove() if collections doesn't exist
                    self.contour.remove()
                self.contour = None
            
            # Create new contour with updated values
            self.contour = self.ax.contourf(
                self.x_grid.numpy(), self.y_grid.numpy(), 
                Z, 
                levels=20, 
                cmap='viridis', 
                alpha=0.6,
                vmin=0, 
                vmax=1
            )
            
            self.landscape_changed = False
        
        # Update agent positions and directions
        self.quiver.set_offsets(self.positions.numpy())
        self.quiver.set_UVC(
            self.directions[:, 0].numpy(), 
            self.directions[:, 1].numpy()
        )
        
        # Update reward text
        for i, text in enumerate(self.reward_texts):
            text.set_position((self.positions[i, 0].item() + 1, self.positions[i, 1].item() + 1))
            text.set_text(f"{rewards[i]:.2f}")
            text.set_color('white' if rewards[i] < 0.5 else 'black')
        
        # Update plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)