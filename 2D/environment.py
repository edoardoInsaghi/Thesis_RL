import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math

class EnvArgs2D:
    def __init__(self, 
                 n_actors: int = 5,
                 velocity: float = 0.01,
                 angular_velocity: float = 45,  # degrees per action
                 movement_noise: float = 0.001,
                 max_steps: int = 1000,
                 starting_position_mean: tuple = (0, 0),
                 starting_position_var: float = 10,
                 cosines: int = 10,
                 circle_start: bool = False,
                 circle_radius: float = 25.0,
                 num_actions: int = 9):  # 8 directions + stand still
        
        self.n_actors = n_actors
        self.starting_position_mean = starting_position_mean
        self.starting_position_var = starting_position_var
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.movement_noise = movement_noise
        self.max_steps = max_steps
        self.cosines = cosines
        self.circle_start = circle_start
        self.circle_radius = circle_radius
        self.num_actions = num_actions


def reward_function_2d(amplitudes, freqs_x, freqs_y, phases, 
                       landscape_center, landscape_width, positions):
    
    x = positions[:, 0].unsqueeze(1)
    y = positions[:, 1].unsqueeze(1)
    
    # Calculate cosine terms
    dot_products = freqs_x * x + freqs_y * y
    cos_terms = amplitudes * torch.cos(dot_products + phases)
    
    reward_sum = torch.sum(cos_terms, dim=1)
    
    # Gaussian envelope
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
        
        self.action_vectors = self._create_action_vectors(
            args.num_actions, 
            args.velocity,
            args.angular_velocity
        )
        
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
        self.landscape_changed = True 
        self.connection_lines = [] 


    def _create_action_vectors(self, num_actions, velocity, angular_velocity):
        vectors = []
        for i in range(num_actions - 1):
            angle_deg = i * angular_velocity
            angle_rad = np.deg2rad(angle_deg)
            x = velocity * np.cos(angle_rad)
            y = velocity * np.sin(angle_rad)
            vectors.append([x, y])
        
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

        if self.args.circle_start:
            angles = torch.linspace(0, 2 * math.pi, self.n_actors + 1)[:-1]  # even distribution
            self.positions = torch.stack([
                torch.cos(angles) * self.args.circle_radius,
                torch.sin(angles) * self.args.circle_radius
            ], dim=1)
        else:
            mean_x, mean_y = self.args.starting_position_mean
            self.positions = torch.normal(
                mean=mean_x, 
                std=self.args.starting_position_var,
                size=(self.n_actors, 2)
            )

        angles = torch.rand(self.n_actors) * 2 * math.pi
        self.directions = torch.stack([
            torch.cos(angles),
            torch.sin(angles)
        ], dim=1)
        
        self.time_elapsed = 0
        self.done = False
        self._init_landscape()
        self.landscape_changed = True # Forces to recumpute landscape, otherwise visualisation breaks
        
        self.connection_lines = []
        
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

    def render(self, rewards: torch.Tensor, edge_index: torch.Tensor = None):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            self.ax.set_title("2D Environment with Agent Connections")
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
                headaxislength=5,
                zorder=10  # Draw agents on top
            )
            
            # Add rewards text
            self.reward_texts = []
            for i in range(self.n_actors):
                text = self.ax.text(
                    self.positions[i, 0].item() + 0.5,
                    self.positions[i, 1].item() + 0.5,
                    f"{rewards[i]:.2f}",
                    color='white' if rewards[i] < 0.5 else 'black',
                    fontsize=8,
                    bbox=dict(facecolor=self.agent_colors[i], alpha=0.7, boxstyle='round,pad=0.2'),
                    zorder=15  # Draw text on top of everything
                )
                self.reward_texts.append(text)
            
            plt.tight_layout()
            plt.show(block=False)
            self.landscape_changed = True

            for line in self.connection_lines:
                line.remove()
            self.connection_lines = []
        
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

            for line in self.connection_lines:
                line.remove()
            self.connection_lines = []
        
        # Update agent connections if provided
        if edge_index is not None:
            # Clear previous connection lines
            for line in self.connection_lines:
                line.remove()
            self.connection_lines = []
            
            # Draw new connections
            for i in range(edge_index.size(1)):
                src_idx = edge_index[0, i].item()
                dst_idx = edge_index[1, i].item()
                
                src_pos = self.positions[src_idx].numpy()
                dst_pos = self.positions[dst_idx].numpy()
                
                # Draw connection line
                line = self.ax.plot(
                    [src_pos[0], dst_pos[0]],
                    [src_pos[1], dst_pos[1]],
                    ':',
                    color='black',
                    alpha=0.8,
                    linewidth=1.0,
                    zorder=5  # Draw behind agents
                )[0]
                self.connection_lines.append(line)
        
        # Update agent positions and directions
        self.quiver.set_offsets(self.positions.numpy())
        self.quiver.set_UVC(
            self.directions[:, 0].numpy(), 
            self.directions[:, 1].numpy()
        )
        
        # Update reward text
        for i, text in enumerate(self.reward_texts):
            text.set_position((self.positions[i, 0].item() + 0.5, 
                              self.positions[i, 1].item() + 0.5))
            text.set_text(f"{rewards[i]:.2f}")
            text.set_color('white' if rewards[i] < 0.5 else 'black')
            # Update background color based on current agent color
            text.get_bbox_patch().set_facecolor(self.agent_colors[i])
        
        # Update plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)