import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from typing import Dict, List
from datetime import datetime


class ActionTimeLogger:
    """Logger to track actions and environment state over time."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the logger for a new episode."""
        self.times = []
        self.actions = []
        self.energy_prices = []
        self.processing_ratios = []
        self.waiting_ratios = []
        self.rewards = []

    def log_step(self, time: datetime, action: int, energy_price: float,
                 processing_ratio: float, waiting_ratio: float, reward: float):
        """Log a single step."""
        self.times.append(time)
        self.actions.append(action)
        self.energy_prices.append(energy_price)
        self.processing_ratios.append(processing_ratio)
        self.waiting_ratios.append(waiting_ratio)
        self.rewards.append(reward)

    def get_data(self):
        """Get all logged data."""
        return {
            'times': self.times,
            'actions': np.array(self.actions),
            'energy_prices': np.array(self.energy_prices),
            'processing_ratios': np.array(self.processing_ratios),
            'waiting_ratios': np.array(self.waiting_ratios),
            'rewards': np.array(self.rewards)
        }


def plot_action_timeline(loggers: Dict[str, ActionTimeLogger],
                         save_path: str, enable_deny: bool = True):
    """Plot action selection over time with energy price for multiple agents."""

    # Action names and colors
    action_names = {
        0: 'FullKV', 1: 'SnapKV-64', 2: 'SnapKV-128',
        3: 'SnapKV-256', 4: 'SnapKV-512', 5: 'SnapKV-1024'
    }
    if enable_deny:
        action_names[6] = 'Deny'

    colors = plt.cm.viridis(np.linspace(0, 1, len(action_names)))
    action_colors = {action: colors[i] for i, action in enumerate(action_names.keys())}

    n_agents = len(loggers)
    fig, axes = plt.subplots(n_agents + 1, 1, figsize=(15, 4 * (n_agents + 1)), sharex=True)

    if n_agents == 1:
        axes = [axes] if not isinstance(axes, list) else axes

    # Get reference data from first agent
    first_logger = list(loggers.values())[0]
    ref_data = first_logger.get_data()

    # Plot energy price on top subplot
    ax_energy = axes[0]
    ax_energy.plot(ref_data['times'], ref_data['energy_prices'], 'r-', linewidth=2, label='Energy Price')
    ax_energy.set_ylabel('Energy Price\n($/MWh)', fontsize=12)
    ax_energy.set_title('Energy Price Over Time', fontsize=14, fontweight='bold')
    ax_energy.grid(True, alpha=0.3)
    ax_energy.legend()

    # Plot actions for each agent
    for i, (agent_name, logger) in enumerate(loggers.items()):
        ax = axes[i + 1]
        data = logger.get_data()

        # Create action timeline as colored segments
        for j in range(len(data['times']) - 1):
            start_time = data['times'][j]
            end_time = data['times'][j + 1]
            action = data['actions'][j]

            duration = (end_time - start_time).total_seconds() / 3600  # hours
            start_num = mdates.date2num(start_time)

            rect = Rectangle(
                (start_num, -0.4), duration / 24, 0.8,  # Convert hours to days for matplotlib
                facecolor=action_colors[action], alpha=0.8, edgecolor='none'
            )
            ax.add_patch(rect)

        ax.set_ylabel(f'{agent_name}\nAction', fontsize=12)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.grid(True, alpha=0.3, axis='x')

        # Add legend for last subplot
        if i == len(loggers) - 1:
            legend_elements = [Rectangle((0, 0), 1, 1, facecolor=action_colors[action],
                                         label=action_names[action])
                               for action in action_names.keys()]
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Format x-axis
    axes[-1].set_xlabel('Time', fontsize=12)
    fig.autofmt_xdate()

    plt.suptitle('Action Selection Timeline with Energy Price', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Action timeline plot saved to: {save_path}")


def plot_performance_comparison(results: Dict, save_path: str):
    """Plot comparison of agents across key metrics."""
    if not results:
        return

    metrics = ['Success Rate', 'Average Score', 'episode_reward', 'Total Profit']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        agent_names = list(results.keys())
        values = [results[agent].get(metric, 0) for agent in agent_names]

        bars = ax.bar(agent_names, values, alpha=0.7)
        ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Performance comparison plot saved to: {save_path}")


def plot_training_curves(metrics_logger, save_path: str):
    """Plot training curves from metrics logger."""
    df = metrics_logger.get_dataframe()

    if df.empty:
        print("No training data to plot")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Reward
    axes[0, 0].plot(df['episode'], df['reward'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)

    # Success rate
    axes[0, 1].plot(df['episode'], df['success_rate'])
    axes[0, 1].set_title('Success Rate')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].grid(True, alpha=0.3)

    # Denial rate
    axes[0, 2].plot(df['episode'], df['denial_rate'])
    axes[0, 2].set_title('Denial Rate')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Denial Rate')
    axes[0, 2].grid(True, alpha=0.3)

    # Average score
    axes[1, 0].plot(df['episode'], df['avg_score'])
    axes[1, 0].set_title('Average Score')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].grid(True, alpha=0.3)

    # Energy cost
    axes[1, 1].plot(df['episode'], df['energy_cost'])
    axes[1, 1].set_title('Energy Cost')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Energy Cost')
    axes[1, 1].grid(True, alpha=0.3)

    # Profit
    axes[1, 2].plot(df['episode'], df['profit'])
    axes[1, 2].set_title('Profit')
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Profit')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Training curves saved to: {save_path}")
