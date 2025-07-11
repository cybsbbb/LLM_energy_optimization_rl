"""
Evaluation script for LLM Data Center RL Agent.
This script evaluates trained RL agents and baseline agents, including
action selection plotting over time with energy price visualization.
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import glob

from rl_environments.llm_datacenter_env import LLMDataCenterEnv
from rl_agents.llm_datacenter_agent import LLMDataCenterAgent
from rl_agents.baseline_agents import get_baseline_agent


class ActionTimeLogger:
    """Logger to track actions and energy prices over time during evaluation."""

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

    def log_step(self, time_ms: int, action: int, energy_price: float,
                 processing_ratio: float, waiting_ratio: float, reward: float):
        """Log a single step."""
        self.times.append(time_ms / 1000 / 3600)  # Convert to hours
        self.actions.append(action)
        self.energy_prices.append(energy_price)
        self.processing_ratios.append(processing_ratio)
        self.waiting_ratios.append(waiting_ratio)
        self.rewards.append(reward)

    def get_data(self):
        """Get all logged data."""
        return {
            'times': np.array(self.times),
            'actions': np.array(self.actions),
            'energy_prices': np.array(self.energy_prices),
            'processing_ratios': np.array(self.processing_ratios),
            'waiting_ratios': np.array(self.waiting_ratios),
            'rewards': np.array(self.rewards)
        }


def evaluate_agent_with_logging(agent, env_kwargs: Dict = None, render: bool = False,
                               enable_deny: bool = True) -> Tuple[Dict, ActionTimeLogger]:
    """Evaluate an agent and log actions over time."""
    if env_kwargs is None:
        env_kwargs = {}

    # Create environment
    env = LLMDataCenterEnv(render_mode="human" if render else None, **env_kwargs)
    logger = ActionTimeLogger()

    obs, _ = env.reset()
    done = False
    episode_reward = 0

    # Action names for reference
    action_names = {
        0: 'fullkv',
        1: 'snapkv_64',
        2: 'snapkv_128',
        3: 'snapkv_256',
        4: 'snapkv_512',
        5: 'snapkv_1024'
    }
    if enable_deny:
        action_names[6] = 'deny'

    while not done:
        # Get action from agent
        if hasattr(agent, 'predict'):
            action = agent.predict(obs, deterministic=True)
        else:
            # For RL agents from stable-baselines3
            action, _ = agent.predict(obs, deterministic=True)

        # Ensure action is an integer
        if isinstance(action, np.ndarray):
            action = int(action.item())
        elif isinstance(action, tuple):
            action = int(action[0].item())
        else:
            action = int(action)

        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated

        # Log this step
        energy_price = env._get_cur_energy_price(env.current_time)
        processing_ratio = env.processing_server_num / env.server_num
        waiting_ratio = len(env.waiting_queue) / env.server_num

        logger.log_step(
            time_ms=env.current_time,
            action=action,
            energy_price=energy_price,
            processing_ratio=processing_ratio,
            waiting_ratio=waiting_ratio,
            reward=reward
        )

        if render:
            env.render()

    # Get final summary
    summary = env.get_final_summary()
    summary['episode_reward'] = episode_reward

    env.close()
    return summary, logger


def plot_action_timeline(loggers: Dict[str, ActionTimeLogger],
                        save_path: str = "../result/action_timeline.png",
                        enable_deny: bool = True):
    """Plot action selection over time with energy price for multiple agents."""

    # Action names and colors
    action_names = {
        0: 'FullKV',
        1: 'SnapKV-64',
        2: 'SnapKV-128',
        3: 'SnapKV-256',
        4: 'SnapKV-512',
        5: 'SnapKV-1024'
    }
    if enable_deny:
        action_names[6] = 'Deny'

    # Colors for each action (using a colormap)
    colors = plt.cm.viridis(np.linspace(0, 1, len(action_names)))
    action_colors = {action: colors[i] for i, action in enumerate(action_names.keys())}

    n_agents = len(loggers)
    fig, axes = plt.subplots(n_agents + 1, 1, figsize=(15, 4 * (n_agents + 1)), sharex=True)

    if n_agents == 1:
        axes = [axes]

    # Get reference times and energy prices from first agent
    first_logger = list(loggers.values())[0]
    ref_data = first_logger.get_data()

    # Plot energy price on top subplot
    ax_energy = axes[0]
    ax_energy.plot(ref_data['times'], ref_data['energy_prices'], 'r-', linewidth=2, label='Energy Price')
    ax_energy.set_ylabel('Energy Price ($/MWh)', fontsize=12)
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

            # Draw rectangle for this time period
            rect = Rectangle(
                (start_time, -0.4),
                end_time - start_time,
                0.8,
                facecolor=action_colors[action],
                alpha=0.8,
                edgecolor='none'
            )
            ax.add_patch(rect)

        # Set up the plot
        ax.set_ylabel(f'{agent_name}\nAction', fontsize=12)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.grid(True, alpha=0.3, axis='x')

        # Add action legend for the last subplot
        if i == len(loggers) - 1:
            legend_elements = [Rectangle((0, 0), 1, 1, facecolor=action_colors[action],
                                       label=action_names[action])
                             for action in action_names.keys()]
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Set x-axis label and formatting
    axes[-1].set_xlabel('Time (hours)', fontsize=12)

    # Format x-axis to show hours nicely
    for ax in axes:
        ax.tick_params(axis='x', labelsize=10)

    plt.suptitle('Action Selection Timeline with Energy Price', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Action timeline plot saved to: {save_path}")


def plot_system_metrics(loggers: Dict[str, ActionTimeLogger],
                       save_path: str = "../result/system_metrics.png"):
    """Plot system metrics over time for multiple agents."""

    n_agents = len(loggers)
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))

    for i, (agent_name, logger) in enumerate(loggers.items()):
        data = logger.get_data()
        color = colors[i]

        # Processing ratio
        axes[0].plot(data['times'], data['processing_ratios'],
                    label=agent_name, color=color, linewidth=2)

        # Waiting ratio
        axes[1].plot(data['times'], data['waiting_ratios'],
                    label=agent_name, color=color, linewidth=2)

        # Cumulative reward
        cumulative_rewards = np.cumsum(data['rewards'])
        axes[2].plot(data['times'], cumulative_rewards,
                    label=agent_name, color=color, linewidth=2)

    # Formatting
    axes[0].set_ylabel('Processing Ratio', fontsize=12)
    axes[0].set_title('Processing Server Utilization', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    axes[1].set_ylabel('Waiting Ratio', fontsize=12)
    axes[1].set_title('Queue Load (Waiting/Total Servers)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].set_ylabel('Cumulative Reward', fontsize=12)
    axes[2].set_xlabel('Time (hours)', fontsize=12)
    axes[2].set_title('Cumulative Reward Over Time', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.suptitle('System Performance Metrics Over Time', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"System metrics plot saved to: {save_path}")


def evaluate_multiple_agents(agents: Dict, env_kwargs: Dict = None,
                           save_plots: bool = True, plot_dir: str = "../result/") -> Dict:
    """Evaluate multiple agents and generate comparison plots."""

    if env_kwargs is None:
        env_kwargs = {}

    print("\n" + "=" * 80)
    print("EVALUATING AGENTS")
    print("=" * 80)

    results = {}
    loggers = {}

    for name, agent in agents.items():
        print(f"\nEvaluating {name}...")
        summary, logger = evaluate_agent_with_logging(
            agent, env_kwargs,
            enable_deny=env_kwargs.get('enable_deny', True)
        )
        results[name] = summary
        loggers[name] = logger
        print_agent_results(name, summary)

    # Generate plots
    if save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Action timeline plot
        action_plot_path = os.path.join(plot_dir, f"action_timeline_{timestamp}.png")
        plot_action_timeline(loggers, action_plot_path,
                           enable_deny=env_kwargs.get('enable_deny', True))

        # # System metrics plot (Disable for now)
        # metrics_plot_path = os.path.join(plot_dir, f"system_metrics_{timestamp}.png")
        # plot_system_metrics(loggers, metrics_plot_path)

        # Comparison plot
        comparison_plot_path = os.path.join(plot_dir, f"agent_comparison_{timestamp}.png")
        plot_comparison(results, comparison_plot_path)

    return results


def print_agent_results(agent_name: str, summary: Dict):
    """Print evaluation results for a single agent."""
    print(f"\nResults for {agent_name}:")
    print(f"  Success Rate: {summary.get('Success Rate', 0):.3f}")
    print(f"  Denial Rate: {summary.get('Denial Rate', 0):.3f}")
    print(f"  Average Score: {summary.get('Average Score', 0):.3f}")
    print(f"  Average Latency: {summary.get('Average Latency (s)', 0):.3f}s")
    print(f"  Total Energy Cost: ${summary.get('Total Energy Cost', 0):.4f}")
    print(f"  Total Profit: ${summary.get('Total Profit', 0):.4f}")
    print(f"  Episode Reward: {summary.get('episode_reward', 0):.3f}")


def plot_comparison(results: Dict, save_path: str = "../result/comparison_plot.png"):
    """Plot comparison of agents across key metrics."""
    metrics = ['Success Rate', 'Average Score', 'Average Latency (s)',
               'Total Energy Cost', 'episode_reward']

    n_agents = len(results)
    n_metrics = len(metrics)

    # Create figure with subplots for metrics and action distribution
    fig = plt.figure(figsize=(20, 10))

    # Metrics plots
    for i, metric in enumerate(metrics):
        ax = plt.subplot(2, n_metrics, i + 1)

        agent_names = list(results.keys())
        x = np.arange(n_agents)

        values = []
        for agent in agent_names:
            values.append(results[agent].get(metric, 0))

        ax.bar(x, values)
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(agent_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

    # Action distribution plot
    ax = plt.subplot(2, 1, 2)

    # Get action names from first agent
    first_agent = list(results.keys())[0]
    if "Action Statistics" in results[first_agent]:
        action_names = list(results[first_agent]["Action Statistics"].keys())

        # Prepare data for stacked bar chart
        action_data = {}
        for action in action_names:
            action_data[action] = []
            for agent in results.keys():
                if "Action Statistics" in results[agent] and action in results[agent]["Action Statistics"]:
                    action_data[action].append(results[agent]["Action Statistics"][action]['percentage'])
                else:
                    action_data[action].append(0)

        # Create stacked bar chart
        x = np.arange(len(results))
        bottom = np.zeros(len(results))

        colors = plt.cm.viridis(np.linspace(0, 1, len(action_names)))

        for i, action in enumerate(action_names):
            ax.bar(x, action_data[action], bottom=bottom, label=action, color=colors[i])
            bottom += action_data[action]

        ax.set_title('Action Distribution (%)')
        ax.set_xlabel('Agent')
        ax.set_ylabel('Percentage')
        ax.set_xticks(x)
        ax.set_xticklabels(list(results.keys()), rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def load_rl_agents_from_directory(model_dir: str, algorithms: List[str],
                                 enable_deny: bool = True) -> Dict:
    """Load RL agents from a directory containing trained models."""
    agents = {}

    for algorithm in algorithms:
        # Look for model files
        pattern = os.path.join(model_dir, algorithm, f"{algorithm}_optimized_*.zip")
        model_files = glob.glob(pattern)

        if model_files:
            # Use the most recent model file
            model_path = max(model_files, key=os.path.getctime)
            print(f"Loading {algorithm} model from: {model_path}")

            try:
                agent = LLMDataCenterAgent(
                    algorithm=algorithm,
                    device='auto'
                )
                # Create dummy environment for loading
                env = LLMDataCenterEnv(enable_deny=enable_deny)
                agent.load(model_path, env)
                agents[f"RL_{algorithm}"] = agent.model
                print(f"Successfully loaded {algorithm} agent")
            except Exception as e:
                print(f"Failed to load {algorithm} agent: {e}")
        else:
            print(f"No {algorithm} model found in {model_dir}")

    return agents


def main():
    parser = argparse.ArgumentParser(description='Evaluate RL agents and baselines for LLM Data Center optimization')

    # Model loading
    parser.add_argument('--load-models', type=str, help='Directory containing trained models')
    parser.add_argument('--load-model', type=str, help='Path to specific model file')
    parser.add_argument('--algorithms', nargs='+', default=['PPO', 'A2C', 'DQN', 'SAC'],
                        choices=['PPO', 'A2C', 'DQN', 'SAC', 'TD3'],
                        help='RL algorithms to evaluate')

    # Environment arguments
    parser.add_argument('--enable-deny', action='store_true', default=False,
                        help='Enable deny action in environment')
    parser.add_argument('--eval-time', type=int, default=86400,
                        help='Evaluation time in seconds (default: 24 hours)')
    parser.add_argument('--server-num', type=int, default=200, help='Number of servers')
    parser.add_argument('--bernoulli-prob', type=float, default=0.12, help='Request arrival probability')
    parser.add_argument('--time-interval', type=int, default=10, help='Time interval in milliseconds')
    parser.add_argument('--max-wait-time', type=int, default=10, help='Max wait time in seconds')
    parser.add_argument('--step-time', type=int, default=10, help='Time window per step in seconds')

    # Baseline agents to include
    parser.add_argument('--baselines', nargs='+',
                        default=['all_fullkv', 'all_snapkv_64', 'rule_based_price', 'random'],
                        help='Baseline agents to evaluate')

    # Output options
    parser.add_argument('--plot-dir', type=str, default='../result/datacenter_evaluation', help='Directory to save plots')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')
    parser.add_argument('--render', action='store_true', help='Render environment during evaluation')

    args = parser.parse_args()

    ### set for evaluation ###
    # args.load_models = "/Users/cybsbbbb/Documents/llm_energy_benchmark/result/datacenter_models/training_caiso"
    # args.load_models = "/Users/cybsbbbb/Documents/llm_energy_benchmark/result/datacenter_models/training_pjm"

    # args.load_model = "/Users/cybsbbbb/Documents/llm_energy_benchmark/result/datacenter_models/training_20250620_111341/DQN/datacenter_DQN_1600000_steps.zip"
    # args.load_model = "/Users/cybsbbbb/Documents/llm_energy_benchmark/result/datacenter_models/training_20250620_111341/DQN/best_model.zip"
    # args.load_model = "/Users/cybsbbbb/Documents/llm_energy_benchmark/result/datacenter_models/training_caiso/PPO2/best_model.zip"
    # args.load_model = "/Users/cybsbbbb/Documents/llm_energy_benchmark/result/datacenter_models/training_caiso/DQN/best_model.zip"
    # args.load_model = "/Users/cybsbbbb/Documents/llm_energy_benchmark/result/datacenter_models/training_pjm/DQN/DQN_final_1000000.zip"
    args.load_models = "../result/optimized_models/20250709_235228"

    # Create output directory
    os.makedirs(args.plot_dir, exist_ok=True)

    # Prepare agents for evaluation
    agents = {}

    # Add baseline agents
    for baseline_name in args.baselines:
        try:
            agents[baseline_name] = get_baseline_agent(baseline_name, enable_deny=args.enable_deny)
            print(f"Added baseline agent: {baseline_name}")
        except ValueError as e:
            print(f"Warning: {e}")

    # Load RL agents
    if args.load_models:
        # Load from directory
        rl_agents = load_rl_agents_from_directory(args.load_models, args.algorithms, args.enable_deny)
        agents.update(rl_agents)
    elif args.load_model:
        # Load single model
        print(f"Loading model from: {args.load_model}")
        # Detect algorithm from path or use first specified
        algorithm = args.algorithms[0]
        for alg in ['PPO', 'A2C', 'DQN', 'SAC', 'TD3']:
            if alg.lower() in args.load_model.lower():
                algorithm = alg
                break

        try:
            agent = LLMDataCenterAgent(algorithm=algorithm, device='auto')
            env = LLMDataCenterEnv(enable_deny=args.enable_deny)
            agent.load(args.load_model, env)
            agents[f"RL_{algorithm}"] = agent.model
            print(f"Successfully loaded {algorithm} agent")
        except Exception as e:
            print(f"Failed to load model: {e}")

    if not agents:
        print("No agents to evaluate. Please specify --load-models, --load-model, or include baseline agents.")
        return

    # Environment settings for evaluation
    eval_env_kwargs = {
        'total_time': args.eval_time * 1000,
        'time_interval': args.time_interval,
        'server_num': args.server_num,
        'bernoulli_prob': args.bernoulli_prob,
        'max_wait_time': args.max_wait_time * 1000,
        'step_time_ms': args.step_time * 1000,
        'enable_deny': args.enable_deny
    }

    print("\n" + "=" * 80)
    print("EVALUATION CONFIGURATION")
    print("=" * 80)
    print(f"Agents to evaluate: {list(agents.keys())}")
    print(f"Evaluation time: {args.eval_time}s ({args.eval_time/3600:.1f} hours)")
    print(f"Environment settings:")
    print(f"  - Servers: {args.server_num}")
    print(f"  - Request rate: {args.bernoulli_prob}")
    print(f"  - Max wait time: {args.max_wait_time}s")
    print(f"  - Step time: {args.step_time}s")
    print(f"  - Enable deny: {args.enable_deny}")
    print(f"Plot directory: {args.plot_dir}")
    print("=" * 80)

    # Evaluate all agents
    results = evaluate_multiple_agents(
        agents,
        eval_env_kwargs,
        save_plots=not args.no_plots,
        plot_dir=args.plot_dir
    )

    # Print final comparison table
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    # Select key metrics for comparison
    metrics = ['Success Rate', 'Denial Rate', 'Average Score', 'Average Latency (s)',
               'Total Energy Cost', 'Total Profit', 'episode_reward']

    # Print header
    header = f"{'Agent':<25}"
    for metric in metrics:
        header += f"{metric:<18}"
    print(header)
    print("-" * len(header))

    # Print data
    for name, summary in results.items():
        row = f"{name:<25}"
        for metric in metrics:
            value = summary.get(metric, 0)
            row += f"{value:<18.3f}"
        print(row)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED")
    if not args.no_plots:
        print(f"Plots saved to: {args.plot_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
