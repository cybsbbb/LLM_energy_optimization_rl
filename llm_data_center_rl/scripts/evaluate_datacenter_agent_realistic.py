#!/usr/bin/env python3
"""
Clean evaluation script for optimized realistic LLM Data Center RL Agent.
Uses the optimized environment with binary search for maximum performance.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import glob

from rl_environments.llm_datacenter_realistic_env import RealisticLLMDataCenterEnv
from rl_agents.llm_datacenter_agent import LLMDataCenterAgent, DiscreteToBoxWrapper, SafeMonitor
from rl_agents.baseline_agents import get_baseline_agent


class RealisticActionLogger:
    """Logger to track actions and performance over real time."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.real_times = []
        self.actions = []
        self.energy_prices = []
        self.request_rates = []
        self.processing_ratios = []
        self.waiting_ratios = []
        self.rewards = []
        self.cumulative_rewards = []

    def log_step(self, real_time: datetime, action: int, energy_price: float,
                 request_rate: float, processing_ratio: float, waiting_ratio: float,
                 reward: float, cumulative_reward: float):
        self.real_times.append(real_time)
        self.actions.append(action)
        self.energy_prices.append(energy_price)
        self.request_rates.append(request_rate)
        self.processing_ratios.append(processing_ratio)
        self.waiting_ratios.append(waiting_ratio)
        self.rewards.append(reward)
        self.cumulative_rewards.append(cumulative_reward)

    def get_data(self):
        return {
            'real_times': self.real_times,
            'actions': np.array(self.actions),
            'energy_prices': np.array(self.energy_prices),
            'request_rates': np.array(self.request_rates),
            'processing_ratios': np.array(self.processing_ratios),
            'waiting_ratios': np.array(self.waiting_ratios),
            'rewards': np.array(self.rewards),
            'cumulative_rewards': np.array(self.cumulative_rewards)
        }


def evaluate_agent(agent, invocations_file: str, energy_price_file: str,
                   simulation_start_time: str, simulation_duration_hours: int = 24,
                   enable_deny: bool = True) -> Tuple[Dict, RealisticActionLogger]:
    """Evaluate an agent on optimized realistic data."""

    # Create optimized environment
    if hasattr(agent, 'algorithm') and agent.algorithm in ['SAC', 'TD3']:
        eval_base_env = RealisticLLMDataCenterEnv(
            invocations_file=invocations_file,
            energy_price_file=energy_price_file,
            simulation_start_time=simulation_start_time,
            simulation_duration_hours=simulation_duration_hours,
            time_acceleration_factor=1.0,
            request_scaling_factor=1.0,
            enable_deny=enable_deny
        )
        eval_wrapped_env = DiscreteToBoxWrapper(eval_base_env)
        env = SafeMonitor(eval_wrapped_env)
    else:
        env = RealisticLLMDataCenterEnv(
            invocations_file=invocations_file,
            energy_price_file=energy_price_file,
            simulation_start_time=simulation_start_time,
            simulation_duration_hours=simulation_duration_hours,
            time_acceleration_factor=1.0,
            request_scaling_factor=1.0,
            enable_deny=enable_deny
        )

    logger = RealisticActionLogger()

    obs, _ = env.reset()
    done = False
    episode_reward = 0
    cumulative_reward = 0
    step_count = 0

    while not done:
        # Get action from agent
        if hasattr(agent, 'predict'):
            action = agent.predict(obs, deterministic=True)
        else:
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
        cumulative_reward += reward
        done = terminated or truncated

        # Get environment state for logging
        base_env = env.env if hasattr(env, 'env') else env
        if hasattr(base_env, 'env'):
            base_env = base_env.env

        real_time = base_env.current_real_time
        energy_price = base_env.get_current_energy_price(real_time)
        request_rate = base_env.get_current_request_rate(real_time)
        processing_ratio = base_env.processing_server_num / base_env.server_num
        waiting_ratio = len(base_env.waiting_queue) / base_env.server_num

        # Log this step
        logger.log_step(
            real_time=real_time,
            action=action,
            energy_price=energy_price,
            request_rate=request_rate,
            processing_ratio=processing_ratio,
            waiting_ratio=waiting_ratio,
            reward=reward,
            cumulative_reward=cumulative_reward
        )

        step_count += 1

    # Get final summary
    summary = base_env.get_final_summary()
    summary['episode_reward'] = episode_reward
    summary['total_steps'] = step_count

    # Get performance stats if available
    if hasattr(base_env, 'get_performance_stats'):
        perf_stats = base_env.get_performance_stats()
        summary['performance_stats'] = perf_stats

    env.close()
    return summary, logger


def plot_action_timeline(loggers: Dict[str, RealisticActionLogger],
                         save_path: str = "../result/optimized_timeline.png",
                         enable_deny: bool = False):
    """Plot action selection over real time with energy prices and request rates."""

    action_names = {
        0: 'FullKV', 1: 'SnapKV-64', 2: 'SnapKV-128',
        3: 'SnapKV-256', 4: 'SnapKV-512', 5: 'SnapKV-1024'
    }
    if enable_deny:
        action_names[6] = 'Deny'

    colors = plt.cm.viridis(np.linspace(0, 1, len(action_names)))
    # action_colors = {action: colors[i] for i, action in enumerate(action_names.keys())}
    action_colors = {0:colors[0], 1:colors[5], 2:colors[4], 3:colors[3], 4:colors[2], 5:colors[1]}

    n_agents = len(loggers)
    fig, axes = plt.subplots(n_agents + 2, 1, figsize=(20, 5 * (n_agents + 2)), sharex=True)

    if n_agents == 0:
        return

    # Get reference data from first agent
    first_logger = list(loggers.values())[0]
    ref_data = first_logger.get_data()
    time_data = [pd.to_datetime(t) for t in ref_data['real_times']]

    # Plot energy price
    ax_energy = axes[0]
    ax_energy.plot(time_data, ref_data['energy_prices'], 'r-', linewidth=2, label='Energy Price')
    ax_energy.set_ylabel('Energy Price\n($/MWh)', fontsize=12)
    ax_energy.set_title('Energy Price Over Time', fontsize=14, fontweight='bold')
    ax_energy.grid(True, alpha=0.3)
    ax_energy.legend()

    # Plot request rate
    ax_requests = axes[1]
    ax_requests.plot(time_data, ref_data['request_rates'], 'b-', linewidth=2, label='Request Rate')
    ax_requests.set_ylabel('Request Rate\n(req/10s)', fontsize=12)
    ax_requests.set_title('Request Rate Over Time', fontsize=14, fontweight='bold')
    ax_requests.grid(True, alpha=0.3)
    ax_requests.legend()

    # Plot actions for each agent
    for i, (agent_name, logger) in enumerate(loggers.items()):
        ax = axes[i + 2]
        data = logger.get_data()
        agent_times = [pd.to_datetime(t) for t in data['real_times']]

        # Create action timeline as colored segments
        for j in range(len(agent_times) - 1):
            start_time = agent_times[j]
            end_time = agent_times[j + 1]
            action = data['actions'][j]

            duration = (end_time - start_time).total_seconds() / 86400
            start_mdates = mdates.date2num(start_time)

            rect = Rectangle(
                (start_mdates, -0.4), duration, 0.8,
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

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Timeline plot saved to: {save_path}")


def plot_performance_comparison(results: Dict, save_path: str = "../result/optimized_comparison.png"):
    """Plot comparison of agents across key metrics."""

    if not results:
        return

    # metrics = ['Success Rate', 'Denial Rate', 'Average Score', 'episode_reward']
    metrics = ['Success Rate', 'Average Latency (s)', 'Average Score', 'episode_reward', 'Total Energy Cost', 'Total Profit']

    n_agents = len(results)

    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        agent_names = list(results.keys())
        values = [results[agent].get(metric, 0) for agent in agent_names]

        bars = ax.bar(agent_names, values, alpha=0.7)
        ax.set_title(f'{metric}')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        if metric == 'episode_reward':
            ax.set_ylim(bottom=45000)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def load_optimized_agents(model_dir: str, algorithms: List[str], enable_deny: bool = True) -> Dict:
    """Load trained agents from optimized models."""
    agents = {}

    for algorithm in algorithms:
        # Look for optimized model files
        patterns = [
            os.path.join(model_dir, algorithm, f"{algorithm}_optimized_realistic_*.zip"),
            os.path.join(model_dir, algorithm, f"{algorithm}_realistic_*.zip"),
            os.path.join(model_dir, algorithm, f"{algorithm}_final_*.zip")
        ]

        model_files = []
        for pattern in patterns:
            model_files.extend(glob.glob(pattern))

        if model_files:
            model_path = max(model_files, key=os.path.getctime)
            print(f"Loading {algorithm} model from: {model_path}")

            try:
                agent = LLMDataCenterAgent(algorithm=algorithm, device='auto')

                # Create dummy environment for loading
                if algorithm in ['SAC', 'TD3']:
                    dummy_env = RealisticLLMDataCenterEnv(
                        invocations_file="../data/llm_inference_trace/processed/2024/invocations_ten_second_Coding.csv",
                        energy_price_file="../data/energy_price/processed/CAISO_20240509_20240521.csv",
                        simulation_duration_hours=1,
                        enable_deny=enable_deny
                    )
                    wrapped_env = DiscreteToBoxWrapper(dummy_env)
                    env = SafeMonitor(wrapped_env)
                else:
                    env = RealisticLLMDataCenterEnv(
                        invocations_file="../data/llm_inference_trace/processed/2024/invocations_ten_second_Coding.csv",
                        energy_price_file="../data/energy_price/processed/CAISO_20240509_20240521.csv",
                        simulation_duration_hours=1,
                        enable_deny=enable_deny
                    )

                agent.load(model_path, env)
                agents[f"RL_{algorithm}"] = agent.model
                print(f"✓ Successfully loaded {algorithm} agent")
                env.close()
            except Exception as e:
                print(f"✗ Failed to load {algorithm} agent: {e}")
        else:
            print(f"✗ No {algorithm} model found in {model_dir}")

    return agents


def print_agent_results(agent_name: str, summary: Dict):
    """Print evaluation results for a single agent."""
    print(f"\n📊 Results for {agent_name}:")
    print(f"  Success Rate: {summary.get('Success Rate', 0):.3f}")
    print(f"  Denial Rate: {summary.get('Denial Rate', 0):.3f}")
    print(f"  Average Score: {summary.get('Average Score', 0):.3f}")
    print(f"  Average Latency: {summary.get('Average Latency (s)', 0):.3f}s")
    print(f"  Total Energy Cost: ${summary.get('Total Energy Cost', 0):.4f}")
    print(f"  Episode Reward: {summary.get('episode_reward', 0):.3f}")
    print(f"  Total Steps: {summary.get('total_steps', 0)}")

    # Print performance stats if available
    if 'performance_stats' in summary:
        perf = summary['performance_stats']
        print(f"  Cache Efficiency: {perf['cache_hit_ratio_estimate']:.1%}")
        print(f"  Cache Sizes: Energy={perf['energy_cache_size']}, Invocations={perf['invocation_cache_size']}")


def evaluate_multiple_agents(agents: Dict, invocations_file: str, energy_price_file: str,
                             evaluation_periods: List[Dict], save_plots: bool = True,
                             plot_dir: str = "../result/") -> Dict:
    """Evaluate multiple agents across different time periods."""

    print("\n" + "=" * 80)
    print("🚀 EVALUATING AGENTS WITH OPTIMIZED ENVIRONMENT")
    print("=" * 80)

    all_results = {}

    for period_info in evaluation_periods:
        period_name = period_info['name']
        start_time = period_info['start_time']
        duration = period_info['duration_hours']

        print(f"\n--- 📅 Evaluation Period: {period_name} ---")
        print(f"Start time: {start_time}")
        print(f"Duration: {duration} hours")

        period_results = {}
        period_loggers = {}
        period_start_time = time.time()

        for name, agent in agents.items():
            print(f"\n🔍 Evaluating {name}...")
            eval_start_time = time.time()

            try:
                summary, logger = evaluate_agent(
                    agent, invocations_file, energy_price_file,
                    start_time, duration, enable_deny=True
                )
                period_results[name] = summary
                period_loggers[name] = logger

                eval_time = time.time() - eval_start_time
                print(f"✓ Completed in {eval_time:.1f}s")
                print_agent_results(name, summary)

            except Exception as e:
                print(f"✗ Failed to evaluate {name}: {e}")
                continue

        period_time = time.time() - period_start_time
        print(f"\n⏱️  Period evaluation completed in {period_time:.1f}s")

        all_results[period_name] = {
            'results': period_results,
            'loggers': period_loggers,
            'period_info': period_info,
            'evaluation_time': period_time
        }

        # Generate plots for this period
        if save_plots and period_loggers:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Timeline plot
            timeline_path = os.path.join(plot_dir, f"optimized_timeline_{period_name}_{timestamp}.png")
            plot_action_timeline(period_loggers, timeline_path, enable_deny=False)

            # Comparison plot
            comparison_path = os.path.join(plot_dir, f"optimized_comparison_{period_name}_{timestamp}.png")
            plot_performance_comparison(period_results, comparison_path)

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate agents using optimized realistic environment')

    # Data files
    parser.add_argument('--invocations-file', type=str, default='../data/LLM_inference_trace/processed/2024/invocations_ten_second_Coding.csv',
                        help='Path to invocations CSV file')
    parser.add_argument('--energy-price-file', type=str, default='../data/energy_price/processed/CAISO_20240509_20240521.csv',
                        help='Path to energy price CSV file')
    # Model loading
    parser.add_argument('--load-models', type=str, help='Directory containing trained models')
    parser.add_argument('--load-model', type=str, help='Path to specific model file')
    parser.add_argument('--algorithms', nargs='+', default=['PPO', 'DQN', 'A2C'],
                        choices=['PPO', 'A2C', 'DQN', 'SAC', 'TD3'],
                        help='RL algorithms to evaluate')
    # Baseline agents
    parser.add_argument('--baselines', nargs='+',
                        default=['all_fullkv', 'all_snapkv_64', 'rule_based_price', 'random'],
                        help='Baseline agents to evaluate')
    # Environment settings
    parser.add_argument('--enable-deny', action='store_true', default=False,
                        help='Enable deny action in environment')
    # Output options
    parser.add_argument('--plot-dir', type=str, default='../result/', help='Directory to save plots')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')

    args = parser.parse_args()

    args.load_models = "../result/optimized_models/20250711_005407"

    # Create output directory
    os.makedirs(args.plot_dir, exist_ok=True)

    print("=" * 80)
    print("🚀 OPTIMIZED REALISTIC LLM DATA CENTER EVALUATION")
    print(f"📁 Data files:")
    print(f"  - Invocations: {args.invocations_file}")
    print(f"  - Energy prices: {args.energy_price_file}")
    print("=" * 80)

    # Verify data files exist
    if not os.path.exists(args.invocations_file):
        print(f"❌ Error: Invocations file not found: {args.invocations_file}")
        return

    if not os.path.exists(args.energy_price_file):
        print(f"❌ Error: Energy price file not found: {args.energy_price_file}")
        return

    # Define evaluation periods
    evaluation_periods = []
    evaluation_periods.append({
        'name': 'All week',
        'start_time': '2024-05-10 00:00:00',
        'duration_hours': 24 * 7
    })

    # Prepare agents for evaluation
    agents = {}

    # Add baseline agents
    for baseline_name in args.baselines:
        try:
            agents[baseline_name] = get_baseline_agent(baseline_name, enable_deny=args.enable_deny)
            print(f"✓ Added baseline agent: {baseline_name}")
        except ValueError as e:
            print(f"⚠️  Warning: {e}")

    # Load RL agents
    if args.load_models:
        rl_agents = load_optimized_agents(args.load_models, args.algorithms, args.enable_deny)
        agents.update(rl_agents)
    elif args.load_model:
        print(f"Loading model from: {args.load_model}")
        algorithm = args.algorithms[0]
        for alg in ['PPO', 'A2C', 'DQN', 'SAC', 'TD3']:
            if alg.lower() in args.load_model.lower():
                algorithm = alg
                break
        try:
            agent = LLMDataCenterAgent(algorithm=algorithm, device='auto')
            if algorithm in ['SAC', 'TD3']:
                dummy_env = RealisticLLMDataCenterEnv(
                    invocations_file=args.invocations_file,
                    energy_price_file=args.energy_price_file,
                    simulation_duration_hours=1,
                    enable_deny=args.enable_deny
                )
                wrapped_env = DiscreteToBoxWrapper(dummy_env)
                env = SafeMonitor(wrapped_env)
            else:
                env = RealisticLLMDataCenterEnv(
                    invocations_file=args.invocations_file,
                    energy_price_file=args.energy_price_file,
                    simulation_duration_hours=1,
                    enable_deny=args.enable_deny
                )
            agent.load(args.load_model, env)
            agents[f"RL_{algorithm}"] = agent.model
            print(f"✓ Successfully loaded {algorithm} agent")
            env.close()
        except Exception as e:
            print(f"✗ Failed to load model: {e}")

    if not agents:
        print("❌ No agents to evaluate. Please specify --load-models, --load-model, or include baseline agents.")
        return

    print(f"\n🎯 Agents to evaluate: {list(agents.keys())}")
    print(f"📅 Evaluation periods: {[p['name'] for p in evaluation_periods]}")

    # Evaluate all agents
    try:
        total_start_time = time.time()
        all_results = evaluate_multiple_agents(
            agents,
            args.invocations_file,
            args.energy_price_file,
            evaluation_periods,
            save_plots=not args.no_plots,
            plot_dir=args.plot_dir
        )
        total_eval_time = time.time() - total_start_time

        # Print final summary
        print("\n" + "=" * 80)
        print("📊 FINAL COMPARISON ACROSS TIME PERIODS")
        print("=" * 80)

        for period_name, period_data in all_results.items():
            print(f"\n📅 {period_name.upper()}:")
            print("-" * 50)

            results = period_data['results']
            if not results:
                print("No results for this period.")
                continue

            # Print comparison table
            metrics = ['Success Rate', 'Denial Rate', 'Average Score', 'episode_reward']

            header = f"{'Agent':<20}"
            for metric in metrics:
                header += f"{metric:<15}"
            print(header)
            print("-" * len(header))

            for name, summary in results.items():
                row = f"{name:<20}"
                for metric in metrics:
                    value = summary.get(metric, 0)
                    row += f"{value:<15.3f}"
                print(row)

            print(f"\nEvaluation time: {period_data['evaluation_time']:.1f}s")

        print("\n" + "=" * 80)
        print("✅ OPTIMIZED EVALUATION COMPLETED")
        print("=" * 80)
        print(f"⏱️  Total evaluation time: {total_eval_time:.1f}s ({total_eval_time / 60:.1f} minutes)")
        if not args.no_plots:
            print(f"📊 Plots saved to: {args.plot_dir}")
        print("=" * 80)

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()