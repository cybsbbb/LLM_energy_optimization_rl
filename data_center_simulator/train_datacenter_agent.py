import argparse
import os
import sys
from datetime import datetime
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt

from llm_datacenter_env import LLMDataCenterEnv
from llm_datacenter_agent import LLMDataCenterAgent
from baseline_agents import (
    get_baseline_agent, AllFullKVAgent, AllSnapKV64Agent,
    RuleBasedPriceAgent, SmartDenyAgent, AdaptiveLoadAgent
)


def evaluate_agent(agent, env_kwargs: Dict = None, n_episodes: int = 5, render: bool = False):
    """Evaluate any agent (baseline or RL) on the environment."""
    if env_kwargs is None:
        env_kwargs = {}

    all_summaries = []
    all_rewards = []

    for episode in range(n_episodes):
        # Create fresh environment for each episode
        env = LLMDataCenterEnv(render_mode="human" if render else None, **env_kwargs)

        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Get action from agent
            if hasattr(agent, 'predict'):
                action = agent.predict(obs, deterministic=True)
            else:
                # For RL agents from stable-baselines3
                action, _ = agent.predict(obs, deterministic=True)

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

            if render:
                env.render()

        # Get final summary
        summary = env.get_final_summary()
        summary['episode_reward'] = episode_reward
        all_summaries.append(summary)
        all_rewards.append(episode_reward)

        env.close()

    # Calculate average metrics
    avg_summary = {}
    for key in all_summaries[0].keys():
        if key == "Action Statistics":
            # Handle action statistics separately
            avg_summary[key] = {}
            # Get all action names from first episode
            action_names = list(all_summaries[0][key].keys())
            for action_name in action_names:
                counts = [s[key][action_name]['count'] for s in all_summaries]
                percentages = [s[key][action_name]['percentage'] for s in all_summaries]
                avg_summary[key][action_name] = {
                    'count': {'mean': np.mean(counts), 'std': np.std(counts)},
                    'percentage': {'mean': np.mean(percentages), 'std': np.std(percentages)}
                }
        else:
            values = [s[key] for s in all_summaries]
            avg_summary[key] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }

    return avg_summary, all_summaries


def print_evaluation_results(agent_name: str, avg_summary: Dict):
    """Print evaluation results for an agent."""
    print(f"\n{'=' * 60}")
    print(f"Agent: {agent_name}")
    print(f"{'=' * 60}")

    # Separate action statistics from other metrics
    action_stats = None

    for metric, stats in avg_summary.items():
        if metric == "Action Statistics":
            action_stats = stats
        elif isinstance(stats, dict):
            print(f"{metric}: {stats['mean']:.3f} ± {stats['std']:.3f}")
        else:
            print(f"{metric}: {stats:.3f}")

    # Print action statistics if available
    if action_stats:
        print("\nAction Usage:")
        print(f"{'Action':<15} {'Avg Count':<12} {'Percentage':<10}")
        print("-" * 37)

        # Calculate total actions for percentages
        total_actions = sum(stats['count']['mean'] for action_name, stats in action_stats.items()
                            if isinstance(stats, dict) and 'count' in stats)

        for action_name, stats in action_stats.items():
            if isinstance(stats, dict) and 'count' in stats:
                avg_count = stats['count']['mean']
                percentage = (avg_count / total_actions * 100) if total_actions > 0 else 0
                print(f"{action_name:<15} {avg_count:<12.1f} {percentage:<10.1f}%")


def compare_agents(agents: Dict, env_kwargs: Dict = None, n_episodes: int = 5):
    """Compare multiple agents."""
    print("\n" + "=" * 80)
    print("COMPARING AGENTS")
    print("=" * 80)

    results = {}

    for name, agent in agents.items():
        print(f"\nEvaluating {name}...")
        avg_summary, _ = evaluate_agent(agent, env_kwargs, n_episodes)
        results[name] = avg_summary
        print_evaluation_results(name, avg_summary)

    # Print comparison table
    print("\n" + "=" * 80)
    print("FINAL COMPARISON - Performance Metrics")
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
            if metric in summary:
                value = summary[metric]['mean']
                row += f"{value:<18.3f}"
            else:
                row += f"{'N/A':<18}"
        print(row)

    # Print action usage comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON - Action Usage")
    print("=" * 80)

    # Get all action names from first agent
    first_agent_name = list(results.keys())[0]
    if "Action Statistics" in results[first_agent_name]:
        action_names = list(results[first_agent_name]["Action Statistics"].keys())

        # Print header
        header = f"{'Agent':<25}"
        for action in action_names:
            header += f"{action:<15}"
        print(header)
        print("-" * len(header))

        # Print data
        for name, summary in results.items():
            if "Action Statistics" in summary:
                row = f"{name:<25}"
                for action in action_names:
                    if action in summary["Action Statistics"]:
                        percentage = summary["Action Statistics"][action]['percentage']['mean']
                        row += f"{percentage:<15.1f}%"
                    else:
                        row += f"{'0.0%':<15}"
                print(row)

    return results


def plot_comparison(results: Dict, save_path: str = "comparison_plot.png"):
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
        errors = []

        for agent in agent_names:
            if metric in results[agent]:
                values.append(results[agent][metric]['mean'])
                errors.append(results[agent][metric]['std'])
            else:
                values.append(0)
                errors.append(0)

        ax.bar(x, values, yerr=errors, capsize=5)
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
                    action_data[action].append(results[agent]["Action Statistics"][action]['percentage']['mean'])
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
    plt.savefig(save_path)
    plt.show()


def train_rl_agents(args):
    """Train multiple RL agents for comparison."""
    agents = {}

    for algorithm in args.algorithms:
        print("\n" + "=" * 80)
        print(f"TRAINING {algorithm} AGENT")
        print("=" * 80)
        print(f"Total timesteps: {args.timesteps}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Enable deny: {args.enable_deny}")
        print("=" * 80 + "\n")

        # Create agent with algorithm-specific parameters
        agent_params = {
            'algorithm': algorithm,
            'learning_rate': args.learning_rate,
            'gamma': args.gamma,
            'device': args.device
        }

        # Add algorithm-specific parameters
        if algorithm in ['PPO', 'A2C']:
            agent_params.update({
                'n_steps': args.n_steps,
                'n_epochs': args.n_epochs if algorithm == 'PPO' else 5,
                'ent_coef': args.ent_coef,
                'vf_coef': args.vf_coef,
                'max_grad_norm': args.max_grad_norm,
                'gae_lambda': args.gae_lambda,
            })
            if algorithm == 'PPO':
                agent_params['clip_range'] = args.clip_range
                agent_params['batch_size'] = args.batch_size

        elif algorithm in ['DQN', 'SAC', 'TD3']:
            agent_params.update({
                'batch_size': args.batch_size,
                'buffer_size': args.buffer_size,
                'learning_starts': args.learning_starts,
                'tau': args.tau,
                'train_freq': args.train_freq,
                'gradient_steps': args.gradient_steps,
            })
            if algorithm == 'DQN':
                agent_params.update({
                    'target_update_interval': args.target_update_interval,
                    'exploration_fraction': args.exploration_fraction,
                    'exploration_initial_eps': args.exploration_initial_eps,
                    'exploration_final_eps': args.exploration_final_eps,
                })

        agent = LLMDataCenterAgent(**agent_params)

        # Create environment with shorter episode for faster training
        env = agent.create_env(
            total_time=args.episode_time * 1000,  # Convert to milliseconds
            server_num=args.server_num,
            bernoulli_prob=args.bernoulli_prob,
            max_wait_time=args.max_wait_time * 1000,  # Convert to milliseconds
            enable_deny=args.enable_deny
        )

        # Create model
        agent.create_model()

        # Train
        print(f"Starting {algorithm} training...")
        callback = agent.train(
            total_timesteps=args.timesteps,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            save_freq=args.save_freq,
            save_path=os.path.join(args.save_path, algorithm),
            log_path=os.path.join(args.log_path, algorithm)
        )

        # Save final model
        final_model_path = os.path.join(args.save_path, algorithm, f"{algorithm}_final_{args.timesteps}.zip")
        agent.save(final_model_path)
        print(f"\n{algorithm} model saved to: {final_model_path}")

        agents[algorithm] = agent

    return agents

    # Create agent
    agent = LLMDataCenterAgent(
        algorithm=args.algorithm,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        device=args.device
    )

    # Create environment with shorter episode for faster training
    env = agent.create_env(
        total_time=args.episode_time * 1000,  # Convert to milliseconds
        server_num=args.server_num,
        bernoulli_prob=args.bernoulli_prob,
        max_wait_time=args.max_wait_time * 1000  # Convert to milliseconds
    )

    # Create model
    agent.create_model()

    # Train
    print("Starting training...")
    callback = agent.train(
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        save_freq=args.save_freq,
        save_path=args.save_path,
        log_path=args.log_path
    )

    # Save final model
    final_model_path = os.path.join(args.save_path, f"{args.algorithm}_final_{args.timesteps}.zip")
    agent.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    return agent


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate RL agent for LLM Data Center optimization')

    # Actions
    parser.add_argument('--train', action='store_true', help='Train the RL agent(s)')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate agents')
    parser.add_argument('--load-model', type=str, help='Path to load pretrained model')
    parser.add_argument('--algorithms', nargs='+', default=['PPO'],
                        choices=['PPO', 'A2C', 'DQN', 'SAC', 'TD3'],
                        help='RL algorithms to train/evaluate')

    # Environment arguments
    parser.add_argument('--enable-deny', action='store_true', default=True,
                        help='Enable deny action in environment')
    parser.add_argument('--disable-deny', dest='enable_deny', action='store_false',
                        help='Disable deny action in environment')

    # General RL arguments
    parser.add_argument('--timesteps', type=int, default=10000, help='Total training timesteps')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu/cuda/auto)')

    # On-policy algorithms (PPO, A2C)
    parser.add_argument('--n-steps', type=int, default=2048, help='Number of steps per update')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--n-epochs', type=int, default=10, help='Number of epochs (PPO only)')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--ent-coef', type=float, default=0.0, help='Entropy coefficient')
    parser.add_argument('--vf-coef', type=float, default=0.5, help='Value function coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Max gradient norm')
    parser.add_argument('--clip-range', type=float, default=0.2, help='PPO clip range')

    # Off-policy algorithms (DQN, SAC, TD3)
    parser.add_argument('--buffer-size', type=int, default=1000000, help='Replay buffer size')
    parser.add_argument('--learning-starts', type=int, default=100, help='Steps before learning')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update coefficient')
    parser.add_argument('--train-freq', type=int, default=1, help='Training frequency')
    parser.add_argument('--gradient-steps', type=int, default=1, help='Gradient steps per update')

    # DQN specific
    parser.add_argument('--target-update-interval', type=int, default=10000, help='Target network update interval')
    parser.add_argument('--exploration-fraction', type=float, default=0.1, help='Exploration fraction')
    parser.add_argument('--exploration-initial-eps', type=float, default=1.0, help='Initial exploration')
    parser.add_argument('--exploration-final-eps', type=float, default=0.05, help='Final exploration')

    # Environment arguments
    parser.add_argument('--episode-time', type=int, default=3600, help='Episode time in seconds for training')
    parser.add_argument('--eval-time', type=int, default=86400, help='Evaluation time in seconds (default: 24 hours)')
    parser.add_argument('--server-num', type=int, default=200, help='Number of servers')
    parser.add_argument('--bernoulli-prob', type=float, default=0.2, help='Request arrival probability')
    parser.add_argument('--max-wait-time', type=int, default=10, help='Max wait time in seconds')

    # Evaluation arguments
    parser.add_argument('--eval-freq', type=int, default=10000, help='Evaluation frequency during training')
    parser.add_argument('--n-eval-episodes', type=int, default=5, help='Number of evaluation episodes')
    parser.add_argument('--save-freq', type=int, default=10000, help='Model save frequency')
    parser.add_argument('--save-path', type=str, default='../result/datacenter_models/', help='Model save path')
    parser.add_argument('--log-path', type=str, default='../result/datacenter_logs/', help='Log path')

    # Baseline agents to include
    parser.add_argument('--baselines', nargs='+',
                        default=['all_fullkv', 'all_snapkv_64', 'rule_based_price', 'rule_based_deny', 'random'],
                        help='Baseline agents to evaluate')

    args = parser.parse_args()

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_path = os.path.join(args.save_path, timestamp)
    args.log_path = os.path.join(args.log_path, timestamp)

    rl_agent = None

    rl_agents = None

    # Train if requested
    if args.train:
        rl_agents = train_rl_agents(args)

    # Load model if specified
    elif args.load_model:
        print(f"\nLoading model from: {args.load_model}")
        # Detect algorithm from path or use first specified
        algorithm = args.algorithms[0]
        for alg in ['PPO', 'A2C', 'DQN', 'SAC', 'TD3']:
            if alg.lower() in args.load_model.lower():
                algorithm = alg
                break

        agent = LLMDataCenterAgent(
            algorithm=algorithm,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            device=args.device
        )
        # Create dummy environment for loading
        env = agent.create_env(enable_deny=args.enable_deny)
        agent.load(args.load_model, env)
        rl_agents = {algorithm: agent}

    # Evaluate if requested
    if args.evaluate:
        # Prepare agents for evaluation
        agents = {}

        # Add baseline agents
        for baseline_name in args.baselines:
            try:
                agents[baseline_name] = get_baseline_agent(baseline_name, enable_deny=args.enable_deny)
            except ValueError as e:
                print(f"Warning: {e}")

        # Add RL agents if available
        if 'rl_agents' in locals() and rl_agents is not None:
            for alg_name, agent in rl_agents.items():
                agents[f"RL_{alg_name}"] = agent.model

        # Environment settings for evaluation (full 24-hour simulation)
        eval_env_kwargs = {
            'total_time': args.eval_time * 1000,
            'server_num': args.server_num,
            'bernoulli_prob': args.bernoulli_prob,
            'max_wait_time': args.max_wait_time * 1000,
            'enable_deny': args.enable_deny
        }

        # Compare all agents
        results = compare_agents(agents, eval_env_kwargs, n_episodes=args.n_eval_episodes)

        # Plot comparison
        plot_comparison(results, save_path=f"comparison_{timestamp}.png")

    # If neither train nor evaluate, just compare baseline agents
    if not args.train and not args.evaluate:
        print("\nNo action specified. Comparing baseline agents...")

        agents = {}
        for baseline_name in args.baselines:
            try:
                agents[baseline_name] = get_baseline_agent(baseline_name, enable_deny=args.enable_deny)
            except ValueError as e:
                print(f"Warning: {e}")

        eval_env_kwargs = {
            'total_time': args.eval_time * 1000,
            'server_num': args.server_num,
            'bernoulli_prob': args.bernoulli_prob,
            'max_wait_time': args.max_wait_time * 1000,
            'enable_deny': args.enable_deny
        }

        results = compare_agents(agents, eval_env_kwargs, n_episodes=3)
        plot_comparison(results)


if __name__ == "__main__":
    main()