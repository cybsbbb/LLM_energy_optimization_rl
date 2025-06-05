"""
Training script for LLM Data Center KV-cache optimization using RL.
"""
import sys
sys.path.append("../")

import argparse
import os

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
        values = [s[key] for s in all_summaries]
        avg_summary[key] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    
    return avg_summary, all_summaries


def print_evaluation_results(agent_name: str, avg_summary: Dict):
    """Print evaluation results for an agent."""
    print(f"\n{'='*60}")
    print(f"Agent: {agent_name}")
    print(f"{'='*60}")
    
    for metric, stats in avg_summary.items():
        if isinstance(stats, dict):
            print(f"{metric}: {stats['mean']:.3f} ± {stats['std']:.3f}")
        else:
            print(f"{metric}: {stats:.3f}")


def compare_agents(agents: Dict, env_kwargs: Dict = None, n_episodes: int = 5):
    """Compare multiple agents."""
    print("\n" + "="*80)
    print("COMPARING AGENTS")
    print("="*80)
    
    results = {}
    
    for name, agent in agents.items():
        print(f"\nEvaluating {name}...")
        avg_summary, _ = evaluate_agent(agent, env_kwargs, n_episodes)
        results[name] = avg_summary
        print_evaluation_results(name, avg_summary)
    
    # Print comparison table
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
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
    
    return results


def plot_comparison(results: Dict, save_path: str = "comparison_plot.png"):
    """Plot comparison of agents across key metrics."""
    metrics = ['Success Rate', 'Average Score', 'Average Latency (s)', 
               'Total Energy Cost', 'episode_reward']
    
    n_agents = len(results)
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(20, 5))
    
    agent_names = list(results.keys())
    x = np.arange(n_agents)
    
    for i, metric in enumerate(metrics):
        values = []
        errors = []
        
        for agent in agent_names:
            if metric in results[agent]:
                values.append(results[agent][metric]['mean'])
                errors.append(results[agent][metric]['std'])
            else:
                values.append(0)
                errors.append(0)
        
        axes[i].bar(x, values, yerr=errors, capsize=5)
        axes[i].set_title(metric)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(agent_names, rotation=45, ha='right')
        axes[i].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def train_rl_agent(args):
    """Train the RL agent."""
    print("\n" + "="*80)
    print("TRAINING RL AGENT")
    print("="*80)
    print(f"Algorithm: {args.algorithm}")
    print(f"Total timesteps: {args.timesteps}")
    print(f"Learning rate: {args.learning_rate}")
    print("="*80 + "\n")
    
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
    parser.add_argument('--train', action='store_true', help='Train the RL agent')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate agents')
    parser.add_argument('--load-model', type=str, help='Path to load pretrained model')
    
    # RL arguments
    parser.add_argument('--algorithm', type=str, default='PPO', choices=['PPO', 'A2C', 'DQN'],
                        help='RL algorithm to use')
    parser.add_argument('--timesteps', type=int, default=500000, help='Total training timesteps')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--n-steps', type=int, default=2048, help='Number of steps per update')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--n-epochs', type=int, default=10, help='Number of epochs (PPO only)')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu/cuda/auto)')
    
    # Environment arguments
    parser.add_argument('--episode-time', type=int, default=86400, help='Episode time in seconds for training')
    parser.add_argument('--eval-time', type=int, default=86400, help='Evaluation time in seconds (default: 24 hours)')
    parser.add_argument('--server-num', type=int, default=200, help='Number of servers')
    parser.add_argument('--bernoulli-prob', type=float, default=0.2, help='Request arrival probability')
    parser.add_argument('--max-wait-time', type=int, default=10, help='Max wait time in seconds')
    
    # Evaluation arguments
    parser.add_argument('--eval-freq', type=int, default=100000, help='Evaluation frequency during training')
    parser.add_argument('--n-eval-episodes', type=int, default=1, help='Number of evaluation episodes')
    parser.add_argument('--save-freq', type=int, default=100000, help='Model save frequency')
    parser.add_argument('--save-path', type=str, default='./datacenter_models/', help='Model save path')
    parser.add_argument('--log-path', type=str, default='./datacenter_logs/', help='Log path')
    
    # Baseline agents to include
    parser.add_argument('--baselines', nargs='+', 
                        default=['all_fullkv', 'all_snapkv_64', 'rule_based_price', 'smart_deny', 'random'],
                        help='Baseline agents to evaluate')
    
    args = parser.parse_args()
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_path = os.path.join(args.save_path, timestamp)
    args.log_path = os.path.join(args.log_path, timestamp)
    
    rl_agent = None
    
    # Train if requested
    if args.train:
        rl_agent = train_rl_agent(args)
    
    # Load model if specified
    elif args.load_model:
        print(f"\nLoading model from: {args.load_model}")
        rl_agent = LLMDataCenterAgent(
            algorithm=args.algorithm,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            device=args.device
        )
        # Create dummy environment for loading
        env = rl_agent.create_env()
        rl_agent.load(args.load_model, env)
    
    # Evaluate if requested
    if args.evaluate:
        # Prepare agents for evaluation
        agents = {}
        
        # Add baseline agents
        for baseline_name in args.baselines:
            try:
                agents[baseline_name] = get_baseline_agent(baseline_name)
            except ValueError as e:
                print(f"Warning: {e}")
        
        # Add RL agent if available
        if rl_agent is not None:
            agents[f"RL_{args.algorithm}"] = rl_agent.model
        
        # Environment settings for evaluation (full 24-hour simulation)
        eval_env_kwargs = {
            'total_time': args.eval_time * 1000,
            'server_num': args.server_num,
            'bernoulli_prob': args.bernoulli_prob,
            'max_wait_time': args.max_wait_time * 1000
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
                agents[baseline_name] = get_baseline_agent(baseline_name)
            except ValueError as e:
                print(f"Warning: {e}")
        
        eval_env_kwargs = {
            'total_time': args.eval_time * 1000,
            'server_num': args.server_num,
            'bernoulli_prob': args.bernoulli_prob,
            'max_wait_time': args.max_wait_time * 1000
        }
        
        results = compare_agents(agents, eval_env_kwargs, n_episodes=args.n_eval_episodes)
        plot_comparison(results)


if __name__ == "__main__":
    main()