"""
Training script for LLM Data Center RL agents.
Handles training of different RL algorithms for KV-cache optimization.
"""

import os
import argparse
from datetime import datetime
from llm_data_center_rl.src.rl_environments.llm_datacenter_env import LLMDataCenterEnv
from rl_agents.llm_datacenter_agent import LLMDataCenterAgent


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
        print(f"Episode time: {args.episode_time} seconds")
        print(f"Server count: {args.server_num}")
        print(f"Request probability: {args.bernoulli_prob}")
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

        # Create environment with specified parameters
        env = agent.create_env(
            total_time=args.episode_time * 1000,  # Convert to milliseconds
            time_interval=args.time_interval,
            server_num=args.server_num,
            bernoulli_prob=args.bernoulli_prob,
            max_wait_time=args.max_wait_time * 1000,  # Convert to milliseconds
            step_time_ms=args.step_time * 1000,  # Convert to milliseconds
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
            save_path=str(os.path.join(args.save_path, algorithm)),
            log_path=str(os.path.join(args.log_path, algorithm))
        )

        # Save final model with detailed naming
        final_model_path = os.path.join(args.save_path, algorithm, f"{algorithm}_final_{args.timesteps}.zip")
        agent.save(final_model_path)
        print(f"\n{algorithm} model saved to: {final_model_path}")

        # Save training configuration
        config_path = os.path.join(args.save_path, algorithm, "training_config.txt")
        save_training_config(args, algorithm, config_path)

        agents[algorithm] = agent

    # Save overall training summary
    summary_path = os.path.join(args.save_path, "training_summary.txt")
    save_training_summary(args, list(agents.keys()), summary_path)

    return agents


def save_training_config(args, algorithm, config_path):
    """Save training configuration to file."""
    with open(config_path, 'w') as f:
        f.write(f"Training Configuration for {algorithm}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Algorithm: {algorithm}\n")
        f.write(f"Total timesteps: {args.timesteps}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Gamma: {args.gamma}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Enable deny: {args.enable_deny}\n\n")

        f.write("Environment Configuration:\n")
        f.write(f"Episode time: {args.episode_time} seconds\n")
        f.write(f"Server count: {args.server_num}\n")
        f.write(f"Bernoulli probability: {args.bernoulli_prob}\n")
        f.write(f"Time interval: {args.time_interval} ms\n")
        f.write(f"Max wait time: {args.max_wait_time} seconds\n")
        f.write(f"Step time: {args.step_time} seconds\n\n")

        f.write("Training Configuration:\n")
        f.write(f"Evaluation frequency: {args.eval_freq}\n")
        f.write(f"Evaluation episodes: {args.n_eval_episodes}\n")
        f.write(f"Save frequency: {args.save_freq}\n")

        if algorithm in ['PPO', 'A2C']:
            f.write(f"N steps: {args.n_steps}\n")
            f.write(f"Batch size: {args.batch_size}\n")
            f.write(f"N epochs: {args.n_epochs}\n")
            f.write(f"GAE lambda: {args.gae_lambda}\n")
            f.write(f"Entropy coefficient: {args.ent_coef}\n")
            f.write(f"Value function coefficient: {args.vf_coef}\n")
            f.write(f"Max grad norm: {args.max_grad_norm}\n")
            if algorithm == 'PPO':
                f.write(f"Clip range: {args.clip_range}\n")

        elif algorithm in ['DQN', 'SAC', 'TD3']:
            f.write(f"Batch size: {args.batch_size}\n")
            f.write(f"Buffer size: {args.buffer_size}\n")
            f.write(f"Learning starts: {args.learning_starts}\n")
            f.write(f"Tau: {args.tau}\n")
            f.write(f"Train frequency: {args.train_freq}\n")
            f.write(f"Gradient steps: {args.gradient_steps}\n")


def save_training_summary(args, algorithms, summary_path):
    """Save overall training summary."""
    with open(summary_path, 'w') as f:
        f.write("Training Session Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Algorithms trained: {', '.join(algorithms)}\n")
        f.write(f"Total timesteps per algorithm: {args.timesteps}\n")
        f.write(f"Episode duration: {args.episode_time} seconds\n")
        f.write(f"Environment servers: {args.server_num}\n")
        f.write(f"Request probability: {args.bernoulli_prob}\n")
        f.write(f"Deny action enabled: {args.enable_deny}\n\n")

        f.write("Saved Models:\n")
        for algorithm in algorithms:
            model_path = os.path.join(
                args.save_path,
                algorithm,
                f"{algorithm}_final_{args.timesteps}_{args.episode_time}s_{args.server_num}servers.zip"
            )
            f.write(f"  {algorithm}: {model_path}\n")


def quick_evaluation(agent, algorithm_name, args):
    """Run a quick evaluation after training."""
    print(f"\nRunning quick evaluation for {algorithm_name}...")

    # Create evaluation environment (shorter for quick check)
    eval_env = LLMDataCenterEnv(
        total_time=24 * 3600 * 1000,  # 1 hour
        server_num=args.server_num,
        bernoulli_prob=args.bernoulli_prob,
        enable_deny=args.enable_deny
    )

    # Run evaluation
    mean_reward, std_reward = agent.evaluate(n_eval_episodes=3, render=False)
    print(f"{algorithm_name} - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    return mean_reward, std_reward


def main():
    parser = argparse.ArgumentParser(description='Train RL agent for LLM Data Center optimization')

    # Core training arguments
    parser.add_argument('--algorithms', nargs='+', default=['PPO', 'A2C', 'DQN', 'SAC', 'TD3'],
                        choices=['PPO', 'A2C', 'DQN', 'SAC', 'TD3'],
                        help='RL algorithms to train')
    parser.add_argument('--timesteps', type=int, default=20000,
                        help='Total training timesteps')
    parser.add_argument('--quick-eval', action='store_true',
                        help='Run quick evaluation after training')

    # Environment arguments
    parser.add_argument('--enable-deny', action='store_true', default=False,
                        help='Enable deny action in environment')
    parser.add_argument('--episode-time', type=int, default=24 * 3600,  # 24 hours default for training
                        help='Episode time in seconds for training (default: 4 hours)')
    parser.add_argument('--server-num', type=int, default=200,
                        help='Number of servers')
    parser.add_argument('--bernoulli-prob', type=float, default=0.12,
                        help='Request arrival probability')
    parser.add_argument('--time-interval', type=int, default=10,
                        help='Time interval in milliseconds')
    parser.add_argument('--max-wait-time', type=int, default=10,
                        help='Max wait time in seconds')
    parser.add_argument('--step-time', type=int, default=10,
                        help='Time window per step in seconds')

    # General RL arguments
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu/cuda/auto)')

    # On-policy algorithms (PPO, A2C)
    parser.add_argument('--n-steps', type=int, default=2048,
                        help='Number of steps per update')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--n-epochs', type=int, default=10,
                        help='Number of epochs (PPO only)')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='GAE lambda')
    parser.add_argument('--ent-coef', type=float, default=0.0,
                        help='Entropy coefficient')
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help='Value function coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='Max gradient norm')
    parser.add_argument('--clip-range', type=float, default=0.2,
                        help='PPO clip range')

    # Off-policy algorithms (DQN, SAC, TD3)
    parser.add_argument('--buffer-size', type=int, default=1000000,
                        help='Replay buffer size')
    parser.add_argument('--learning-starts', type=int, default=100,
                        help='Steps before learning')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Soft update coefficient')
    parser.add_argument('--train-freq', type=int, default=1,
                        help='Training frequency')
    parser.add_argument('--gradient-steps', type=int, default=1,
                        help='Gradient steps per update')

    # DQN specific
    parser.add_argument('--target-update-interval', type=int, default=10000,
                        help='Target network update interval')
    parser.add_argument('--exploration-fraction', type=float, default=0.1,
                        help='Exploration fraction')
    parser.add_argument('--exploration-initial-eps', type=float, default=1.0,
                        help='Initial exploration')
    parser.add_argument('--exploration-final-eps', type=float, default=0.05,
                        help='Final exploration')

    # Training monitoring
    parser.add_argument('--eval-freq', type=int, default=10000,
                        help='Evaluation frequency during training')
    parser.add_argument('--n-eval-episodes', type=int, default=5,
                        help='Number of evaluation episodes')
    parser.add_argument('--save-freq', type=int, default=10000,
                        help='Model save frequency')

    # Paths
    parser.add_argument('--save-path', type=str, default='../result/datacenter_models/',
                        help='Model save path')
    parser.add_argument('--log-path', type=str, default='../result/datacenter_logs/',
                        help='Log path')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name (default: timestamp)')

    args = parser.parse_args()

    # Create experiment directory
    if args.experiment_name:
        experiment_dir = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = f"training_{timestamp}"

    args.save_path = os.path.join(args.save_path, experiment_dir)
    args.log_path = os.path.join(args.log_path, experiment_dir)

    # Create base directories
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    print("=" * 80)
    print("LLM DATA CENTER RL AGENT TRAINING")
    print("=" * 80)
    print(f"Experiment: {experiment_dir}")
    print(f"Algorithms: {', '.join(args.algorithms)}")
    print(f"Total timesteps: {args.timesteps}")
    print(f"Episode time: {args.episode_time} seconds")
    print(f"Models will be saved to: {args.save_path}")
    print("=" * 80)

    # Train agents
    trained_agents = train_rl_agents(args)

    # Quick evaluation if requested
    if args.quick_eval:
        print("\n" + "=" * 80)
        print("QUICK EVALUATION RESULTS")
        print("=" * 80)

        eval_results = {}
        for algorithm, agent in trained_agents.items():
            mean_reward, std_reward = quick_evaluation(agent, algorithm, args)
            eval_results[algorithm] = (mean_reward, std_reward)

        # Print summary
        print(f"\n{'Algorithm':<15} {'Mean Reward':<15} {'Std Reward':<15}")
        print("-" * 45)
        for algorithm, (mean_rew, std_rew) in eval_results.items():
            print(f"{algorithm:<15} {mean_rew:<15.3f} {std_rew:<15.3f}")

    print(f"\nTraining completed! Models saved in: {args.save_path}")
    print("Use evaluate_datacenter_agent.py to run detailed evaluation and analysis.")


if __name__ == "__main__":
    main()
