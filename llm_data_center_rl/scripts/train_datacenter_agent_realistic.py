#!/usr/bin/env python3
"""
Optimized training script for Realistic LLM Data Center RL Agent.
Uses the optimized environment with binary search for maximum performance.
"""

import os
import sys
import argparse
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List
import matplotlib.pyplot as plt

# Import optimized environment
from rl_environments.llm_datacenter_realistic_env import RealisticLLMDataCenterEnv
from rl_agents.llm_datacenter_agent_realistic import LLMDataCenterAgent


class OptimizedRealisticLLMDataCenterAgent(LLMDataCenterAgent):
    """Enhanced agent class that works with the optimized realistic environment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_start_time = None
        self.lookup_performance_stats = {}

    def create_env(self, **env_kwargs):
        """Create the optimized realistic environment."""
        from rl_agents.llm_datacenter_agent_realistic import SafeMonitor, DiscreteToBoxWrapper

        # Use optimized environment
        base_env = RealisticLLMDataCenterEnv(**env_kwargs)

        # For continuous algorithms, wrap with DiscreteToBoxWrapper
        if self.algorithm in ['SAC', 'TD3']:
            self.wrapped_env = DiscreteToBoxWrapper(base_env)
            self.env = SafeMonitor(self.wrapped_env)
        else:
            from stable_baselines3.common.monitor import Monitor
            self.env = Monitor(base_env)

        return self.env

    def train(self, total_timesteps: int = 100000, **kwargs):
        """Enhanced training with performance monitoring."""
        print(f"\n🚀 Starting optimized training...")
        print(f"Algorithm: {self.algorithm}")
        print(f"Total timesteps: {total_timesteps:,}")

        # Record training start time
        self.training_start_time = time.time()

        # Get initial performance stats
        base_env = self._get_base_env()
        if hasattr(base_env, 'get_performance_stats'):
            initial_stats = base_env.get_performance_stats()
            print(f"Environment loaded with {initial_stats['energy_data_points']:,} energy data points")
            print(f"and {initial_stats['invocation_data_points']:,} invocation data points")

        # Call parent training method
        callback = super().train(total_timesteps=total_timesteps, **kwargs)

        # Record training end time and collect final stats
        training_time = time.time() - self.training_start_time

        if hasattr(base_env, 'get_performance_stats'):
            final_stats = base_env.get_performance_stats()
            self.lookup_performance_stats = final_stats

            print(f"\n📊 Training Performance Summary:")
            print(f"  Total training time: {training_time:.1f} seconds ({training_time / 60:.1f} minutes)")
            # print(f"  Cache efficiency: {final_stats['cache_hit_ratio_estimate']:.1%}")
            print(
                f"  Final cache sizes: Energy={final_stats['energy_cache_size']}, Invocations={final_stats['invocation_cache_size']}")

        return callback

    def _get_base_env(self):
        """Get the base environment from wrapped environment."""
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        return env


def train_optimized_realistic_agents(args):
    """Train RL agents using the optimized realistic environment."""
    agents = {}

    for algorithm in args.algorithms:
        print("\n" + "=" * 80)
        print(f"TRAINING {algorithm} WITH OPTIMIZED ENVIRONMENT")
        print("=" * 80)
        print(f"🎯 Configuration:")
        print(f"  Algorithm: {algorithm}")
        print(f"  Total timesteps: {args.timesteps:,}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Training duration: {args.training_duration_hours} hours")
        print(f"  Time acceleration: {args.time_acceleration_factor}x")
        print(f"  Request scaling: {args.request_scaling_factor}x")
        print(f"  Enable deny: {args.enable_deny}")
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

        agent = OptimizedRealisticLLMDataCenterAgent(**agent_params)

        # Create optimized environment
        print(f"🔧 Creating optimized environment...")
        env_start_time = time.time()

        env = agent.create_env(
            invocations_file=args.invocations_file,
            energy_price_file=args.energy_price_file,
            simulation_start_time=args.training_start_time,
            simulation_duration_hours=args.training_duration_hours,
            time_acceleration_factor=args.time_acceleration_factor,
            server_num=args.server_num,
            request_scaling_factor=args.request_scaling_factor,
            enable_deny=args.enable_deny
        )

        env_creation_time = time.time() - env_start_time
        print(f"✓ Environment created in {env_creation_time:.2f}s")

        # Create model
        print(f"🤖 Creating {algorithm} model...")
        model_start_time = time.time()
        agent.create_model()
        model_creation_time = time.time() - model_start_time
        print(f"✓ Model created in {model_creation_time:.2f}s")

        # Train with performance monitoring
        callback = agent.train(
            total_timesteps=args.timesteps,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            save_freq=args.save_freq,
            save_path=str(os.path.join(args.save_path, algorithm)),
            log_path=str(os.path.join(args.log_path, algorithm))
        )

        # Save final model
        final_model_path = os.path.join(args.save_path, algorithm,
                                        f"{algorithm}_optimized_realistic_{args.timesteps}.zip")
        agent.save(final_model_path)
        print(f"\n💾 {algorithm} model saved to: {final_model_path}")

        agents[algorithm] = agent

    return agents


def main():
    parser = argparse.ArgumentParser(description='Train RL agent using optimized realistic environment')
    # Data files
    parser.add_argument('--invocations-file', type=str, default='../data/LLM_inference_trace/processed/2024/invocations_ten_second_Coding.csv',
                        help='Path to invocations CSV file')
    parser.add_argument('--energy-price-file', type=str, default='../data/energy_price/processed/CAISO_20240509_20240521.csv',
                        help='Path to energy price CSV file')

    # Training configuration
    parser.add_argument('--algorithms', nargs='+', default=['PPO', 'A2C', 'DQN', 'SAC', 'TD3'],
                        choices=['PPO', 'A2C', 'DQN', 'SAC', 'TD3'],
                        help='RL algorithms to train')
    parser.add_argument('--timesteps', type=int, default=1000000, help='Total training timesteps')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu/cuda/auto)')

    # Optimized simulation parameters
    parser.add_argument('--training-start-time', type=str, default='2024-05-10 00:00:00',
                        help='Start time for training simulation')
    parser.add_argument('--training-duration-hours', type=int, default=24*7,
                        help='Duration of each training episode in hours')
    parser.add_argument('--time-acceleration-factor', type=float, default=1.0,
                        help='Time acceleration factor (1.0 = real time)')
    parser.add_argument('--request-scaling-factor', type=float, default=1.0,
                        help='Scale request volume (0.2 = 20% of real volume)')

    # Environment arguments
    parser.add_argument('--enable-deny', action='store_true', default=False,
                        help='Enable deny action in environment')
    parser.add_argument('--server-num', type=int, default=200, help='Number of servers')

    # Algorithm hyperparameters
    parser.add_argument('--n-steps', type=int, default=2048, help='Number of steps per update')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--n-epochs', type=int, default=10, help='Number of epochs (PPO only)')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--ent-coef', type=float, default=0.0, help='Entropy coefficient')
    parser.add_argument('--vf-coef', type=float, default=0.5, help='Value function coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Max gradient norm')
    parser.add_argument('--clip-range', type=float, default=0.2, help='PPO clip range')

    # Off-policy algorithms
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

    # Training monitoring
    parser.add_argument('--eval-freq', type=int, default=10000, help='Evaluation frequency during training')
    parser.add_argument('--n-eval-episodes', type=int, default=3, help='Number of evaluation episodes during training')
    parser.add_argument('--save-freq', type=int, default=10000, help='Model save frequency')
    parser.add_argument('--save-path', type=str, default='../result/optimized_models/', help='Model save path')
    parser.add_argument('--log-path', type=str, default='../result/optimized_logs/', help='Log path')

    args = parser.parse_args()

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_path = os.path.join(args.save_path, timestamp)
    args.log_path = os.path.join(args.log_path, timestamp)

    # Create directories
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    print("=" * 80)
    print("🚀 OPTIMIZED REALISTIC LLM DATA CENTER RL TRAINING")
    print("=" * 80)
    print(f"🔧 Using optimized binary search environment")
    print(f"📁 Data files:")
    print(f"  - Invocations: {args.invocations_file}")
    print(f"  - Energy prices: {args.energy_price_file}")
    print(f"🎯 Training configuration:")
    print(f"  - Algorithms: {args.algorithms}")
    print(f"  - Total timesteps: {args.timesteps:,}")
    print(f"  - Training duration: {args.training_duration_hours} hours")
    print(f"  - Time acceleration: {args.time_acceleration_factor}x")
    print(f"  - Request scaling: {args.request_scaling_factor}x")
    print(f"💾 Save path: {args.save_path}")
    print("=" * 80)

    # Verify data files exist
    if not os.path.exists(args.invocations_file):
        print(f"❌ Error: Invocations file not found: {args.invocations_file}")
        return

    if not os.path.exists(args.energy_price_file):
        print(f"❌ Error: Energy price file not found: {args.energy_price_file}")
        return

    if args.skip_training:
        print("⏭️  Skipping training (--skip-training flag)")
        return

    # Train agents
    try:
        training_start_time = time.time()
        trained_agents = train_optimized_realistic_agents(args)
        total_training_time = time.time() - training_start_time

        print("\n" + "=" * 80)
        print("✅ OPTIMIZED TRAINING COMPLETED")
        print("=" * 80)
        print(f"🎯 Results:")
        print(f"  - Trained {len(trained_agents)} agent(s)")
        print(f"  - Total training time: {total_training_time:.1f}s ({total_training_time / 60:.1f} minutes)")
        print(f"  - Using optimized O(log n) lookups")

        print(f"\n💾 Saved models:")
        for alg_name in trained_agents.keys():
            model_path = os.path.join(args.save_path, alg_name, f"{alg_name}_optimized_realistic_{args.timesteps}.zip")
            print(f"  - {alg_name}: {model_path}")

        print(f"\n🔬 To evaluate the trained agents:")
        print(f"python evaluate_realistic_agent.py --load-models {args.save_path}")

        print(f"\n📊 To test performance improvement:")
        print(f"python test_performance_improvement.py")

        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()