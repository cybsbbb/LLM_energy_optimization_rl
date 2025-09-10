"""
Main training script for LLM Data Center RL agents.
"""

import argparse
import os
import sys
from datetime import datetime

# Add the parent directory to the path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llm_datacenter_rl.config.config import ExperimentConfig, get_config
from llm_datacenter_rl.training.trainer import Trainer
from llm_datacenter_rl.utils.logger import get_logger


def main():
    parser = argparse.ArgumentParser(description='Train RL agents for LLM Data Center optimization')

    # Configuration
    parser.add_argument('--config', type=str, default='realistic_mode',
                        choices=['synthetic_mode', 'realistic_mode'],
                        help='Predefined configuration to use')
    parser.add_argument('--experiment-name', type=str, help='Custom experiment name')

    # Override training parameters
    parser.add_argument('--algorithms', nargs='+',
                        choices=['PPO', 'A2C', 'DQN', 'SAC', 'TD3'],
                        help='RL algorithms to train')
    parser.add_argument('--timesteps', type=int, help='Total training timesteps')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')

    # Override environment parameters
    parser.add_argument('--enable-deny', action='store_true', help='Enable deny action')
    parser.add_argument('--server-num', type=int, help='Number of servers')
    parser.add_argument('--simulation-hours', type=int, help='Simulation duration in hours')

    # Realistic mode data files
    parser.add_argument('--invocations-file', type=str, help='Path to invocations CSV file')
    parser.add_argument('--energy-price-file', type=str, help='Path to energy price CSV file')
    parser.add_argument('--simulation-start-time', type=str, help='Simulation start time (YYYY-MM-DD HH:MM:SS)')

    # Output directory
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')

    args = parser.parse_args()

    # Load base configuration
    config = get_config(args.config)

    # Apply command line overrides
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.algorithms:
        config.algorithms = args.algorithms
    if args.timesteps:
        config.training.total_timesteps = args.timesteps
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.enable_deny:
        config.environment.enable_deny = args.enable_deny
    if args.server_num:
        config.environment.server_num = args.server_num
    if args.simulation_hours:
        config.environment.simulation_duration_hours = args.simulation_hours
    if args.invocations_file:
        config.environment.invocations_file = args.invocations_file
    if args.energy_price_file:
        config.environment.energy_price_file = args.energy_price_file
    if args.simulation_start_time:
        config.environment.simulation_start_time = args.simulation_start_time
    if args.output_dir:
        config.output_dir = args.output_dir

    # Set experiment name if not provided
    if not config.experiment_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.experiment_name = f"training_{timestamp}"

    # Initialize logger
    logger = get_logger(__name__)

    # Print configuration
    logger.info("=" * 80)
    logger.info("LLM DATA CENTER RL TRAINING")
    logger.info("=" * 80)
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Algorithms: {config.algorithms}")
    logger.info(f"Total timesteps: {config.training.total_timesteps:,}")
    logger.info(f"Learning rate: {config.training.learning_rate}")
    logger.info(f"Environment mode: {'Realistic' if config.environment.is_realistic_mode else 'Synthetic'}")
    logger.info(f"Simulation duration: {config.environment.simulation_duration_hours} hours")
    logger.info(f"Servers: {config.environment.server_num}")
    logger.info(f"Enable deny: {config.environment.enable_deny}")
    logger.info(f"Output directory: {config.get_output_paths()['base']}")
    logger.info("=" * 80)

    # Verify data files for realistic mode
    if config.environment.is_realistic_mode:
        if not os.path.exists(config.environment.invocations_file):
            logger.error(f"Invocations file not found: {config.environment.invocations_file}")
            return 1
        if not os.path.exists(config.environment.energy_price_file):
            logger.error(f"Energy price file not found: {config.environment.energy_price_file}")
            return 1
        logger.info(f"Using invocations file: {config.environment.invocations_file}")
        logger.info(f"Using energy price file: {config.environment.energy_price_file}")

    try:
        # Create trainer and run training
        trainer = Trainer(config)
        results = trainer.train_all_agents()

        # Print results summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 80)

        successful_trainings = [alg for alg, result in results.items() if "error" not in result]
        failed_trainings = [alg for alg, result in results.items() if "error" in result]

        logger.info(f"Successfully trained: {successful_trainings}")
        if failed_trainings:
            logger.info(f"Failed to train: {failed_trainings}")

        logger.info(f"\nModels saved in: {config.get_output_paths()['models']}")
        logger.info(f"To evaluate the trained models, run:")
        logger.info(
            f"python scripts/evaluate.py --models-dir {config.get_output_paths()['models']} --config {args.config}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

