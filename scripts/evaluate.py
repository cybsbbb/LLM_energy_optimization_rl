"""
Main evaluation script for LLM Data Center RL agents.
"""

import argparse
import os
import sys
from datetime import datetime

# Add the parent directory to the path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llm_datacenter_rl.config.config import ExperimentConfig, get_config
from llm_datacenter_rl.evaluation.evaluator import Evaluator
from llm_datacenter_rl.utils.logger import get_logger


def main():
    parser = argparse.ArgumentParser(description='Evaluate RL agents for LLM Data Center optimization')

    # Configuration
    parser.add_argument('--config', type=str, default='realistic_mode',
                        choices=['synthetic_mode', 'realistic_mode'],
                        help='Predefined configuration to use')
    parser.add_argument('--experiment-name', type=str, help='Custom experiment name for output')

    # Model loading
    parser.add_argument('--models-dir', type=str, required=True,
                        help='Directory containing trained models')

    # Override evaluation parameters
    parser.add_argument('--n-eval-episodes', type=int, default=5,
                        help='Number of evaluation episodes')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')
    parser.add_argument('--render', action='store_true', help='Render environment during evaluation')

    # Algorithm and baseline selection
    parser.add_argument('--algorithms', nargs='+',
                        choices=['PPO', 'A2C', 'DQN', 'SAC', 'TD3'],
                        help='RL algorithms to evaluate')
    parser.add_argument('--baselines', nargs='+',
                        choices=['all_fullkv', 'all_snapkv_64', 'rule_based_price', 'smart_deny', 'adaptive_load', 'random'],
                        help='Baseline agents to evaluate')

    # Override environment parameters
    parser.add_argument('--enable-deny', action='store_true', help='Enable deny action')
    parser.add_argument('--server-num', type=int, help='Number of servers')
    parser.add_argument('--simulation-hours', type=int, help='Simulation duration in hours')

    # Realistic mode data files
    parser.add_argument('--invocations-file', type=str, help='Path to invocations CSV file')
    parser.add_argument('--energy-price-file', type=str, help='Path to energy price CSV file')
    parser.add_argument('--simulation-start-time', type=str, help='Simulation start time')

    # Output directory
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')

    args = parser.parse_args()

    # Verify models directory exists
    if not os.path.exists(args.models_dir):
        print(f"Error: Models directory not found: {args.models_dir}")
        return 1

    # Load base configuration
    config = get_config(args.config)

    # Apply command line overrides
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.algorithms:
        config.algorithms = args.algorithms
    if args.baselines:
        config.baseline_agents = args.baselines
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

    # Set evaluation parameters
    config.evaluation.n_eval_episodes = args.n_eval_episodes
    config.evaluation.save_plots = not args.no_plots
    config.evaluation.render = args.render

    # Set experiment name if not provided
    if not config.experiment_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.experiment_name = f"evaluation_{timestamp}"

    # Initialize logger
    logger = get_logger(__name__)

    # Print configuration
    logger.info("=" * 80)
    logger.info("LLM DATA CENTER RL EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Models directory: {args.models_dir}")
    logger.info(f"Algorithms: {config.algorithms}")
    logger.info(f"Baseline agents: {config.baseline_agents}")
    logger.info(f"Evaluation episodes: {config.evaluation.n_eval_episodes}")
    logger.info(f"Environment mode: {'Realistic' if config.environment.is_realistic_mode else 'Synthetic'}")
    logger.info(f"Simulation duration: {config.environment.simulation_duration_hours} hours")
    logger.info(f"Servers: {config.environment.server_num}")
    logger.info(f"Enable deny: {config.environment.enable_deny}")
    logger.info(f"Generate plots: {config.evaluation.save_plots}")
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

    try:
        # Create evaluator and run evaluation
        evaluator = Evaluator(config)
        results = evaluator.load_and_evaluate_trained_models(args.models_dir)

        if not results:
            logger.error("No results to display. Check that models exist and are loadable.")
            return 1

        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Evaluated {len(results)} agents successfully")

        if config.evaluation.save_plots:
            logger.info(f"Plots saved in: {config.get_output_paths()['plots']}")

        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
