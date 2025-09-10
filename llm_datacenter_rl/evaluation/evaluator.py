import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from ..config.config import ExperimentConfig, EvaluationConfig
from ..environments.environment import LLMDataCenterEnv
from ..agents.rl_based_agents import get_rl_agent, RLAgent
from ..agents.rule_based_agents import get_baseline_agent
from ..utils.logger import get_logger
from ..utils.plotting import ActionTimeLogger, plot_action_timeline, plot_performance_comparison


class Evaluator:
    """Unified evaluator for all types of agents."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = get_logger(__name__)

        # Create directories
        config.create_directories()
        self.plots_dir = config.get_output_paths()["plots"]

    def evaluate_agent(self, agent, agent_name: str) -> Tuple[Dict, ActionTimeLogger]:
        """Evaluate a single agent."""
        self.logger.info(f"Evaluating {agent_name}")

        # Create environment
        env = LLMDataCenterEnv(self.config.environment)

        # Wrap environment if it's an RL agent that needs it
        if isinstance(agent, RLAgent):
            env = agent.create_env(env)

        logger = ActionTimeLogger()

        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0

        while not done:
            action = agent.predict(obs, deterministic=self.config.evaluation.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

            # Log step data
            current_time = env.get_wrapper_attr("current_real_time")
            energy_price = env.get_wrapper_attr("data_provider").get_energy_price(current_time)
            processing_ratio = env.get_wrapper_attr("processing_server_num") / env.get_wrapper_attr("config").server_num
            waiting_ratio = len(env.get_wrapper_attr("waiting_queue")) / env.get_wrapper_attr("config").server_num

            # current_time = env.current_real_time
            # energy_price = env.data_provider.get_energy_price(current_time)
            # processing_ratio = env.processing_server_num / env.config.server_num
            # waiting_ratio = len(env.waiting_queue) / env.config.server_num

            logger.log_step(
                current_time, action, energy_price,
                processing_ratio, waiting_ratio, reward
            )

            step_count += 1

            if self.config.evaluation.render:
                env.render()

        # Get final summary

        # summary = env.get_final_summary()
        summary = env.get_wrapper_attr("get_final_summary")()
        summary['episode_reward'] = episode_reward
        summary['total_steps'] = step_count

        env.close()

        self.logger.info(f"{agent_name} evaluation completed: "
                         f"Reward={episode_reward:.3f}, "
                         f"Success Rate={summary.get('Success Rate', 0):.3f}")

        return summary, logger

    def evaluate_multiple_agents(self, agents: Dict) -> Dict:
        """Evaluate multiple agents and generate comparison."""
        results = {}
        loggers = {}

        self.logger.info(f"Evaluating {len(agents)} agents")

        for name, agent in agents.items():
            try:
                summary, logger = self.evaluate_agent(agent, name)
                results[name] = summary
                loggers[name] = logger
            except Exception as e:
                self.logger.error(f"Failed to evaluate {name}: {e}")
                continue

        # Generate plots if requested
        if self.config.evaluation.save_plots and loggers:
            self._generate_plots(results, loggers)

        # Print comparison
        self._print_comparison(results)

        return results

    def load_and_evaluate_trained_models(self, models_dir: str) -> Dict:
        """Load trained models and evaluate them along with baselines."""
        agents = {}

        # Load baseline agents
        for baseline_name in self.config.baseline_agents:
            try:
                agent = get_baseline_agent(baseline_name, self.config.environment.enable_deny)
                agents[baseline_name] = agent
                self.logger.info(f"Added baseline agent: {baseline_name}")
            except ValueError as e:
                self.logger.warning(f"Skipping baseline {baseline_name}: {e}")

        # Load RL agents
        for algorithm in self.config.algorithms:
            algorithm_dir = os.path.join(models_dir, algorithm)
            if not os.path.exists(algorithm_dir):
                self.logger.warning(f"No trained model found for {algorithm}")
                continue

            # Look for final model
            model_files = [f for f in os.listdir(algorithm_dir) if f.endswith('.zip') and 'final' in f]
            if not model_files:
                model_files = [f for f in os.listdir(algorithm_dir) if f.endswith('.zip')]

            if model_files:
                model_path = os.path.join(algorithm_dir, model_files[0])
                try:
                    # Create dummy environment for loading
                    dummy_env = LLMDataCenterEnv(self.config.environment)
                    agent = get_rl_agent(algorithm, self.config.training, self.config.environment.enable_deny)
                    wrapped_env = agent.create_env(dummy_env)
                    agent.load(model_path, wrapped_env)

                    agents[f"RL_{algorithm}"] = agent
                    self.logger.info(f"Loaded RL agent: {algorithm}")
                    dummy_env.close()
                except Exception as e:
                    self.logger.error(f"Failed to load {algorithm}: {e}")

        if not agents:
            self.logger.error("No agents to evaluate")
            return {}

        return self.evaluate_multiple_agents(agents)

    def _generate_plots(self, results: Dict, loggers: Dict):
        """Generate evaluation plots."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.config.evaluation.plot_timeline:
            timeline_path = os.path.join(self.plots_dir, f"action_timeline_{timestamp}.png")
            plot_action_timeline(loggers, timeline_path, self.config.environment.enable_deny)

        if self.config.evaluation.plot_comparison:
            comparison_path = os.path.join(self.plots_dir, f"performance_comparison_{timestamp}.png")
            plot_performance_comparison(results, comparison_path)

    def _print_comparison(self, results: Dict):
        """Print comparison table of results."""
        if not results:
            return

        self.logger.info("\n" + "=" * 80)
        self.logger.info("EVALUATION RESULTS COMPARISON")
        self.logger.info("=" * 80)

        # Key metrics for comparison
        metrics = ['Success Rate', 'Denial Rate', 'Average Score', 'episode_reward', 'Total Profit']

        # Print header
        header = f"{'Agent':<20}"
        for metric in metrics:
            header += f"{metric:<15}"
        print(header)
        print("-" * len(header))

        # Print data
        for name, summary in results.items():
            row = f"{name:<20}"
            for metric in metrics:
                value = summary.get(metric, 0)
                row += f"{value:<15.3f}"
            print(row)

        self.logger.info("=" * 80)
