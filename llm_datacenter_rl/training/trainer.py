import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

from ..config.config import ExperimentConfig
from ..environments.environment import LLMDataCenterEnv
from ..agents.rule_based_agents import get_baseline_agent
from ..agents.rl_based_agents import get_rl_agent
from ..utils.logger import get_logger
from ..utils.metrics import MetricsLogger
from ..utils.plotting import plot_training_curves


class DataCenterCallback(BaseCallback):
    """Custom callback for logging data center specific metrics."""

    def __init__(self, metrics_logger: MetricsLogger, verbose=0):
        super().__init__(verbose)
        self.metrics_logger = metrics_logger
        self.episode_count = 0

    def _on_step(self) -> bool:
        if self.locals.get("dones")[0]:
            info = self.locals.get("infos")[0]
            self.episode_count += 1

            # Log episode metrics
            self.metrics_logger.log_episode(
                episode=self.episode_count,
                reward=info['episode_reward'],
                success_rate=info.get('success_rate', 0),
                denial_rate=info.get('denial_rate', 0),
                avg_score=info.get('avg_score', 0),
                energy_cost=info['total_energy_cost'],
                profit=info.get('profit', 0)
            )

            if self.verbose > 0 and self.episode_count % 1 == 0:
                print(f"Episode {self.episode_count}:\n"
                      f"Reward={info['episode_reward']:.3f},\n"
                      f"Success Rate={info.get('success_rate', 0):.3f},\n"
                      f"Denial Rate={info.get('denial_rate', 0):.3f},\n"
                      f"Avg Score={info.get('avg_score', 0):.3f},\n"
                      f"Enery Cost={info.get('total_energy_cost', 0):.3f},\n"
                      f"Profit={info.get('profit', 0):.3f}\n")
        return True


class Trainer:
    """Unified trainer for RL agents."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.metrics_logger = MetricsLogger()

        # Create directories
        config.create_directories()
        paths = config.get_output_paths()
        self.models_dir = paths["models"]
        self.logs_dir = paths["logs"]
        self.plots_dir = paths["plots"]

    def train_agent(self, algorithm: str) -> Dict:
        """Train a single RL agent."""
        self.logger.info(f"Starting training for {algorithm}")
        start_time = time.time()

        # Create environment
        env = LLMDataCenterEnv(self.config.environment)

        # Create agent
        agent = get_rl_agent(algorithm, self.config.training, self.config.environment.enable_deny)
        wrapped_env = agent.create_env(env)
        agent.create_model()

        # Create evaluation environment
        eval_env = LLMDataCenterEnv(self.config.environment)
        eval_wrapped_env = agent.create_env(eval_env)

        # Initialize the metrics_logger
        self.metrics_logger = MetricsLogger()

        # Setup callbacks
        algorithm_dir = os.path.join(self.models_dir, algorithm)
        log_dir = os.path.join(self.logs_dir, algorithm)
        plot_dir = os.path.join(self.plots_dir, algorithm)
        os.makedirs(algorithm_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)

        callbacks = [
            DataCenterCallback(self.metrics_logger, verbose=1),
            CheckpointCallback(
                save_freq=self.config.training.save_freq,
                save_path=algorithm_dir,
                name_prefix=f"{algorithm}_checkpoint"
            ),
            EvalCallback(
                eval_wrapped_env,
                best_model_save_path=algorithm_dir,
                log_path=log_dir,
                eval_freq=self.config.training.eval_freq,
                n_eval_episodes=self.config.training.n_eval_episodes,
                deterministic=True,
                render=False
            )
        ]

        # Train
        agent.model.learn(
            total_timesteps=self.config.training.total_timesteps,
            callback=callbacks,
            progress_bar=True,
            log_interval=10000,
        )

        # Save final model
        final_model_path = os.path.join(
            algorithm_dir,
            f"{algorithm}_final_{self.config.training.total_timesteps}.zip"
        )
        agent.save(final_model_path)

        training_time = time.time() - start_time
        self.logger.info(f"{algorithm} training completed in {training_time:.2f}s")

        # Save and plot training curve
        training_curve_path = os.path.join(plot_dir, f"{algorithm}_training_curve.png")
        training_data_path = os.path.join(plot_dir, f"{algorithm}_training_data.csv")
        self.metrics_logger.get_dataframe().to_csv(training_data_path)
        plot_training_curves(self.metrics_logger, training_curve_path)

        # Cleanup
        wrapped_env.close()
        eval_wrapped_env.close()

        return {
            "algorithm": algorithm,
            "model_path": final_model_path,
            "training_time": training_time,
            "total_timesteps": self.config.training.total_timesteps
        }

    def train_all_agents(self) -> Dict[str, Dict]:
        """Train all specified RL agents."""
        results = {}
        total_start_time = time.time()

        self.logger.info(f"Training {len(self.config.algorithms)} algorithms: {self.config.algorithms}")

        for algorithm in self.config.algorithms:
            try:
                result = self.train_agent(algorithm)
                results[algorithm] = result
            except Exception as e:
                self.logger.error(f"Failed to train {algorithm}: {e}")
                results[algorithm] = {"error": str(e)}

        total_time = time.time() - total_start_time
        self.logger.info(f"Total training time: {total_time:.2f}s ({total_time / 60:.1f} minutes)")

        # Save training summary
        self._save_training_summary(results, total_time)

        return results

    def _save_training_summary(self, results: Dict, total_time: float):
        """Save training summary to file."""
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(self.config.get_output_paths()["base"], f"training_summary.txt")

        with open(summary_path, 'w') as f:
            f.write("LLM Data Center RL Training Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Experiment: {self.config.experiment_name}\n")
            f.write(f"Total training time: {total_time:.2f}s ({total_time / 60:.1f} minutes)\n\n")

            f.write("Configuration:\n")
            f.write(f"  Algorithms: {self.config.algorithms}\n")
            f.write(f"  Total timesteps: {self.config.training.total_timesteps}\n")
            f.write(f"  Learning rate: {self.config.training.learning_rate}\n")
            f.write(
                f"  Environment mode: {'Realistic' if self.config.environment.is_realistic_mode else 'Synthetic'}\n")
            f.write(f"  Simulation duration: {self.config.environment.simulation_duration_hours} hours\n")
            f.write(f"  Enable deny: {self.config.environment.enable_deny}\n\n")

            f.write("Results:\n")
            for algorithm, result in results.items():
                if "error" in result:
                    f.write(f"  {algorithm}: FAILED - {result['error']}\n")
                else:
                    f.write(f"  {algorithm}: SUCCESS\n")
                    f.write(f"    Model: {result['model_path']}\n")
                    f.write(f"    Training time: {result['training_time']:.2f}s\n")
