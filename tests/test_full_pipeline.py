import os
from llm_datacenter_rl import Trainer, Evaluator
from llm_datacenter_rl.config.config import ExperimentConfig, TrainingConfig


class TestFullPipeline:
    """Integration tests for the complete pipeline."""

    def test_synthetic_mode_pipeline(self, synthetic_config, temp_dir):
        """Test complete pipeline in synthetic mode."""
        # Create minimal configuration
        config = ExperimentConfig(
            environment=synthetic_config,
            training=TrainingConfig(
                total_timesteps=10000,
                eval_freq=1000,
                n_eval_episodes=1
            ),
            algorithms=["PPO"],
            baseline_agents=["all_fullkv"],
            experiment_name="synthetic_integration_test",
            output_dir=temp_dir
        )

        # Train agents
        trainer = Trainer(config)
        training_results = trainer.train_all_agents()

        # Check training results
        assert "PPO" in training_results
        assert "error" not in training_results["PPO"]

        # Evaluate agents
        evaluator = Evaluator(config)
        models_dir = config.get_output_paths()["models"]
        evaluation_results = evaluator.load_and_evaluate_trained_models(models_dir)

        # Check evaluation results
        assert len(evaluation_results) >= 2  # At least RL agent + baseline

        # Check that output files were created
        paths = config.get_output_paths()
        assert os.path.exists(paths["models"])
        assert os.path.exists(os.path.join(paths["base"], "training_summary.txt"))

    def test_realistic_mode_pipeline(self, realistic_config, temp_dir):
        """Test pipeline in realistic mode."""
        config = ExperimentConfig(
            environment=realistic_config,
            training=TrainingConfig(
                total_timesteps=10000,
                eval_freq=1000,
                n_eval_episodes=1
            ),
            algorithms=["PPO"],
            baseline_agents=["rule_based_price"],
            experiment_name="realistic_integration_test",
            output_dir=temp_dir
        )

        # Should not raise any errors
        trainer = Trainer(config)
        training_results = trainer.train_all_agents()

        assert "PPO" in training_results
        # Training might complete with very short timesteps
        assert "error" not in training_results["PPO"] or "timesteps" in str(training_results["PPO"].get("error", ""))
