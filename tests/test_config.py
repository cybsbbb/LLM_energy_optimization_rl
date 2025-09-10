import pytest
import tempfile
import os
from llm_datacenter_rl.config.config import (
    EnvironmentConfig, TrainingConfig, ExperimentConfig, get_config
)


class TestConfigurations:
    """Test cases for configuration classes."""

    def test_environment_config_defaults(self):
        """Test EnvironmentConfig default values."""
        config = EnvironmentConfig()

        assert config.server_num == 200
        assert config.enable_deny == False
        assert config.simulation_duration_hours == 24 * 7
        assert not config.is_realistic_mode

    def test_environment_config_realistic_mode(self):
        """Test EnvironmentConfig realistic mode detection."""
        config = EnvironmentConfig(
            invocations_file="test.csv",
            energy_price_file="test2.csv"
        )

        assert config.is_realistic_mode

    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()

        assert config.total_timesteps == 1000000
        assert config.learning_rate == 3e-4
        assert config.gamma == 0.99
        assert config.n_steps == 2048

    def test_experiment_config_paths(self):
        """Test ExperimentConfig path generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                experiment_name="test_exp",
                output_dir=temp_dir
            )

            paths = config.get_output_paths()

            assert "base" in paths
            assert "models" in paths
            assert "logs" in paths
            assert "plots" in paths

            # Test directory creation
            config.create_directories()
            assert os.path.exists(paths["base"])
            assert os.path.exists(paths["models"])

    def test_get_config_predefined(self):
        """Test getting predefined configurations."""
        config_names = ["synthetic_mode", "realistic_mode"]

        for name in config_names:
            config = get_config(name)
            assert isinstance(config, ExperimentConfig)
            assert config.experiment_name == name

    def test_get_config_invalid(self):
        """Test getting invalid configuration name."""
        with pytest.raises(ValueError):
            get_config("invalid_config")
