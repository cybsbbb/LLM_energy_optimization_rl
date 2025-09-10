import os
import pytest
import shutil
from pathlib import Path

from llm_datacenter_rl.config.config import EnvironmentConfig, TrainingConfig, ExperimentConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir: str = os.path.join(Path(__file__).resolve().parent, "temp")
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_data_dir():
    """Get sample data files for testing."""
    data_dir: str = os.path.join(Path(__file__).resolve().parent.parent, "data")
    invocations_file: str = os.path.join(data_dir, "LLM_inference_trace/processed/2024/invocations_ten_second_Coding.csv")
    energy_price_file: str = os.path.join(data_dir, "energy_price/processed/CAISO_20240509_20240521.csv")
    # gov_report_dir: str = os.path.join(data_dir, "kv_compression_simulate_data/gov_report")
    sample_energy_price_file: str = os.path.join(data_dir, "energy_price/processed/CAISO_20250421_20250422.csv")

    return {
        'data_dir': data_dir,
        'invocations_file': invocations_file,
        'energy_file': energy_price_file,
        # 'gov_report_dir': gov_report_dir,
        'sample_energy_price_file': sample_energy_price_file
    }


@pytest.fixture
def synthetic_config(sample_data_dir):
    """Create a synthetic environment configuration for testing."""
    return EnvironmentConfig(
        server_num=200,
        simulation_duration_hours=1,
        use_sample_energy_price=True,
        sample_energy_price_file=sample_data_dir['sample_energy_price_file'],
        enable_deny=True,
        bernoulli_prob=0.2,
        time_interval=10
    )


@pytest.fixture
def realistic_config(sample_data_dir):
    """Create a realistic environment configuration for testing."""
    return EnvironmentConfig(
        server_num=200,
        simulation_duration_hours=1,
        enable_deny=True,
        invocations_file=sample_data_dir['invocations_file'],
        energy_price_file=sample_data_dir['energy_file'],
        simulation_start_time="2024-05-10 00:00:00",
        time_interval=10
    )


@pytest.fixture
def training_config():
    """Create a training configuration for testing."""
    return TrainingConfig(
        total_timesteps=1000,
        eval_freq=500,
        n_eval_episodes=1,
        save_freq=500
    )


@pytest.fixture
def experiment_config(synthetic_config, training_config, temp_dir):
    """Create a complete experiment configuration for testing."""
    return ExperimentConfig(
        environment=synthetic_config,
        training=training_config,
        algorithms=["PPO"],
        baseline_agents=["all_fullkv"],
        experiment_name="test_experiment",
        output_dir=temp_dir
    )