import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


root_dir: str = str(Path(__file__).resolve().parent.parent.parent)

@dataclass
class EnvironmentConfig:
    """Configuration for the LLM Data Center environment."""

    # Basic environment settings
    server_num: int = 200
    max_wait_time: int = 10000  # milliseconds
    pue: float = 1.3
    enable_deny: bool = False

    # Time settings
    step_time_ms: int = 10000  # 10 seconds per step
    simulation_duration_hours: int = 24 * 7
    time_acceleration_factor: float = 1.0

    # Request generation (for synthetic mode)
    bernoulli_prob: float = 0.2
    time_interval: int = 50  # milliseconds

    # Synthetic mode setting
    use_sample_energy_price: Optional[bool] = None
    sample_energy_price_file: Optional[str] = None

    # Realistic mode settings
    invocations_file: Optional[str] = None
    energy_price_file: Optional[str] = None
    simulation_start_time: Optional[str] = None
    request_scaling_factor: float = 1.0

    # Reward parameters
    success_reward: float = 0.5
    deny_penalty: float = -0.1
    timeout_penalty: float = -0.0
    score_weight: float = 2.0
    latency_penalty_weight: float = 0.0
    energy_penalty_weight: float = 1.0

    @property
    def is_realistic_mode(self) -> bool:
        """Check if realistic mode is enabled."""
        return (self.invocations_file is not None and
                self.energy_price_file is not None)


@dataclass
class TrainingConfig:
    """Configuration for RL agent training."""

    # Training parameters
    total_timesteps: int = 1000000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    device: str = "auto"

    # Policy network parameters
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)

    # On-policy specific (PPO, A2C)
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gae_lambda: float = 0.95
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    clip_range: float = 0.2  # PPO only

    # Off-policy specific (DQN, SAC, TD3)
    buffer_size: int = 1000000
    learning_starts: int = 100
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1

    # DQN specific
    target_update_interval: int = 10000
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05

    # Training monitoring
    eval_freq: int = 100000
    n_eval_episodes: int = 5
    save_freq: int = 100000


@dataclass
class EvaluationConfig:
    """Configuration for agent evaluation."""

    n_eval_episodes: int = 5
    render: bool = False
    deterministic: bool = True
    save_plots: bool = True
    plot_timeline: bool = True
    plot_comparison: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    # Sub-configurations
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Experiment metadata
    experiment_name: Optional[str] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Paths
    data_dir: str = "data"
    output_dir: str = "results"
    models_dir: str = "models"
    logs_dir: str = "logs"
    plots_dir: str = "plots"

    # Algorithms to train/evaluate
    algorithms: List[str] = field(default_factory=lambda: ['PPO', 'A2C', 'DQN', 'SAC', 'TD3'])
    baseline_agents: List[str] = field(default_factory=lambda: [
        "all_fullkv", "all_snapkv_64", "rule_based_price", "random"
    ])
    # debug use
    # baseline_agents: List[str] = field(default_factory=lambda: [])

    def get_output_paths(self) -> Dict[str, str]:
        """Get all output paths for the experiment."""
        base_path = os.path.join(root_dir, self.output_dir, self.experiment_name or "default")
        return {
            "base": base_path,
            "models": os.path.join(base_path, self.models_dir),
            "logs": os.path.join(base_path, self.logs_dir),
            "plots": os.path.join(base_path, self.plots_dir),
        }

    def create_directories(self):
        """Create all necessary directories."""
        paths = self.get_output_paths()
        for path in paths.values():
            os.makedirs(path, exist_ok=True)


# Default configurations for different scenarios
DEFAULT_CONFIGS = {
    "synthetic_mode": ExperimentConfig(
        training=TrainingConfig(total_timesteps=1000000, eval_freq=100000),
        environment=EnvironmentConfig(
            simulation_duration_hours=24,  # 1 days
            server_num=200,
            use_sample_energy_price=True,
            sample_energy_price_file=os.path.join(root_dir, "data/energy_price/processed/CAISO_20250421_20250422.csv"),
            simulation_start_time="2024-05-10 00:00:00"
        ),
        experiment_name="synthetic_mode"
    ),

    "realistic_mode": ExperimentConfig(
        training=TrainingConfig(total_timesteps=12000000, eval_freq=200000),
        environment=EnvironmentConfig(
            simulation_duration_hours=24 * 7,  # 1 week
            server_num=200,
            invocations_file=os.path.join(root_dir, "data/LLM_inference_trace/processed/2024/invocations_ten_second_Coding.csv"),
            energy_price_file=os.path.join(root_dir, "data/energy_price/processed/CAISO_20240509_20240521.csv"),
            simulation_start_time="2024-05-10 00:00:00"
        ),
        experiment_name="realistic_mode"
    ),
}


def get_config(config_name: str = "realistic_mode") -> ExperimentConfig:
    """Get a predefined configuration."""
    if config_name not in DEFAULT_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(DEFAULT_CONFIGS.keys())}")
    return DEFAULT_CONFIGS[config_name]
