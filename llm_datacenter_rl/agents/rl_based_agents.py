import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from .base_agent import BaseAgent
from ..config.config import TrainingConfig


class DiscreteToBoxWrapper(gym.Wrapper):
    """Wrapper to convert discrete action space to box for continuous RL algorithms."""

    def __init__(self, env):
        super().__init__(env)
        self.n_actions = env.action_space.n
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def step(self, action):
        # Convert continuous action to discrete
        if isinstance(action, np.ndarray):
            action_value = action[0]
        else:
            action_value = action

        action_value = np.clip(action_value, -1.0, 1.0)
        discrete_action = int((action_value + 1) * self.n_actions / 2)
        discrete_action = np.clip(discrete_action, 0, self.n_actions - 1)

        return self.env.step(discrete_action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class SafeMonitor(Monitor):
    """Monitor wrapper with safe action handling."""

    def step(self, action):
        if hasattr(self.env, 'n_actions'):
            if isinstance(action, (int, float)):
                action = np.array([action], dtype=np.float32)
            elif isinstance(action, np.ndarray) and action.ndim == 0:
                action = np.array([action.item()], dtype=np.float32)
        return super().step(action)


class RLAgent(BaseAgent):
    """Unified RL agent wrapper for all supported algorithms."""

    def __init__(self, algorithm: str, config: TrainingConfig, enable_deny: bool = True):
        super().__init__(f"RL_{algorithm}", enable_deny)
        self.algorithm = algorithm
        self.config = config
        self.model = None
        self.env = None

    def create_env(self, env):
        """Wrap environment appropriately for the algorithm."""
        if self.algorithm in ['SAC', 'TD3']:
            wrapped_env = DiscreteToBoxWrapper(env)
            self.env = Monitor(wrapped_env)
        else:
            self.env = Monitor(env)
        return self.env

    def create_model(self, env=None):
        """Create the RL model."""
        if env is None:
            env = self.env

        self.logger.info(f"Creating {self.algorithm} model")

        if self.algorithm == "PPO":
            self.model = PPO(
                "MultiInputPolicy",
                env,
                learning_rate=self.config.learning_rate,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                n_epochs=self.config.n_epochs,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                clip_range=self.config.clip_range,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                max_grad_norm=self.config.max_grad_norm,
                policy_kwargs=self.config.policy_kwargs,
                verbose=1,
                device=self.config.device,
            )
        elif self.algorithm == "A2C":
            self.model = A2C(
                "MultiInputPolicy",
                env,
                learning_rate=self.config.learning_rate,
                n_steps=self.config.n_steps,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                max_grad_norm=self.config.max_grad_norm,
                policy_kwargs=self.config.policy_kwargs,
                verbose=1,
                device=self.config.device,
            )
        elif self.algorithm == "DQN":
            self.model = DQN(
                "MultiInputPolicy",
                env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                buffer_size=self.config.buffer_size,
                learning_starts=self.config.learning_starts,
                tau=self.config.tau,
                train_freq=self.config.train_freq,
                gradient_steps=self.config.gradient_steps,
                target_update_interval=self.config.target_update_interval,
                exploration_fraction=self.config.exploration_fraction,
                exploration_initial_eps=self.config.exploration_initial_eps,
                exploration_final_eps=self.config.exploration_final_eps,
                policy_kwargs=self.config.policy_kwargs,
                verbose=1,
                device=self.config.device,
            )
        elif self.algorithm == "SAC":
            self.model = SAC(
                "MultiInputPolicy",
                env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                buffer_size=self.config.buffer_size,
                learning_starts=self.config.learning_starts,
                tau=self.config.tau,
                train_freq=self.config.train_freq,
                gradient_steps=self.config.gradient_steps,
                ent_coef="auto",
                policy_kwargs=self.config.policy_kwargs,
                verbose=1,
                device=self.config.device,
            )
        elif self.algorithm == "TD3":
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=0.1 * np.ones(n_actions)
            )

            self.model = TD3(
                "MultiInputPolicy",
                env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                buffer_size=self.config.buffer_size,
                learning_starts=self.config.learning_starts,
                tau=self.config.tau,
                train_freq=self.config.train_freq,
                gradient_steps=self.config.gradient_steps,
                action_noise=action_noise,
                policy_kwargs=self.config.policy_kwargs,
                verbose=1,
                device=self.config.device,
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return self.model

    def predict(self, observation: Dict, deterministic: bool = True) -> int:
        """Make a prediction."""
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")

        action, _ = self.model.predict(observation, deterministic=deterministic)

        # For continuous algorithms, convert back to discrete
        if self.algorithm in ['SAC', 'TD3']:
            if isinstance(action, np.ndarray):
                action_value = action[0]
            else:
                action_value = action

            action_value = np.clip(action_value, -1.0, 1.0)
            n_actions = 7 if self.enable_deny else 6
            discrete_action = int((action_value + 1) * n_actions / 2)
            action = np.clip(discrete_action, 0, n_actions - 1)

        return int(action)

    def save(self, path: str):
        """Save the model."""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(path)
        self.logger.info(f"Model saved to {path}")

    def load(self, path: str, env=None):
        """Load a saved model."""
        if env is None:
            env = self.env

        if self.algorithm == "PPO":
            self.model = PPO.load(path, env=env)
        elif self.algorithm == "A2C":
            self.model = A2C.load(path, env=env)
        elif self.algorithm == "DQN":
            self.model = DQN.load(path, env=env)
        elif self.algorithm == "SAC":
            self.model = SAC.load(path, env=env)
        elif self.algorithm == "TD3":
            self.model = TD3.load(path, env=env)

        self.logger.info(f"Model loaded from {path}")


# Factory function for RL agents
def get_rl_agent(algorithm: str, config: TrainingConfig, enable_deny: bool = True) -> RLAgent:
    """Get an RL agent by algorithm name."""
    supported_algorithms = ['PPO', 'A2C', 'DQN', 'SAC', 'TD3']

    if algorithm not in supported_algorithms:
        raise ValueError(f"Unknown RL algorithm: {algorithm}. Available: {supported_algorithms}")

    return RLAgent(algorithm, config, enable_deny)
