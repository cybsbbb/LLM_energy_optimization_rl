import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import os

from llm_datacenter_env import LLMDataCenterEnv


class DataCenterCallback(BaseCallback):
    """Custom callback for logging data center specific metrics."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_success_rates = []
        self.episode_avg_scores = []
        self.episode_energy_costs = []
        self.episode_denial_rates = []
        
    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get("dones")[0]:
            info = self.locals.get("infos")[0]
            
            # Calculate rates
            total_requests = info['success_requests'] + info['failed_requests'] + info['denied_requests']
            success_rate = info['success_requests'] / total_requests if total_requests > 0 else 0
            denial_rate = info['denied_requests'] / total_requests if total_requests > 0 else 0
            
            # Log metrics
            self.episode_rewards.append(info['episode_reward'])
            self.episode_success_rates.append(success_rate)
            # self.episode_avg_scores.append(info.get('avg_score', 0))
            self.episode_energy_costs.append(info['total_energy_cost'])
            self.episode_denial_rates.append(denial_rate)
            
            if self.verbose > 0:
                print(f"\nEpisode {len(self.episode_rewards)}:")
                print(f"  Reward: {info['episode_reward']:.3f}")
                print(f"  Success Rate: {success_rate:.3f}")
                print(f"  Denial Rate: {denial_rate:.3f}")
                # print(f"  Avg Score: {info.get('avg_score', 0):.3f}")
                print(f"  Energy Cost: {info['total_energy_cost']:.2f}")
                
        return True


class LLMDataCenterAgent:
    """RL Agent for LLM Data Center KV-cache optimization."""
    
    def __init__(self, 
                 algorithm: str = "PPO",
                 learning_rate: float = 3e-4,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 device: str = "auto"):
        
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.device = device
        
        self.model = None
        self.env = None
        
    def create_env(self, **env_kwargs):
        """Create the environment."""
        self.env = Monitor(LLMDataCenterEnv(**env_kwargs))
        return self.env
    
    def create_model(self, env=None):
        """Create the RL model."""
        if env is None:
            env = self.env
            
        if self.algorithm == "PPO":
            self.model = PPO(
                "MultiInputPolicy",
                env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                verbose=1,
                device=self.device,
                tensorboard_log="./datacenter_tensorboard/"
            )
        elif self.algorithm == "A2C":
            self.model = A2C(
                "MultiInputPolicy",
                env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                gamma=self.gamma,
                verbose=1,
                device=self.device,
                tensorboard_log="./datacenter_tensorboard/"
            )
        elif self.algorithm == "DQN":
            self.model = DQN(
                "MultiInputPolicy",
                env,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                gamma=self.gamma,
                verbose=1,
                device=self.device,
                tensorboard_log="./datacenter_tensorboard/"
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
            
        return self.model
    
    def train(self, 
              total_timesteps: int = 100000,
              eval_freq: int = 10000,
              n_eval_episodes: int = 5,
              save_freq: int = 10000,
              save_path: str = "./datacenter_models/",
              log_path: str = "./datacenter_logs/"):
        """Train the agent."""
        
        # Create directories
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        
        # Create evaluation environment
        eval_env = Monitor(LLMDataCenterEnv())
        
        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=save_path,
            name_prefix=f"datacenter_{self.algorithm}"
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            log_path=log_path,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False
        )
        
        datacenter_callback = DataCenterCallback(verbose=1)
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback, datacenter_callback],
            progress_bar=True
        )
        
        # Plot training curves
        self.plot_training_curves(datacenter_callback)
        
        return datacenter_callback
    
    def plot_training_curves(self, callback: DataCenterCallback):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(callback.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Success rates
        axes[0, 1].plot(callback.episode_success_rates)
        axes[0, 1].set_title('Success Rates')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Success Rate')
        
        # Denial rates
        axes[0, 2].plot(callback.episode_denial_rates)
        axes[0, 2].set_title('Denial Rates')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Denial Rate')
        
        # Average scores
        axes[1, 0].plot(callback.episode_avg_scores)
        axes[1, 0].set_title('Average Scores')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Score')
        
        # Energy costs
        axes[1, 1].plot(callback.episode_energy_costs)
        axes[1, 1].set_title('Total Energy Costs')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Energy Cost')
        
        # Remove empty subplot
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.show()
    
    def save(self, path: str):
        """Save the model."""
        self.model.save(path)
        
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
            
    def predict(self, observation: Dict, deterministic: bool = True):
        """Make a prediction."""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action
    
    def evaluate(self, n_eval_episodes: int = 10, render: bool = False):
        """Evaluate the trained model."""
        eval_env = LLMDataCenterEnv(render_mode="human" if render else None)
        
        mean_reward, std_reward = evaluate_policy(
            self.model,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            render=render,
            return_episode_rewards=False
        )
        
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        
        # Run one detailed evaluation episode
        print("\nDetailed evaluation:")
        obs, _ = eval_env.reset()
        done = False
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            
        # Print final statistics
        total_requests = info['success_requests'] + info['failed_requests'] + info['denied_requests']
        success_rate = info['success_requests'] / total_requests if total_requests > 0 else 0
        denial_rate = info['denied_requests'] / total_requests if total_requests > 0 else 0
        
        print(f"\nFinal Statistics:")
        print(f"  Total Success Requests: {info['success_requests']}")
        print(f"  Total Failed Requests: {info['failed_requests']}")
        print(f"  Total Denied Requests: {info['denied_requests']}")
        print(f"  Success Rate: {success_rate:.3f}")
        print(f"  Denial Rate: {denial_rate:.3f}")
        # print(f"  Average Score: {info.get('avg_score', 0):.3f}")
        # print(f"  Average Latency: {info.get('avg_latency', 0)/1000:.3f} s")
        print(f"  Total Energy: {info['total_energy']} J")
        print(f"  Total Energy Cost: {info['total_energy_cost'] * eval_env.pue / eval_env.j2mwh:.4f}")
        print(f"  Total Profit: ${info['success_requests'] / 1000 * 0.02:.4f}")
        
        eval_env.close()
        return mean_reward, std_reward


# Wrapper function for compatibility with the original simulator
def create_rl_agent_function(agent: LLMDataCenterAgent):
    """Create a function compatible with the extended simulator."""
    def agent_function(state: Dict) -> str:
        # Convert state to observation format
        obs = {
            'processing_ratio': np.array([state['processing_server_num'] / state['total_server_num']], dtype=np.float32),
            'waiting_ratio': np.array([min(state['waiting_request_num'] / state['total_server_num'], 1.0)], dtype=np.float32),
            'energy_price': np.array([state['energy_price'] / 100.0], dtype=np.float32),
            'time_of_day': np.array([0.5], dtype=np.float32),  # Default since not provided
            'recent_latency': np.array([0.0], dtype=np.float32),  # Default
            'recent_score': np.array([0.5], dtype=np.float32),  # Default
        }
        
        # Get action from model
        action = agent.predict(obs, deterministic=True)
        
        # Convert action to KV size string or deny
        action_to_string = {
            0: 'fullkv',
            1: 'snapkv_64',
            2: 'snapkv_128',
            3: 'snapkv_256',
            4: 'snapkv_512',
            5: 'snapkv_1024',
            6: 'deny'
        }
        
        return action_to_string[int(action)]
    
    return agent_function


if __name__ == "__main__":
    # Example usage
    print("Creating LLM Data Center RL Agent...")
    
    # Create agent
    agent = LLMDataCenterAgent(
        algorithm="PPO",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99
    )
    
    # Create environment
    env = agent.create_env(
        total_time=3600 * 1000,  # 1 hour for faster training
        server_num=200,
        bernoulli_prob=0.2
    )
    
    # Create model
    agent.create_model()
    
    # Train (reduce timesteps for demo)
    print("\nStarting training...")
    callback = agent.train(
        total_timesteps=50000,
        eval_freq=10000,
        n_eval_episodes=3,
        save_freq=10000
    )
    
    # Evaluate
    print("\nEvaluating trained agent...")
    agent.evaluate(n_eval_episodes=5, render=False)
    
    # Save model
    agent.save("datacenter_ppo_final")
    
    # Example of using with simulator
    print("\nCreating agent function for simulator...")
    rl_agent_function = create_rl_agent_function(agent)
    
    # Can now use with extended simulator:
    # from llm_data_center_simulator import LLMDataCenterSimulator
    # simulator = LLMDataCenterSimulator(rl_agent_function)
    # simulator.run_simulator()