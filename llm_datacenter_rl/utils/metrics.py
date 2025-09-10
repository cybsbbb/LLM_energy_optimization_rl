# llm_datacenter_rl/utils/metrics.py

from typing import Dict, List
import numpy as np
import pandas as pd
from datetime import datetime


class MetricsLogger:
    """Logger for tracking training and evaluation metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all logged metrics."""
        self.episodes = []
        self.rewards = []
        self.success_rates = []
        self.denial_rates = []
        self.avg_scores = []
        self.energy_costs = []
        self.profits = []
        self.timestamps = []

    def log_episode(self, episode: int, reward: float, success_rate: float,
                    denial_rate: float, avg_score: float, energy_cost: float, profit: float):
        """Log metrics for an episode."""
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.success_rates.append(success_rate)
        self.denial_rates.append(denial_rate)
        self.avg_scores.append(avg_score)
        self.energy_costs.append(energy_cost)
        self.profits.append(profit)
        self.timestamps.append(datetime.now())

    def get_dataframe(self) -> pd.DataFrame:
        """Get all metrics as a pandas DataFrame."""
        return pd.DataFrame({
            'episode': self.episodes,
            'reward': self.rewards,
            'success_rate': self.success_rates,
            'denial_rate': self.denial_rates,
            'avg_score': self.avg_scores,
            'energy_cost': self.energy_costs,
            'profit': self.profits,
            'timestamp': self.timestamps
        })

    def get_summary_stats(self) -> Dict:
        """Get summary statistics."""
        if not self.rewards:
            return {}

        return {
            'total_episodes': len(self.rewards),
            'mean_reward': np.mean(self.rewards),
            'std_reward': np.std(self.rewards),
            'mean_success_rate': np.mean(self.success_rates),
            'mean_denial_rate': np.mean(self.denial_rates),
            'mean_avg_score': np.mean(self.avg_scores),
            'total_profit': np.sum(self.profits)
        }
