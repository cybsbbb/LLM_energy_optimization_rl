import pytest
import numpy as np
from llm_datacenter_rl.agents.rl_based_agents import get_rl_agent
from llm_datacenter_rl.agents.rule_based_agents import get_baseline_agent, AllFullKVAgent, RuleBasedPriceAgent, RandomAgent
from llm_datacenter_rl.config.config import TrainingConfig


class TestBaselineAgents:
    """Test cases for baseline agents."""

    def test_get_baseline_agent(self):
        """Test baseline agent factory function."""
        agent_names = ['all_fullkv', 'all_snapkv_64', 'rule_based_price', 'smart_deny', 'random']

        for name in agent_names:
            agent = get_baseline_agent(name)
            assert agent.name is not None
            assert hasattr(agent, 'predict')

    def test_invalid_baseline_agent(self):
        """Test invalid baseline agent name."""
        with pytest.raises(ValueError):
            get_baseline_agent('invalid_agent')

    def test_all_fullkv_agent(self):
        """Test AllFullKVAgent behavior."""
        agent = AllFullKVAgent()

        # Create dummy observation
        obs = {
            'processing_ratio': np.array([0.5]),
            'waiting_ratio': np.array([0.3]),
            'energy_price': np.array([0.2]),
            'time_of_day': np.array([0.5]),
            'recent_latency': np.array([0.1]),
            'recent_score': np.array([0.9])
        }

        action = agent.predict(obs)
        assert action == 0  # Should always choose fullkv

    def test_rule_based_price_agent(self):
        """Test RuleBasedPriceAgent behavior."""
        agent = RuleBasedPriceAgent()

        # Test with different energy prices
        base_obs = {
            'processing_ratio': np.array([0.5]),
            'waiting_ratio': np.array([0.3]),
            'time_of_day': np.array([0.5]),
            'recent_latency': np.array([0.1]),
            'recent_score': np.array([0.9])
        }

        # Low energy price
        obs_low = base_obs.copy()
        obs_low['energy_price'] = np.array([-0.8])  # Very low normalized price
        action_low = agent.predict(obs_low)

        # High energy price
        obs_high = base_obs.copy()
        obs_high['energy_price'] = np.array([0.8])  # Very high normalized price
        action_high = agent.predict(obs_high)

        # Should choose different actions for different prices
        assert action_low != action_high

    def test_random_agent(self):
        """Test RandomAgent behavior."""
        agent = RandomAgent(enable_deny=True)

        obs = {
            'processing_ratio': np.array([0.5]),
            'waiting_ratio': np.array([0.3]),
            'energy_price': np.array([0.2]),
            'time_of_day': np.array([0.5]),
            'recent_latency': np.array([0.1]),
            'recent_score': np.array([0.9])
        }

        # Test multiple predictions
        actions = [agent.predict(obs) for _ in range(10)]

        # Should be valid actions
        for action in actions:
            assert 0 <= action < agent.n_actions

        # Should have some variability (not all the same)
        assert len(set(actions)) > 1


class TestRLAgents:
    """Test cases for RL agents."""

    def test_get_rl_agent(self):
        """Test RL agent factory function."""
        config = TrainingConfig(total_timesteps=1000)
        algorithms = ['PPO', 'A2C', 'DQN', 'SAC', 'TD3']

        for algorithm in algorithms:
            agent = get_rl_agent(algorithm, config)
            assert agent.algorithm == algorithm
            assert agent.config == config

    def test_invalid_rl_agent(self):
        """Test invalid RL algorithm name."""
        config = TrainingConfig()

        with pytest.raises(ValueError):
            get_rl_agent('INVALID_ALGO', config)
