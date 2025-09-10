import numpy as np
from llm_datacenter_rl.environments.environment import LLMDataCenterEnv


class TestLLMDataCenterEnv:
    """Test cases for the LLM Data Center environment."""

    def test_synthetic_environment_creation(self, synthetic_config):
        """Test creating a synthetic environment."""
        env = LLMDataCenterEnv(synthetic_config)

        assert env.config == synthetic_config
        assert env.action_space.n == 7 if synthetic_config.enable_deny else 6
        assert not env.config.is_realistic_mode

        env.close()

    def test_realistic_environment_creation(self, realistic_config):
        """Test creating a realistic environment."""
        env = LLMDataCenterEnv(realistic_config)

        assert env.config == realistic_config
        assert env.config.is_realistic_mode
        assert hasattr(env.data_provider, 'energy_timestamps_ms')

        env.close()

    def test_environment_reset(self, realistic_config):
        """Test environment reset functionality."""
        env = LLMDataCenterEnv(realistic_config)

        obs, info = env.reset()

        # Check observation structure
        assert isinstance(obs, dict)
        expected_keys = ['processing_ratio', 'waiting_ratio', 'energy_price',
                         'time_of_day', 'recent_latency', 'recent_score']
        for key in expected_keys:
            assert key in obs
            assert obs[key].shape == (1,)

        # Check initial state
        assert env.processing_server_num == 0
        assert len(env.waiting_queue) == 0
        assert env.episode_reward == 0

        env.close()

    def test_environment_step(self, realistic_config):
        """Test environment step functionality."""
        env = LLMDataCenterEnv(realistic_config)
        obs, _ = env.reset()

        # Take a few steps
        for i in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # Check return types
            assert isinstance(obs, dict)
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)

            # Check info structure
            expected_info_keys = ['success_requests', 'failed_requests',
                                  'denied_requests', 'timeout_requests']
            for key in expected_info_keys:
                assert key in info

            if terminated or truncated:
                break

        env.close()

    def test_action_validation(self, realistic_config):
        """Test action validation and conversion."""
        env = LLMDataCenterEnv(realistic_config)
        obs, _ = env.reset()

        # Test different action formats
        test_actions = [
            0,  # Integer
            np.array([1]),  # NumPy array
            np.array(2),  # 0-dim NumPy array
            (3,),  # Tuple
        ]

        for action in test_actions:
            obs, reward, terminated, truncated, info = env.step(action)
            assert not (terminated or truncated)  # Should not fail

        # Test out-of-bounds action
        obs, reward, terminated, truncated, info = env.step(999)
        assert not (terminated or truncated)  # Should clip and continue

        env.close()

    def test_deny_action(self, realistic_config):
        """Test deny action functionality."""
        env = LLMDataCenterEnv(realistic_config)
        obs, _ = env.reset()

        # Take deny action multiple times
        deny_action = 6 if realistic_config.enable_deny else 0
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step(deny_action)
            if terminated or truncated:
                break

        # Should have some denied requests
        if realistic_config.enable_deny:
            assert info['denied_requests'] >= 0

        env.close()

    def test_environment_summary(self, realistic_config):
        """Test final summary generation."""
        env = LLMDataCenterEnv(realistic_config)
        obs, _ = env.reset()

        # Run a short episode
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        summary = env.get_final_summary()
        print(summary)

        # Check summary structure
        expected_keys = ['Total Success Requests', 'Total Failed Requests',
                         'Success Rate', 'Average Score', 'Total Energy (J)',
                         'Total Profit', 'Action Statistics']
        for key in expected_keys:
            assert key in summary

        env.close()
