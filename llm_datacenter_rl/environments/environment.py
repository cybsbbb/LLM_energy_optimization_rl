import random
import numpy as np
import pandas as pd
import collections
import gymnasium as gym
from gymnasium import spaces
from heapq import heappush, heappop
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional


from .data_provider import SyntheticDataProvider, RealisticDataProvider
from ..config.config import EnvironmentConfig
from ..datasets.gov_report_sim_dataset import GovReportSimDataset
from ..utils.logger import get_logger


class LLMDataCenterEnv(gym.Env):
    """
    Unified LLM Data Center environment supporting both synthetic and realistic modes.
    """

    metadata = {'render_modes': ['human', 'ansi']}

    def __init__(self, config: EnvironmentConfig):
        super().__init__()

        self.config = config
        self.logger = get_logger(__name__)

        # Initialize data provider (environment data: request_rate, energy price) based on mode
        if config.is_realistic_mode:
            self.data_provider = RealisticDataProvider(config)
        else:
            self.data_provider = SyntheticDataProvider(config)

        # Initialize dataset for LLM performance simulation
        self.dataset = GovReportSimDataset()

        # Constants
        self.j2mwh = 3_600_000_000

        # Action space: 6 KV cache configurations + optional deny action
        self.n_actions = 7 if config.enable_deny else 6
        self.action_space = spaces.Discrete(self.n_actions)

        # Build action mapping
        self.action_to_kv_size = {
            0: 'fullkv',
            1: 'snapkv_64',
            2: 'snapkv_128',
            3: 'snapkv_256',
            4: 'snapkv_512',
            5: 'snapkv_1024',
        }
        if config.enable_deny:
            self.action_to_kv_size[6] = 'deny'

        # Observation space
        self.observation_space = self._create_observation_space()

        # Time related variables
        # self.current_real_time = None
        # self.sim_start_real_time = None
        # self.sim_end_real_time = None
        # self.current_sim_time = None

        # Initialize state
        self.reset()

    def _create_observation_space(self) -> spaces.Dict:
        """Create observation space based on environment mode."""
        obs_space = {
            'processing_ratio': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'waiting_ratio': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'energy_price': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'time_of_day': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'recent_latency': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'recent_score': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        }

        if self.config.is_realistic_mode:
            obs_space.update({
                'energy_price_trend': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                'day_of_week': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'request_rate': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'request_rate_trend': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            })

        return spaces.Dict(obs_space)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Initialize time
        if self.config.is_realistic_mode and self.config.simulation_start_time:
            self.current_real_time = pd.to_datetime(self.config.simulation_start_time)
            self.sim_start_real_time = self.current_real_time
            self.sim_end_real_time = (
                self.current_real_time +
                timedelta(hours=self.config.simulation_duration_hours)
            )
        else:
            if self.config.simulation_start_time:
                self.current_real_time = pd.to_datetime(self.config.simulation_start_time)
            else:
                self.current_real_time = datetime.now()
            self.sim_start_real_time = self.current_real_time
            self.sim_end_real_time = (
                self.current_real_time +
                timedelta(hours=self.config.simulation_duration_hours)
            )

        self.current_sim_time = 0  # Simulation time in milliseconds

        # Reset simulation state
        self.event_queue = []
        self.processing_server_num = 0
        self.waiting_queue = []

        # Reset metrics
        self._reset_metrics()

        # Initialize performance tracking
        self.recent_latencies = collections.deque()
        self.recent_scores = collections.deque()

        # Episode tracking
        self.episode_reward = 0
        self.episode_length = 0

        # Action statistics
        self.action_counts = {i: 0 for i in range(self.n_actions)}

        # Step tracking
        self.completed_requests_rewards = []

        return self._get_observation(), {}

    def _reset_metrics(self):
        """Reset all metrics."""
        self.success_request_num = 0
        self.failed_request_num = 0
        self.denied_request_num = 0
        self.timeout_request_num = 0

        self.total_latency = 0.0
        self.total_score = 0.0
        self.latency_count = 0
        self.score_count = 0

        self.tot_energy = 0
        self.tot_energy_cost = 0

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.episode_length += 1

        # Safely convert and validate action
        action = self._safe_action_conversion(action)
        self.action_counts[action] += 1

        # Initialize step tracking
        step_start_time = self.current_sim_time
        step_end_time = step_start_time + self.config.step_time_ms
        requests_processed = 0
        self.completed_requests_rewards = []

        # Process events for this step
        while self.current_sim_time < step_end_time:
            # Update real time
            self.current_real_time = self._get_current_real_time()

            # Check if simulation ended
            if self.current_real_time >= self.sim_end_real_time:
                break

            # Generate new requests
            if self.data_provider.should_generate_request(self.current_real_time):
                self._process_new_request(action)
                requests_processed += 1

            # Process existing events
            self._process_events()

            # Advance simulation time
            self.current_sim_time += self.config.time_interval  # time increments

        # Update the current_real_time to avoid 1 extra step
        self.current_real_time = self._get_current_real_time()

        # Calculate step reward
        if self.completed_requests_rewards:
            step_reward = np.mean(self.completed_requests_rewards)
        else:
            step_reward = 0.0

        self.episode_reward += step_reward

        # Check if episode is done
        terminated = self.current_real_time >= self.sim_end_real_time
        truncated = False

        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        info['requests_processed'] = requests_processed
        info['step_duration_ms'] = self.current_sim_time - step_start_time

        return observation, step_reward, terminated, truncated, info

    def _safe_action_conversion(self, action) -> int:
        """Safely convert action to integer and ensure it's within bounds."""
        if isinstance(action, np.ndarray):
            action = action.item() if action.ndim == 0 else action[0]
        elif isinstance(action, tuple):
            action = action[0]

        action = int(action)
        return np.clip(action, 0, self.n_actions - 1)

    def _get_current_real_time(self) -> datetime:
        """Get current real time based on simulation time."""
        elapsed_real_ms = self.current_sim_time / self.config.time_acceleration_factor
        return self.sim_start_real_time + timedelta(milliseconds=elapsed_real_ms)

    def _process_new_request(self, action: int):
        """Process a new request with the given action."""
        action_type = self.action_to_kv_size[action]

        if action_type == 'deny':
            self.denied_request_num += 1
            self.completed_requests_rewards.append(self.config.deny_penalty)
        else:
            # Create new request
            new_request = self.dataset.get_random_item(action_type)
            new_request["energy_price"] = self.data_provider.get_energy_price(self.current_real_time)
            new_request["arrival_time"] = self.current_sim_time
            new_request["action"] = action

            # Try to process immediately or queue
            if self.processing_server_num < self.config.server_num:
                self._start_processing(new_request)
            elif len(self.waiting_queue) < self.config.server_num * 5:
                heappush(self.waiting_queue, (self.current_sim_time, new_request))
            else:
                # System overloaded
                self.failed_request_num += 1
                self.completed_requests_rewards.append(self.config.timeout_penalty)

    def _start_processing(self, request: Dict):
        """Start processing a request."""
        request_time = round(request["time"] * 1000)
        finish_time = self.current_sim_time + request_time
        self.processing_server_num += 1
        request["processing_start_time"] = self.current_sim_time
        heappush(self.event_queue, (finish_time, random.random(), {"type": "finish", "request": request}))

    def _process_events(self):
        """Process all events that should occur at current time."""
        # Handle timeouts
        self._handle_timeouts()

        # Handle finished requests
        while (self.event_queue and self.event_queue[0][0] <= self.current_sim_time):
            finish_time, _, event = heappop(self.event_queue)
            if event["type"] == "finish":
                self._handle_finish(event["request"])

    def _handle_timeouts(self):
        """Remove requests that waited too long."""
        current_time = self.current_sim_time
        while (self.waiting_queue and
               self.waiting_queue[0][0] < current_time - self.config.max_wait_time):
            arrival_time, request = heappop(self.waiting_queue)
            self.timeout_request_num += 1
            self.completed_requests_rewards.append(self.config.timeout_penalty)

    def _handle_finish(self, request: Dict):
        """Handle request completion."""
        self.success_request_num += 1
        self.processing_server_num -= 1

        # Calculate metrics
        score = request["score"]
        self.total_score += score
        self.score_count += 1

        self.recent_scores.append(score)
        if len(self.recent_scores) > 100:
            self.recent_scores.popleft()

        self.tot_energy += request["energy"]
        self.tot_energy_cost += request["energy"] * request["energy_price"] * self.config.pue / self.j2mwh

        # Calculate actual latency
        actual_latency = 0
        if "arrival_time" in request:
            actual_latency = request.get("processing_start_time", self.current_sim_time) - request["arrival_time"]

        # Calculate completion reward
        reward = self._calculate_completion_reward(request, actual_latency)
        self.completed_requests_rewards.append(reward)

        # Process next waiting request if any
        if self.waiting_queue:
            arrival_time, waiting_request = heappop(self.waiting_queue)
            latency = self.current_sim_time - arrival_time
            self.total_latency += latency
            self.latency_count += 1

            self.recent_latencies.append(latency)
            if len(self.recent_latencies) > 100:
                self.recent_latencies.popleft()

            waiting_request["processing_start_time"] = self.current_sim_time
            self._start_processing(waiting_request)

    def _calculate_completion_reward(self, request: Dict, latency: float) -> float:
        """Calculate reward for a completed request."""
        reward = self.config.success_reward
        reward += request["score"] * self.config.score_weight

        normalized_latency = min(latency / self.config.max_wait_time, 1.0)
        reward -= normalized_latency * self.config.latency_penalty_weight

        energy_cost = request["energy"] * request["energy_price"] * self.config.pue / self.j2mwh
        normalized_energy = min(energy_cost / 0.0002, 1.0)
        reward -= normalized_energy * self.config.energy_penalty_weight

        return reward

    def _get_observation(self) -> Dict:
        """Get current observation."""
        processing_ratio = self.processing_server_num / self.config.server_num
        waiting_ratio = min(len(self.waiting_queue) / self.config.server_num * 5, 1.0)

        # Get energy price
        current_energy_price = self.data_provider.get_energy_price(self.current_real_time)
        normalized_energy_price = self._normalize_energy_price(current_energy_price)

        # Time features
        time_of_day = self._get_time_of_day()

        # Recent performance metrics
        recent_latency = self._get_recent_latency()
        recent_score = self._get_recent_score()

        obs = {
            'processing_ratio': np.array([processing_ratio], dtype=np.float32),
            'waiting_ratio': np.array([waiting_ratio], dtype=np.float32),
            'energy_price': np.array([normalized_energy_price], dtype=np.float32),
            'time_of_day': np.array([time_of_day], dtype=np.float32),
            'recent_latency': np.array([recent_latency], dtype=np.float32),
            'recent_score': np.array([recent_score], dtype=np.float32),
        }

        # Add realistic mode features
        if self.config.is_realistic_mode:
            current_request_rate = self.data_provider.get_request_rate(self.current_real_time)
            normalized_request_rate = self._normalize_request_rate(current_request_rate)
            day_of_week = self._get_day_of_week()
            energy_price_trend = self.data_provider.get_energy_price_trend(self.current_real_time)
            request_rate_trend = self.data_provider.get_request_rate_trend(self.current_real_time)
            obs.update({
                'day_of_week': np.array([day_of_week], dtype=np.float32),
                'request_rate': np.array([normalized_request_rate], dtype=np.float32),
                'energy_price_trend': np.array([energy_price_trend], dtype=np.float32),
                'request_rate_trend': np.array([request_rate_trend], dtype=np.float32),
            })

        return obs

    def _normalize_energy_price(self, price: float) -> float:
        """Normalize energy price to [-1, 1] range."""
        # Simple normalization, can be improved with statistics
        return np.clip(price / 100.0, -1.0, 1.0)

    def _normalize_request_rate(self, request_rate: float) -> float:
        return np.clip(request_rate / 900.0, 0, 1.0)

    def _get_time_of_day(self) -> float:
        """Get normalized time of day [0, 1]."""
        total_seconds = (self.current_real_time.hour * 3600 +
                         self.current_real_time.minute * 60 +
                         self.current_real_time.second)
        return total_seconds / (24 * 3600)

    def _get_day_of_week(self) -> float:
        """Get normalized day of week [0, 1]."""
        return self.current_real_time.weekday() / 6.0

    def _get_recent_latency(self) -> float:
        """Get normalized recent latency."""
        if self.recent_latencies:
            avg_latency = np.mean(self.recent_latencies)
            return min(avg_latency / self.config.max_wait_time, 1.0)
        return 0.0

    def _get_recent_score(self) -> float:
        """Get recent score."""
        if self.recent_scores:
            return float(np.mean(self.recent_scores))
        return 0.5

    def _get_info(self) -> Dict:
        """Get additional info about the environment state."""
        total_requests = (self.success_request_num + self.failed_request_num +
                          self.denied_request_num + self.timeout_request_num)

        info = {
            'success_requests': self.success_request_num,
            'failed_requests': self.failed_request_num,
            'denied_requests': self.denied_request_num,
            'timeout_requests': self.timeout_request_num,
            'total_energy': self.tot_energy,
            'total_energy_cost': self.tot_energy_cost,
            'current_sim_time': self.current_sim_time,
            'episode_reward': self.episode_reward,
            'episode_length': self.episode_length,
            'action_counts': self.action_counts.copy(),
        }

        if total_requests > 0:
            info['success_rate'] = self.success_request_num / total_requests
            info['denial_rate'] = self.denied_request_num / total_requests
            info['timeout_rate'] = self.timeout_request_num / total_requests
        else:
            info['success_rate'] = 0.0
            info['denial_rate'] = 0.0
            info['timeout_rate'] = 0.0

        if self.score_count > 0:
            info['avg_score'] = (self.total_score / self.score_count) * 100

        if self.latency_count > 0:
            info['avg_latency'] = self.total_latency / self.latency_count

        info['profit'] = self.success_request_num / 1000 * 0.02

        return info

    def render(self, mode: str = "human"):
        """Render the environment."""
        if mode == "human":
            current_energy_price = self.data_provider.get_energy_price(self.current_real_time)

            print(f"\n--- Time: {self.current_real_time.strftime('%Y-%m-%d %H:%M:%S')} ---")
            print(f"Processing: {self.processing_server_num}/{self.config.server_num}")
            print(f"Waiting: {len(self.waiting_queue)}")
            print(f"Success/Failed/Denied/Timeout: {self.success_request_num}/"
                  f"{self.failed_request_num}/{self.denied_request_num}/"
                  f"{self.timeout_request_num}")
            print(f"Energy Price: ${current_energy_price:.2f}/MWh")
            if self.recent_scores:
                print(f"Recent Score: {np.mean(self.recent_scores):.3f}")

    def close(self):
        """Clean up resources."""
        pass

    def get_final_summary(self) -> Dict:
        """Get final summary statistics."""
        total_requests = (self.success_request_num + self.failed_request_num +
                          self.denied_request_num + self.timeout_request_num)

        summary = {
            "Total Success Requests": self.success_request_num,
            "Total Failed Requests": self.failed_request_num,
            "Total Denied Requests": self.denied_request_num,
            "Total Timeout Requests": self.timeout_request_num,
            "Success Rate": self.success_request_num / total_requests if total_requests > 0 else 0,
            "Denial Rate": self.denied_request_num / total_requests if total_requests > 0 else 0,
            "Timeout Rate": self.timeout_request_num / total_requests if total_requests > 0 else 0,
            "Average Latency (s)": (self.total_latency / self.latency_count) / 1000 if self.latency_count > 0 else 0,
            "Average Score": (self.total_score / self.score_count) * 100 if self.score_count > 0 else 0,
            "Total Energy (J)": self.tot_energy,
            "Total Energy Cost": self.tot_energy_cost * self.config.pue / self.j2mwh,
            "Total Profit": self.success_request_num / 1000 * 0.02,
            "episode_reward": self.episode_reward,
            "Action Statistics": {}
        }

        # Add action statistics
        total_actions = sum(self.action_counts.values())
        action_names = {
            0: 'fullkv', 1: 'snapkv_64', 2: 'snapkv_128',
            3: 'snapkv_256', 4: 'snapkv_512', 5: 'snapkv_1024'
        }
        if self.config.enable_deny:
            action_names[6] = 'deny'

        for action, count in self.action_counts.items():
            action_name = action_names.get(action, f"action_{action}")
            percentage = (count / total_actions * 100) if total_actions > 0 else 0
            summary["Action Statistics"][action_name] = {
                "count": count,
                "percentage": percentage
            }

        return summary
