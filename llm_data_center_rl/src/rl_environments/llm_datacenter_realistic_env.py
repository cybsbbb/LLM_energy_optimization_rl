import random
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from heapq import heappush, heappop
from typing import Dict, Tuple, Optional, Any, List
from datetime import datetime, timedelta
import os
import time
import bisect
import traceback
from kv_compression_simulate_data.gov_report.gov_report_sim_dataset import GovReportSimDataset
from utils.utils import setup_random, generate_bernoulli


class RealisticLLMDataCenterEnv(gym.Env):
    """
    Optimized Realistic OpenAI Gym environment for LLM Data Center simulation.
    Uses real-world request patterns and energy price data with optimized lookups.
    """

    metadata = {'render_modes': ['human', 'ansi']}

    def __init__(self,
                 # Data file paths
                 invocations_file: str = "/home/cybsbbbb/llm_project/llm_energy_benchmark/llm_data_center_rl/data/LLM_inference_trace/processed/2024/invocations_ten_second_Coding.csv",
                 energy_price_file: str = "/home/cybsbbbb/llm_project/llm_energy_benchmark/llm_data_center_rl/data/energy_price/processed/CAISO_20240509_20240521.csv",
                 # Environment parameters
                 server_num: int = 200,
                 max_wait_time: int = 10000,  # 10 seconds
                 pue: float = 1.3,
                 enable_deny: bool = False,
                 # Simulation settings
                 simulation_start_time: Optional[str] = None,  # "2024-05-10 00:00:00"
                 simulation_duration_hours: int = 24 * 7,
                 time_acceleration_factor: float = 1.0,  # 1.0 (real time), 60.0 (1 minute real = 1 second sim)
                 # Request scaling
                 request_scaling_factor: float = 1.0,  # Scale request volume
                 server_capacity_per_request: float = 1.0,  # How much server capacity each request uses
                 # Reward parameters
                 success_reward: float = 0.5,
                 deny_penalty: float = -0.1,
                 timeout_penalty: float = -0.0,
                 score_weight: float = 2.0,
                 latency_penalty_weight: float = 0.0,
                 energy_penalty_weight: float = 1.0,
                 render_mode: Optional[str] = None,
                 seed: Optional[int] = None):

        super().__init__()

        # Environment parameters
        self.server_num = server_num
        self.max_wait_time = max_wait_time
        self.pue = pue
        self.enable_deny = enable_deny
        self.render_mode = render_mode
        self.simulation_duration_hours = simulation_duration_hours
        self.time_acceleration_factor = time_acceleration_factor
        self.request_scaling_factor = request_scaling_factor
        self.server_capacity_per_request = server_capacity_per_request

        # Reward parameters
        self.success_reward = success_reward
        self.deny_penalty = deny_penalty
        self.timeout_penalty = timeout_penalty
        self.score_weight = score_weight
        self.latency_penalty_weight = latency_penalty_weight
        self.energy_penalty_weight = energy_penalty_weight

        # Load real-world data with optimization
        self.load_and_optimize_data(invocations_file, energy_price_file)

        # Set simulation time range
        self.set_simulation_time_range(simulation_start_time)

        # Dataset for LLM requests
        self.gov_report_sim_dataset = GovReportSimDataset()

        # Action space: 6 KV cache configurations + optional deny action
        self.n_actions = 7 if enable_deny else 6
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
        if enable_deny:
            self.action_to_kv_size[6] = 'deny'

        # Enhanced observation space with real-world context
        self.observation_space = spaces.Dict({
            'processing_ratio': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'waiting_ratio': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'energy_price': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),  # Normalized
            'energy_price_trend': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),  # Recent trend
            'time_of_day': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'day_of_week': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'request_rate': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # Current request rate
            'request_rate_trend': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),  # Recent trend
            'recent_latency': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'recent_score': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })

        # Initialize state
        self.reset(seed=seed)

    def load_and_optimize_data(self, invocations_file: str, energy_price_file: str):
        """Load real-world data and create optimized lookup structures."""
        # Load invocations data
        if os.path.exists(invocations_file):
            self.invocations_df = pd.read_csv(invocations_file)
            self.invocations_df['Time'] = pd.to_datetime(self.invocations_df['Time'])
            self.invocations_df.set_index('Time', inplace=True)
            self.invocations_df.sort_index(inplace=True)  # Ensure sorted for binary search
            print(f"Loaded {len(self.invocations_df)} invocation records")
        else:
            raise FileNotFoundError(f"Invocations file not found: {invocations_file}")

        # Load energy price data
        if os.path.exists(energy_price_file):
            self.energy_df = pd.read_csv(energy_price_file)
            self.energy_df['INTERVALSTARTTIME_GMT'] = pd.to_datetime(self.energy_df['INTERVALSTARTTIME_GMT'])
            self.energy_df.set_index('INTERVALSTARTTIME_GMT', inplace=True)
            self.energy_df.sort_index(inplace=True)  # Ensure sorted for binary search
            print(f"Loaded {len(self.energy_df)} energy price records")
        else:
            raise FileNotFoundError(f"Energy price file not found: {energy_price_file}")

        # Create optimized lookup structures for O(log n) access
        self._create_optimized_lookups()

        # Calculate energy prices statistics
        self.energy_price_mean = self.energy_df['VALUE'].mean()
        self.energy_price_std = self.energy_df['VALUE'].std()
        print(f"Energy price stats: mean={self.energy_price_mean:.2f}, std={self.energy_price_std:.2f}")

        # Calculate request rate statistics
        self.request_rate_mean = self.invocations_df['TIMESTAMP'].mean()
        self.request_rate_std = self.invocations_df['TIMESTAMP'].std()
        print(f"Request rate stats: mean={self.request_rate_mean:.2f}, std={self.request_rate_std:.2f}")

    def _create_optimized_lookups(self):
        """Create optimized lookup structures for fast binary search."""
        print("Creating optimized lookup structures...")

        # Energy price lookup - convert timestamps to milliseconds for faster comparison
        self.energy_timestamps_ms = []
        self.energy_values = []

        for timestamp, row in self.energy_df.iterrows():
            timestamp_ms = int(timestamp.timestamp() * 1000)
            self.energy_timestamps_ms.append(timestamp_ms)
            self.energy_values.append(row['VALUE'])

        print(f"Created energy price lookup with {len(self.energy_timestamps_ms)} entries")

        # Invocations lookup - convert timestamps to milliseconds for faster comparison
        self.invocation_timestamps_ms = []
        self.invocation_values = []

        for timestamp, row in self.invocations_df.iterrows():
            timestamp_ms = int(timestamp.timestamp() * 1000)
            self.invocation_timestamps_ms.append(timestamp_ms)
            self.invocation_values.append(row['TIMESTAMP'])

        print(f"Created invocation lookup with {len(self.invocation_timestamps_ms)} entries")

    def get_current_energy_price(self, current_time: datetime) -> float:
        """Get the current energy price using optimized binary search."""
        current_time_ms = int(current_time.timestamp() * 1000)

        # Use binary search to find the appropriate energy price interval
        # Find the rightmost timestamp that is <= current_time_ms
        idx = bisect.bisect_right(self.energy_timestamps_ms, current_time_ms) - 1

        if idx >= 0:
            energy_price = self.energy_values[idx]
        else:
            # If before all data, use first available price
            energy_price = self.energy_values[0]

        return energy_price

    def get_current_request_rate(self, current_time: datetime) -> float:
        """Get the current request rate using optimized binary search."""
        current_time_ms = int(current_time.timestamp() * 1000)

        # Use binary search to find the closest time in the invocations data
        # Find the closest timestamp
        idx = bisect.bisect_right(self.invocation_timestamps_ms, current_time_ms) - 1

        if idx >= 0:
            request_rate = self.invocation_values[idx] * self.request_scaling_factor
        else:
            # If before all data, use first available price
            request_rate = self.invocation_values[0] * self.request_scaling_factor

        return request_rate

    def get_request_rate_trend(self, current_time: datetime, lookback_minutes: int = 30) -> float:
        """Calculate request rate trend over recent period using optimized lookups."""
        current_time_ms = int(current_time.timestamp() * 1000)
        lookback_ms = lookback_minutes * 60 * 1000
        lookback_start_ms = current_time_ms - lookback_ms

        # Use binary search to find the range of relevant timestamps
        start_idx = bisect.bisect_left(self.invocation_timestamps_ms, lookback_start_ms)
        end_idx = bisect.bisect_right(self.invocation_timestamps_ms, current_time_ms)

        if end_idx - start_idx < 2:
            return 0.0

        # Get the relevant data points
        times = np.arange(end_idx - start_idx)
        rates = np.array(self.invocation_values[start_idx:end_idx])

        if len(times) >= 2:
            # Calculate linear trend
            trend = np.polyfit(times, rates, 1)[0]  # Slope of linear fit
            return np.clip(trend / self.request_rate_std, -1, 1)  # Normalize

        return 0.0

    def get_energy_price_trend(self, current_time: datetime, lookback_minutes: int = 30) -> float:
        """Calculate energy price trend over recent period using optimized lookups."""
        current_time_ms = int(current_time.timestamp() * 1000)
        lookback_ms = lookback_minutes * 60 * 1000
        lookback_start_ms = current_time_ms - lookback_ms

        # Use binary search to find the range of relevant timestamps
        start_idx = bisect.bisect_left(self.energy_timestamps_ms, lookback_start_ms)
        end_idx = bisect.bisect_right(self.energy_timestamps_ms, current_time_ms)

        if end_idx - start_idx < 2:
            return 0.0

        # Get the relevant data points
        times = np.arange(end_idx - start_idx)
        prices = np.array(self.energy_values[start_idx:end_idx])

        if len(times) >= 2:
            # Calculate linear trend
            trend = np.polyfit(times, prices, 1)[0]  # Slope of linear fit
            return np.clip(trend / self.energy_price_std, -1, 1)  # Normalize

        return 0.0

    def set_simulation_time_range(self, start_time: Optional[str] = None):
        """Set the time range for simulation."""
        if start_time is None:
            # Use the start of the invocations data
            self.simulation_start = self.invocations_df.index[0]
        else:
            self.simulation_start = pd.to_datetime(start_time)

        self.simulation_end = self.simulation_start + timedelta(hours=self.simulation_duration_hours)

        # Filter data to simulation range (for reference, actual lookups use binary search)
        self.sim_invocations = self.invocations_df[
            (self.invocations_df.index >= self.simulation_start) &
            (self.invocations_df.index <= self.simulation_end)
            ].copy()

        if len(self.sim_invocations) == 0:
            print(
                f"Warning: No invocation data found in exact time range {self.simulation_start} to {self.simulation_end}")
            print("Using binary search for closest matches...")

        print(f"Simulation time range: {self.simulation_start} to {self.simulation_end}")
        print(f"Simulation duration: {self.simulation_duration_hours} hours")

    def _safe_action_conversion(self, action):
        """Safely convert action to integer and ensure it's within bounds."""
        if isinstance(action, np.ndarray):
            if action.ndim == 0:
                action = action.item()
            else:
                action = action[0]
        elif isinstance(action, tuple):
            action = action[0]

        action = int(action)
        action = np.clip(action, 0, self.n_actions - 1)
        return action

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        if seed is not None:
            setup_random(seed)

        # Reset simulation state
        self.current_real_time = self.simulation_start
        self.current_sim_time = 0  # Simulation time in milliseconds
        self.event_queue = []
        self.processing_server_num = 0
        self.waiting_queue = []

        # Reset metrics
        self.success_request_num = 0
        self.failed_request_num = 0
        self.denied_request_num = 0
        self.timeout_request_num = 0

        # For efficient average calculation
        self.total_latency = 0.0
        self.total_score = 0.0
        self.latency_count = 0
        self.score_count = 0

        self.tot_energy = 0
        self.tot_energy_cost = 0

        # Recent metrics for observation
        self.recent_latencies = []
        self.recent_scores = []
        self.recent_energy_prices = []

        # Episode metrics
        self.episode_reward = 0
        self.episode_length = 0

        # Action statistics
        self.action_counts = {i: 0 for i in range(self.n_actions)}
        self.action_names = {
            0: 'fullkv',
            1: 'snapkv_64',
            2: 'snapkv_128',
            3: 'snapkv_256',
            4: 'snapkv_512',
            5: 'snapkv_1024',
        }
        if self.enable_deny:
            self.action_names[6] = 'deny'

        # Step tracking
        self.step_start_time = 0
        self.requests_in_current_step = 0
        self.current_action = None

        # For completion-based rewards
        self.completed_requests_rewards = []

        # Initialize the next request generation
        self.next_request_check_time = 0
        self.last_request_time = self.current_real_time

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.episode_length += 1

        # Safely convert and validate action
        action = self._safe_action_conversion(action)

        # Set the action for this step
        self.current_action = action

        if action in self.action_counts:
            self.action_counts[action] += 1
        else:
            action = np.clip(action, 0, self.n_actions - 1)
            self.action_counts[action] += 1

        # Initialize step tracking
        self.step_start_time = self.current_sim_time
        step_duration_ms = 10000  # 10 seconds per step
        step_end_time = self.step_start_time + step_duration_ms
        self.requests_in_current_step = 0
        self.completed_requests_rewards = []

        # Process events for this step
        while self.current_sim_time < step_end_time:
            # Update real time
            self.current_real_time = self.simulation_start + timedelta(
                milliseconds=self.current_sim_time / self.time_acceleration_factor
            )

            # Check if we've reached the end of simulation
            if self.current_real_time >= self.simulation_end:
                break

            # Generate new requests based on real data (using optimized lookup)
            self._generate_requests_from_real_data()

            # Process existing events
            self._process_events()

            # Advance simulation time
            self.current_sim_time += 50  # Advance by 50 ms

        # Calculate step reward
        if self.completed_requests_rewards:
            step_reward = sum(self.completed_requests_rewards) / len(self.completed_requests_rewards)
        else:
            step_reward = 0.0

        self.episode_reward += step_reward

        # Check if episode is done
        terminated = self.current_real_time >= self.simulation_end
        truncated = False

        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        info['requests_processed'] = self.requests_in_current_step
        info['step_duration_ms'] = self.current_sim_time - self.step_start_time
        info['real_time'] = self.current_real_time.isoformat()

        return observation, step_reward, terminated, truncated, info

    def _generate_requests_from_real_data(self):
        """Generate requests based on real-world data using optimized lookup."""
        # Get current request rate (now using binary search)
        current_request_rate = self.get_current_request_rate(self.current_real_time)

        # Convert to requests per second
        requests_per_second = current_request_rate / 900.0

        # Generate requests using Poisson process
        # Probability of at least one request in this 1-second interval
        prob_request = min(1.0, requests_per_second * self.time_acceleration_factor)

        if random.random() < prob_request:
            self._process_new_request()
            self.requests_in_current_step += 1

    def _process_new_request(self):
        """Process a new request with the current action."""
        action_type = self.action_to_kv_size[self.current_action]

        if action_type == 'deny':
            self.denied_request_num += 1
            self.completed_requests_rewards.append(self.deny_penalty)
        else:
            # Create new request
            new_request = self.gov_report_sim_dataset.get_random_item(action_type)
            new_request["energy_price"] = self.get_current_energy_price(self.current_real_time)  # Now O(log n)
            new_request["arrival_time"] = self.current_sim_time
            new_request["action"] = self.current_action

            # Try to process immediately or queue
            if self.processing_server_num < self.server_num:
                self._start_processing(new_request)
            elif len(self.waiting_queue) < self.server_num * 5:
                heappush(self.waiting_queue, (self.current_sim_time, new_request))
            else:
                # System overloaded
                self.failed_request_num += 1
                self.completed_requests_rewards.append(self.timeout_penalty)

    def _start_processing(self, request: Dict):
        """Start processing a request."""
        request_time = round(request["time"] * 1000)  # Convert to milliseconds
        finish_time = self.current_sim_time + request_time
        self.processing_server_num += 1
        request["processing_start_time"] = self.current_sim_time
        heappush(self.event_queue, (finish_time, random.random(), {"type": "finish", "request": request}))

    def _process_events(self):
        """Process all events that should occur at current time."""
        # Handle timeouts
        self._handle_timeouts()

        # Handle finished requests
        while (len(self.event_queue) > 0 and
               self.event_queue[0][0] <= self.current_sim_time):
            finish_time, _, event = heappop(self.event_queue)
            if event["type"] == "finish":
                self._handle_finish(event["request"])

    def _handle_timeouts(self):
        """Remove requests that waited too long."""
        current_time = self.current_sim_time
        while (len(self.waiting_queue) > 0 and
               self.waiting_queue[0][0] < current_time - self.max_wait_time):
            arrival_time, request = heappop(self.waiting_queue)
            self.timeout_request_num += 1
            self.completed_requests_rewards.append(self.timeout_penalty)

    def _handle_finish(self, request: Dict):
        """Handle request completion."""
        self.success_request_num += 1
        self.processing_server_num -= 1

        # Calculate actual latency
        if "arrival_time" in request:
            actual_latency = request.get("processing_start_time", self.current_sim_time) - request["arrival_time"]
        else:
            actual_latency = 0

        # Update metrics
        score = request["score"]
        self.total_score += score
        self.score_count += 1

        self.recent_scores.append(score)
        if len(self.recent_scores) > 100:
            self.recent_scores.pop(0)

        self.tot_energy += request["energy"]
        self.tot_energy_cost += request["energy"] * request["energy_price"]

        # Calculate completion reward
        reward = self._calculate_completion_reward(request, actual_latency)
        self.completed_requests_rewards.append(reward)

        # Process next waiting request if any
        if len(self.waiting_queue) > 0:
            arrival_time, waiting_request = heappop(self.waiting_queue)
            latency = self.current_sim_time - arrival_time
            self.total_latency += latency
            self.latency_count += 1

            self.recent_latencies.append(latency)
            if len(self.recent_latencies) > 100:
                self.recent_latencies.pop(0)

            waiting_request["processing_start_time"] = self.current_sim_time
            self._start_processing(waiting_request)

    def _calculate_completion_reward(self, request: Dict, latency: float) -> float:
        """Calculate reward for a completed request."""
        reward = self.success_reward
        reward += request["score"] * self.score_weight

        normalized_latency = min(latency / self.max_wait_time, 1.0)
        reward -= normalized_latency * self.latency_penalty_weight

        energy_cost = request["energy"] * request["energy_price"] * self.pue / 3_600_000_000
        normalized_energy = min(energy_cost / 0.0002, 1.0)
        reward -= normalized_energy * self.energy_penalty_weight

        return reward

    def _get_observation(self) -> Dict:
        """Get current observation using optimized lookups."""
        processing_ratio = self.processing_server_num / self.server_num
        waiting_ratio = min(len(self.waiting_queue) / (self.server_num * 5), 1.0)

        # Get normalized energy price (now O(log n))
        current_energy_price = self.get_current_energy_price(self.current_real_time)
        normalized_energy_price = (current_energy_price - self.energy_price_mean) / self.energy_price_std
        normalized_energy_price = np.clip(normalized_energy_price, -3, 3) / 3  # Clip to [-1, 1]

        # Get energy price trend (now O(log n))
        energy_price_trend = self.get_energy_price_trend(self.current_real_time)

        # Time features
        time_of_day = (self.current_real_time.hour * 3600 +
                       self.current_real_time.minute * 60 +
                       self.current_real_time.second) / (24 * 3600)  # Normalize to [0, 1]

        day_of_week = self.current_real_time.weekday() / 6.0  # Monday=0, Sunday=6, normalize to [0, 1]

        # Request rate features (now O(log n))
        current_request_rate = self.get_current_request_rate(self.current_real_time)
        normalized_request_rate = (current_request_rate - self.request_rate_mean) / self.request_rate_std
        normalized_request_rate = np.clip(normalized_request_rate, -3, 3) / 3  # Clip to [-1, 1]
        normalized_request_rate = (normalized_request_rate + 1) / 2  # Convert to [0, 1]

        # Request rate trend (now O(log n))
        request_rate_trend = self.get_request_rate_trend(self.current_real_time)

        # Recent performance metrics
        recent_latency = 0.0
        if self.recent_latencies:
            recent_latency = min(sum(self.recent_latencies) / len(self.recent_latencies) / self.max_wait_time, 1.0)

        recent_score = 0.5  # Default
        if self.recent_scores:
            recent_score = sum(self.recent_scores) / len(self.recent_scores)

        return {
            'processing_ratio': np.array([processing_ratio], dtype=np.float32),
            'waiting_ratio': np.array([waiting_ratio], dtype=np.float32),
            'energy_price': np.array([normalized_energy_price], dtype=np.float32),
            'energy_price_trend': np.array([energy_price_trend], dtype=np.float32),
            'time_of_day': np.array([time_of_day], dtype=np.float32),
            'day_of_week': np.array([day_of_week], dtype=np.float32),
            'request_rate': np.array([normalized_request_rate], dtype=np.float32),
            'request_rate_trend': np.array([request_rate_trend], dtype=np.float32),
            'recent_latency': np.array([recent_latency], dtype=np.float32),
            'recent_score': np.array([recent_score], dtype=np.float32),
        }

    def _get_info(self) -> Dict:
        """Get additional info about the environment state."""
        total_requests = self.success_request_num + self.failed_request_num + self.denied_request_num + self.timeout_request_num

        info = {
            'success_requests': self.success_request_num,
            'failed_requests': self.failed_request_num,
            'denied_requests': self.denied_request_num,
            'timeout_requests': self.timeout_request_num,
            'total_energy': self.tot_energy,
            'total_energy_cost': self.tot_energy_cost,
            'current_sim_time': self.current_sim_time,
            'current_real_time': self.current_real_time.isoformat(),
            'episode_reward': self.episode_reward,
            'episode_length': self.episode_length,
            'current_request_rate': self.get_current_request_rate(self.current_real_time),
            'current_energy_price': self.get_current_energy_price(self.current_real_time),
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

        # Calculate profit
        info['profit'] = self.success_request_num / 1000 * 0.02

        # Add action statistics
        info['action_counts'] = self.action_counts.copy()

        return info

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            print(f"\n--- Real Time: {self.current_real_time.strftime('%Y-%m-%d %H:%M:%S')} ---")
            print(f"Sim Time: {self.current_sim_time / 1000:.1f}s")
            print(f"Processing: {self.processing_server_num}/{self.server_num}")
            print(f"Waiting: {len(self.waiting_queue)}")
            print(
                f"Success/Failed/Denied/Timeout: {self.success_request_num}/{self.failed_request_num}/{self.denied_request_num}/{self.timeout_request_num}")
            print(f"Current Request Rate: {self.get_current_request_rate(self.current_real_time):.1f} req/10s")
            print(f"Energy Price: ${self.get_current_energy_price(self.current_real_time):.2f}/MWh")
            if self.recent_scores:
                print(f"Recent Score: {sum(self.recent_scores) / len(self.recent_scores):.3f}")
            if self.current_action is not None:
                print(f"Current Action: {self.action_names[self.current_action]}")

    def get_final_summary(self) -> Dict:
        """Get final summary statistics."""
        total_requests = self.success_request_num + self.failed_request_num + self.denied_request_num + self.timeout_request_num

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
            "Total Energy Cost": self.tot_energy_cost * self.pue / 3_600_000_000,
            "Total Profit": self.success_request_num / 1000 * 0.02,
            "Simulation Start": self.simulation_start.isoformat(),
            "Simulation End": self.simulation_end.isoformat(),
            "Action Statistics": {}
        }

        # Add action statistics
        total_actions = sum(self.action_counts.values())
        for action, count in self.action_counts.items():
            action_name = self.action_names.get(action, f"action_{action}")
            percentage = (count / total_actions * 100) if total_actions > 0 else 0
            summary["Action Statistics"][action_name] = {
                "count": count,
                "percentage": percentage
            }

        return summary

    def print_summary(self):
        """Print summary statistics."""
        print("\nOptimized Realistic Simulation Summary:")
        summary = self.get_final_summary()

        # Print basic statistics
        for key, value in summary.items():
            if key not in ["Action Statistics", "Simulation Start", "Simulation End", "Cache Performance"]:
                if isinstance(value, float):
                    print(f"{key}: {value:.3f}")
                else:
                    print(f"{key}: {value}")
            elif key in ["Simulation Start", "Simulation End"]:
                print(f"{key}: {value}")
            elif key == "Cache Performance":
                print(f"{key}: {value}")

        # Print action statistics
        print("\nAction Statistics:")
        print(f"{'Action':<15} {'Count':<10} {'Percentage':<10}")
        print("-" * 35)

        action_stats = summary["Action Statistics"]
        for action_name, stats in action_stats.items():
            print(f"{action_name:<15} {stats['count']:<10} {stats['percentage']:<10.1f}%")

    def close(self):
        """Clean up resources."""
        # Clear caches to free memory
        pass


# Example usage and testing
if __name__ == "__main__":
    print("Testing Optimized Realistic LLM Data Center Environment")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Creating optimized environment for functional test...")

    try:
        # Create optimized environment
        env = RealisticLLMDataCenterEnv(
            invocations_file="../../data/LLM_inference_trace/processed/2024/invocations_ten_second_Coding.csv",
            energy_price_file="../../data/energy_price/processed/CAISO_20240509_20240521.csv",
            simulation_start_time="2024-05-10 00:00:00",
            simulation_duration_hours=2,  # 2 hours for testing
            # server_num=100,
            # request_scaling_factor=1.0,
            # time_acceleration_factor=1.0,
            enable_deny=False,
        )

        print("Optimized environment created successfully!")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")

        # Test episode
        obs, info = env.reset()
        print(f"\nInitial observation keys: {list(obs.keys())}")

        # Run a few steps to test performance
        start_time = time.time()

        for step in range(100):  # More steps to test caching
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)

            if step % 10 == 0:
                print(f"Step {step}: action={action}, reward={reward:.3f}, "
                      f"energy_price=${info.get('current_energy_price', 0):.2f}, "
                      f"request_rate={info.get('current_request_rate', 0):.1f}")

            if terminated or truncated:
                print("Episode ended early")
                break

        elapsed_time = time.time() - start_time
        print(
            f"\nSimulation performance: {elapsed_time:.2f}s for {step + 1} steps ({elapsed_time * 1000 / (step + 1):.1f}ms per step)")

        # Print summary
        env.print_summary()
        env.close()

        print("\n✓ Optimized environment test completed successfully!")

    except Exception as e:
        print(f"Error testing optimized environment: {e}")

        traceback.print_exc()
