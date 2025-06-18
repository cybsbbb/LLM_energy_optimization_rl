import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from heapq import heappush, heappop
from typing import Dict, Tuple, Optional, Any, List
import random

from data_center_simulator.simulate_data.gov_report.gov_report_sim_dataset import GovReportSimDataset
from data_center_simulator.energy_price.caiso import data_20250421
from data_center_simulator.utils import setup_random, generate_bernoulli


class LLMDataCenterEnv(gym.Env):
    """OpenAI Gym compatible environment for LLM Data Center simulation with delayed rewards."""

    metadata = {'render_modes': ['human', 'ansi']}

    def __init__(self,
                 total_time: int = 24 * 3600 * 1000,  # 24 hours in milliseconds
                 time_interval: int = 100,  # 10ms
                 bernoulli_prob: float = 0.2,
                 server_num: int = 200,
                 max_wait_time: int = 10000,  # 10 seconds, customer will leave after 10s wait
                 pue: float = 1.3,  # PUE (Power usage effectiveness)
                 enable_deny: bool = True,  # Option to enable/disable deny action
                 # Reward weights
                 success_reward: float = 0.2,
                 deny_penalty: float = -0.2,
                 latency_penalty_weight: float = 0.5,
                 score_weight: float = 10.0,
                 energy_penalty_weight: float = 0.3,
                 render_mode: Optional[str] = None,
                 seed: Optional[int] = None):

        super().__init__()

        # Environment parameters
        self.total_time = total_time
        self.time_interval = time_interval
        self.bernoulli_prob = bernoulli_prob
        self.server_num = server_num
        self.max_wait_time = max_wait_time
        self.pue = pue
        self.enable_deny = enable_deny
        self.render_mode = render_mode

        # Reward parameters
        self.success_reward = success_reward
        self.deny_penalty = deny_penalty
        self.latency_penalty_weight = latency_penalty_weight
        self.score_weight = score_weight
        self.energy_penalty_weight = energy_penalty_weight

        # Dataset and energy price
        self.gov_report_sim_dataset = GovReportSimDataset()
        self.energy_price = data_20250421
        self.five_minutes_ms = 300 * 1000
        self.j2mwh = 3_600_000_000

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

        # Observation space
        self.observation_space = spaces.Dict({
            'processing_ratio': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'waiting_ratio': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'energy_price': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'time_of_day': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'recent_latency': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'recent_score': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })

        # Initialize state
        if seed is not None:
            self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        if seed is not None:
            setup_random(seed)

        # Reset simulation state
        self.current_time = 0
        self.event_queue = []
        self.processing_server_num = 0
        self.waiting_queue = []

        # Reset metrics
        self.success_request_num = 0
        self.failed_request_num = 0
        self.denied_request_num = 0

        # For efficient average calculation
        self.total_latency = 0.0
        self.total_score = 0.0
        self.latency_count = 0
        self.score_count = 0

        self.tot_energy = 0
        self.tot_energy_cost = 0

        # Recent metrics for observation (keep small fixed-size buffers)
        self.recent_latencies = []
        self.recent_scores = []

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

        # For delayed rewards - track pending requests and their actions
        self.pending_rewards = []  # List of rewards to be delivered
        self.request_to_action = {}  # Map request ID to the action that created it
        self.next_request_id = 0

        # Initialize event queue
        heappush(self.event_queue, (0, random.random(), {"type": "timestamp"}))

        # Advance simulation to first decision point
        self._advance_to_decision_point()

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.episode_length += 1

        # Ensure action is an integer (some RL algorithms return numpy arrays)
        if isinstance(action, np.ndarray):
            action = int(action.item())
        elif isinstance(action, tuple):
            action = int(action[0].item())
        else:
            action = int(action)

        # Track action taken
        self.action_counts[action] += 1

        # Store current request ID
        current_request_id = self.next_request_id
        self.next_request_id += 1

        # Convert action to KV size or deny
        action_type = self.action_to_kv_size[action]

        # Calculate immediate reward from any pending completed requests
        immediate_reward = self._collect_pending_rewards()

        # Handle immediate deny action
        if action_type == 'deny':
            self.denied_request_num += 1
            # Add deny penalty as immediate reward
            immediate_reward += self.deny_penalty

            # Advance simulation to next decision point
            self._advance_to_decision_point()

            # Collect any new pending rewards
            immediate_reward += self._collect_pending_rewards()

        else:
            # Get new request with the selected KV size
            new_request = self.gov_report_sim_dataset.get_random_item(action_type)
            new_request["energy_price"] = self._get_cur_energy_price(self.current_time)
            new_request["request_id"] = current_request_id
            new_request["arrival_time"] = self.current_time

            # Store the action that created this request
            self.request_to_action[current_request_id] = action

            # Handle the new request
            self._handle_new_request(self.current_time, new_request)

            # Advance simulation to next decision point
            self._advance_to_decision_point()

            # Collect any new pending rewards
            immediate_reward += self._collect_pending_rewards()

        self.episode_reward += immediate_reward

        # Check if episode is done
        terminated = self.current_time >= self.total_time
        truncated = False

        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        info['pending_rewards'] = len(self.pending_rewards)

        return observation, immediate_reward, terminated, truncated, info

    def _collect_pending_rewards(self) -> float:
        """Collect all pending rewards from completed requests."""
        total_reward = 0.0

        while self.pending_rewards:
            reward_info = self.pending_rewards.pop(0)
            total_reward += reward_info['reward']

        return total_reward

    def _calculate_request_reward(self, request: Dict, latency: float, success: bool) -> float:
        """Calculate reward for a specific request based on its outcome."""
        if not success:
            # Failed request (timed out) - no reward
            return 0.0

        # Base success reward
        reward = self.success_reward

        # Quality score component (0-1)
        score = request['score']
        reward += score * self.score_weight

        # Latency penalty (normalized by max wait time)
        normalized_latency = min(latency / self.max_wait_time, 1.0)
        reward -= normalized_latency * self.latency_penalty_weight

        # Energy cost penalty
        energy_cost = request['energy'] * request['energy_price'] * self.pue / self.j2mwh
        # Normalize energy cost (typical range 0-0.1)
        normalized_energy_cost = min(energy_cost / 0.1, 1.0)
        reward -= normalized_energy_cost * self.energy_penalty_weight

        return reward

    def _advance_to_decision_point(self):
        """Advance simulation until next decision point (new request)."""
        while len(self.event_queue) > 0:
            cur_time, _, cur_event = heappop(self.event_queue)

            if cur_time >= self.total_time:
                self.current_time = self.total_time
                break

            self.current_time = cur_time

            if cur_event["type"] == "timestamp":
                # Handle failed requests
                self._handle_failed_request(cur_time)

                # Schedule next timestamp
                heappush(self.event_queue, (cur_time + self.time_interval, random.random(), cur_event))

                # Check if new request arrives
                if generate_bernoulli(self.bernoulli_prob):
                    # Decision point reached
                    break

            elif cur_event["type"] == "finish":
                self._handle_finish(cur_time, cur_event)

    def _handle_failed_request(self, cur_time: int):
        """Remove requests that waited too long."""
        while len(self.waiting_queue) > 0 and self.waiting_queue[0][0] < cur_time - self.max_wait_time:
            arrival_time, waiting_event = self.waiting_queue.pop(0)
            self.failed_request_num += 1

            # Calculate reward for failed request
            request = waiting_event["request"]
            if "request_id" in request and request["request_id"] in self.request_to_action:
                latency = cur_time - arrival_time
                reward = self._calculate_request_reward(request, latency, success=False)

                # Add to pending rewards
                self.pending_rewards.append({
                    'request_id': request["request_id"],
                    'reward': reward,
                    'type': 'failed',
                    'latency': latency
                })

                # Clean up
                del self.request_to_action[request["request_id"]]

    def _handle_new_request(self, cur_time: int, new_request: Dict):
        """Process a new request."""
        if self.processing_server_num < self.server_num:
            new_request_time = round(new_request["time"] * 1000)
            finish_time = cur_time + new_request_time
            self.processing_server_num += 1

            # No initial latency for immediately processed requests
            new_request["processing_start_time"] = cur_time

            heappush(self.event_queue, (finish_time, random.random(), {"type": "finish", "request": new_request}))
        else:
            heappush(self.waiting_queue, (cur_time, {"type": "waiting", "request": new_request}))

    def _handle_finish(self, cur_time: int, cur_event: Dict):
        """Handle request completion."""
        self._handle_failed_request(cur_time)
        self.success_request_num += 1
        self.processing_server_num -= 1

        request = cur_event["request"]

        # Calculate latency (time from arrival to processing start)
        if "processing_start_time" in request and "arrival_time" in request:
            latency = request["processing_start_time"] - request["arrival_time"]
        else:
            latency = 0

        # Update metrics
        score = request["score"]
        self.total_score += score
        self.score_count += 1

        # Update recent scores buffer (keep small)
        self.recent_scores.append(score)
        if len(self.recent_scores) > 10:
            self.recent_scores.pop(0)

        self.tot_energy += request["energy"]
        self.tot_energy_cost += request["energy"] * request["energy_price"]

        # Calculate reward for this completed request
        if "request_id" in request and request["request_id"] in self.request_to_action:
            reward = self._calculate_request_reward(request, latency, success=True)

            # Add to pending rewards
            self.pending_rewards.append({
                'request_id': request["request_id"],
                'reward': reward,
                'type': 'success',
                'latency': latency,
                'score': score,
                'energy': request["energy"]
            })

            # Clean up
            del self.request_to_action[request["request_id"]]

        # Process waiting request if any
        if len(self.waiting_queue) > 0:
            oldest_time, oldest_event = heappop(self.waiting_queue)
            waiting_request = oldest_event["request"]

            # Calculate latency for the waiting request
            wait_latency = cur_time - oldest_time
            self.total_latency += wait_latency
            self.latency_count += 1

            # Update recent latencies buffer
            self.recent_latencies.append(wait_latency)
            if len(self.recent_latencies) > 10:
                self.recent_latencies.pop(0)

            # Update request with processing start time
            waiting_request["processing_start_time"] = cur_time

            self._handle_new_request(cur_time, waiting_request)

    def _get_cur_energy_price(self, cur_time: int) -> float:
        """Get current energy price."""
        return self.energy_price[min(cur_time // self.five_minutes_ms, len(self.energy_price) - 1)]

    def _get_observation(self) -> Dict:
        """Get current observation."""
        processing_ratio = self.processing_server_num / self.server_num
        waiting_ratio = min(len(self.waiting_queue) / self.server_num, 1.0)
        energy_price = self._get_cur_energy_price(self.current_time) / 100.0  # Normalize
        time_of_day = (self.current_time % (24 * 3600 * 1000)) / (24 * 3600 * 1000)

        # Recent performance metrics
        recent_latency = 0.0
        if self.recent_latencies:
            recent_latency = min(sum(self.recent_latencies) / len(self.recent_latencies) / self.max_wait_time, 1.0)

        recent_score = 0.5  # Default middle score
        if self.recent_scores:
            recent_score = sum(self.recent_scores) / len(self.recent_scores)

        return {
            'processing_ratio': np.array([processing_ratio], dtype=np.float32),
            'waiting_ratio': np.array([waiting_ratio], dtype=np.float32),
            'energy_price': np.array([energy_price], dtype=np.float32),
            'time_of_day': np.array([time_of_day], dtype=np.float32),
            'recent_latency': np.array([recent_latency], dtype=np.float32),
            'recent_score': np.array([recent_score], dtype=np.float32),
        }

    def _get_info(self) -> Dict:
        """Get additional info about the environment state."""
        total_requests = self.success_request_num + self.failed_request_num + self.denied_request_num

        info = {
            'success_requests': self.success_request_num,
            'failed_requests': self.failed_request_num,
            'denied_requests': self.denied_request_num,
            'total_energy': self.tot_energy,
            'total_energy_cost': self.tot_energy_cost,
            'current_time': self.current_time,
            'episode_reward': self.episode_reward,
            'episode_length': self.episode_length,
        }

        if total_requests > 0:
            info['success_rate'] = self.success_request_num / total_requests
            info['denial_rate'] = self.denied_request_num / total_requests
        else:
            info['success_rate'] = 0.0
            info['denial_rate'] = 0.0

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
            print(f"\n--- Time: {self.current_time / 1000:.1f}s ---")
            print(f"Processing: {self.processing_server_num}/{self.server_num}")
            print(f"Waiting: {len(self.waiting_queue)}")
            print(
                f"Success/Failed/Denied: {self.success_request_num}/{self.failed_request_num}/{self.denied_request_num}")
            print(f"Pending Rewards: {len(self.pending_rewards)}")
            if self.recent_scores:
                print(f"Recent Score: {sum(self.recent_scores) / len(self.recent_scores):.3f}")
            print(f"Energy Price: ${self._get_cur_energy_price(self.current_time):.2f}")
        elif self.render_mode == "ansi":
            output = []
            output.append(f"Time: {self.current_time / 1000:.1f}s")
            output.append(f"Processing: {self.processing_server_num}/{self.server_num}")
            output.append(f"Waiting: {len(self.waiting_queue)}")
            output.append(
                f"Success/Failed/Denied: {self.success_request_num}/{self.failed_request_num}/{self.denied_request_num}")
            return "\n".join(output)

    def get_final_summary(self) -> Dict:
        """Get final summary statistics."""
        total_requests = self.success_request_num + self.failed_request_num + self.denied_request_num

        summary = {
            "Total Success Requests": self.success_request_num,
            "Total Failed Requests": self.failed_request_num,
            "Total Denied Requests": self.denied_request_num,
            "Success Rate": self.success_request_num / total_requests if total_requests > 0 else 0,
            "Denial Rate": self.denied_request_num / total_requests if total_requests > 0 else 0,
            "Average Latency (s)": (self.total_latency / self.latency_count) / 1000 if self.latency_count > 0 else 0,
            "Average Score": (self.total_score / self.score_count) * 100 if self.score_count > 0 else 0,
            "Total Energy (J)": self.tot_energy,
            "Total Energy Cost": self.tot_energy_cost * self.pue / self.j2mwh,
            "Total Profit": self.success_request_num / 1000 * 0.02,
        }

        # Add action statistics
        summary["Action Statistics"] = {}
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
        print("\nSummary:")
        summary = self.get_final_summary()

        # Print basic statistics
        for key, value in summary.items():
            if key != "Action Statistics":
                if isinstance(value, float):
                    print(f"{key}: {value:.3f}")
                else:
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
        pass