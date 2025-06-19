import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from heapq import heappush, heappop
from typing import Dict, Tuple, Optional, Any, List
from data_center_simulator.simulate_data.gov_report.gov_report_sim_dataset import GovReportSimDataset
from data_center_simulator.energy_price.caiso import data_20250421
from data_center_simulator.utils import setup_random, generate_bernoulli


class LLMDataCenterEnv(gym.Env):
    """
    OpenAI Gym compatible environment for LLM Data Center simulation.

    This environment uses a different approach where each step represents
    completing a batch of requests, providing aggregate rewards that better
    represent the impact of recent actions.
    """

    metadata = {'render_modes': ['human', 'ansi']}

    def __init__(self,
                 total_time: int = 24 * 3600 * 1000,  # 24 hours in milliseconds
                 time_interval: int = 10,  # Time interval in milliseconds
                 bernoulli_prob: float = 0.15,
                 server_num: int = 200,
                 max_wait_time: int = 10000,  # 10 seconds
                 pue: float = 1.3,
                 enable_deny: bool = True,
                 # Step configuration
                 step_time_ms: int = 10000,  # Time window per step in milliseconds (default 10 seconds)
                 # Reward values
                 success_reward: float = 0.5,
                 deny_penalty: float = -0.0,
                 timeout_penalty: float = -0.0,
                 # Reward weights
                 score_weight: float = 2.0,
                 latency_penalty_weight: float = 0.0,
                 energy_penalty_weight: float = 1.0,
                 render_mode: Optional[str] = None,
                 seed: Optional[int] = None,
                 # Logging configuration
                 enable_detailed_logging: bool = False):

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
        self.step_time_ms = step_time_ms  # Time window for each step
        self.enable_detailed_logging = enable_detailed_logging

        # Reward parameters
        self.success_reward = success_reward
        self.deny_penalty = deny_penalty
        self.timeout_penalty = timeout_penalty
        self.score_weight = score_weight
        self.latency_penalty_weight = latency_penalty_weight
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

        # Enhanced logging for action tracking
        self.step_log = []  # Store detailed step information
        self.reset_logging()

        # Initialize state
        if seed is not None:
            self.reset(seed=seed)

    def reset_logging(self):
        """Reset logging data."""
        self.step_log = []

    def log_step_details(self, action: int, obs: Dict, reward: float, info: Dict):
        """Log detailed step information for later analysis."""
        if self.enable_detailed_logging:
            step_info = {
                'time_ms': self.current_time,
                'time_hours': self.current_time / 1000 / 3600,
                'action': action,
                'action_name': self.action_to_kv_size[action],
                'energy_price': self._get_cur_energy_price(self.current_time),
                'processing_ratio': obs['processing_ratio'][0],
                'waiting_ratio': obs['waiting_ratio'][0],
                'recent_latency': obs['recent_latency'][0],
                'recent_score': obs['recent_score'][0],
                'reward': reward,
                'success_requests': info.get('success_requests', 0),
                'denied_requests': info.get('denied_requests', 0),
                'cumulative_reward': info.get('episode_reward', 0),
            }
            self.step_log.append(step_info)

    def get_step_log(self) -> List[Dict]:
        """Get the complete step log for this episode."""
        return self.step_log.copy()

    def _calculate_completion_reward(self, request: Dict, latency: float) -> float:
        """Calculate reward for a completed request based on actual outcomes."""
        # Base success reward
        reward = self.success_reward

        # Quality bonus (actual score)
        reward += request["score"] * self.score_weight

        # Latency penalty (actual latency)
        normalized_latency = min(latency / self.max_wait_time, 1.0)
        reward -= normalized_latency * self.latency_penalty_weight

        # Energy cost penalty (actual cost)
        energy_cost = request["energy"] * request["energy_price"] * self.pue / self.j2mwh
        normalized_energy = min(energy_cost / 0.0002, 1.0)
        reward -= normalized_energy * self.energy_penalty_weight

        return reward

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
        self.completed_requests_rewards = []  # Rewards from requests completed in this step

        # Reset logging
        self.reset_logging()

        # Initialize event queue
        heappush(self.event_queue, (0, random.random(), {"type": "timestamp"}))

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        Each step processes all requests within a fixed time window (step_time_ms)
        with the same action to provide meaningful feedback about the action's effectiveness.
        """
        self.episode_length += 1

        # Ensure action is an integer
        if isinstance(action, np.ndarray):
            action = int(action.item())
        elif isinstance(action, tuple):
            action = int(action[0].item())
        else:
            action = int(action)

        # Set the action for this step
        self.current_action = action
        self.action_counts[action] += 1

        # Initialize step tracking
        self.step_start_time = self.current_time
        step_end_time = min(self.step_start_time + self.step_time_ms, self.total_time)
        self.requests_in_current_step = 0
        self.completed_requests_rewards = []

        # Process all events within this time window
        while self.current_time < step_end_time:
            # Advance to next event
            if not self._advance_to_next_event(step_end_time):
                break

            # If it's a new request, process it with the current action
            if hasattr(self, '_pending_request') and self._pending_request:
                self._process_request_with_action(action)
                self.requests_in_current_step += 1
                self._pending_request = False

        # Calculate aggregate reward for this step
        # Always use completion-based rewards (from requests that finished in this step)
        if self.completed_requests_rewards:
            step_reward = sum(self.completed_requests_rewards) / len(self.completed_requests_rewards)
        else:
            step_reward = 0.0

        self.episode_reward += step_reward

        # Check if episode is done
        terminated = self.current_time >= self.total_time
        truncated = False

        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        info['requests_processed'] = self.requests_in_current_step
        info['step_duration_ms'] = self.current_time - self.step_start_time
        info['step_rewards'] = self.completed_requests_rewards.copy()

        # Log step details for analysis
        self.log_step_details(action, observation, step_reward, info)

        return observation, step_reward, terminated, truncated, info

    def _advance_to_next_event(self, max_time: int) -> bool:
        """
        Advance simulation to the next event, but not beyond max_time.
        Returns True if an event was processed, False if we reached max_time.
        """
        self._pending_request = False

        while len(self.event_queue) > 0:
            # Peek at next event
            next_time = self.event_queue[0][0]

            # Don't process events beyond the step boundary
            if next_time >= max_time:
                self.current_time = max_time
                return False

            # Process the event
            cur_time, _, cur_event = heappop(self.event_queue)
            self.current_time = cur_time

            if cur_event["type"] == "timestamp":
                # Handle timeouts
                self._handle_timeouts(cur_time)

                # Schedule next timestamp
                heappush(self.event_queue, (cur_time + self.time_interval, random.random(), cur_event))

                # Check if new request arrives
                if generate_bernoulli(self.bernoulli_prob):
                    # Mark that we have a pending request to process
                    self._pending_request = True
                    return True

            elif cur_event["type"] == "finish":
                self._handle_finish(cur_time, cur_event)
                # Continue to process more events

        # No more events
        self.current_time = max_time
        return False

    def _process_request_with_action(self, action: int):
        """Process a single request with the given action."""
        action_type = self.action_to_kv_size[action]

        if action_type == 'deny':
            # Deny the request
            self.denied_request_num += 1
            # Denials get immediate feedback since there's no completion
            self.completed_requests_rewards.append(self.deny_penalty)
        else:
            # Try to process the request
            new_request = self.gov_report_sim_dataset.get_random_item(action_type)
            new_request["energy_price"] = self._get_cur_energy_price(self.current_time)
            new_request["arrival_time"] = self.current_time
            new_request["action"] = action  # Store the action that created this request

            # Try to add to processing or queue
            if self.processing_server_num < self.server_num:
                # Can process immediately
                self._start_processing(new_request)
                # No reward yet - will get it on completion
            elif len(self.waiting_queue) < self.server_num * 5:
                # Add to queue
                heappush(self.waiting_queue, (self.current_time, new_request))
                # No reward yet - will get it on completion or timeout
            else:
                # System overloaded, request failed immediately
                self.failed_request_num += 1
                # Immediate failure gets immediate feedback
                self.completed_requests_rewards.append(self.timeout_penalty)

    def _start_processing(self, request: Dict):
        """Start processing a request."""
        request_time = round(request["time"] * 1000)
        finish_time = self.current_time + request_time
        self.processing_server_num += 1
        request["processing_start_time"] = self.current_time
        heappush(self.event_queue, (finish_time, random.random(), {"type": "finish", "request": request}))

    def _handle_timeouts(self, cur_time: int):
        """Remove requests that waited too long."""
        while len(self.waiting_queue) > 0 and self.waiting_queue[0][0] < cur_time - self.max_wait_time:
            arrival_time, request = self.waiting_queue.pop(0)
            self.timeout_request_num += 1

            # Timeout penalty
            if hasattr(self, 'completed_requests_rewards'):
                self.completed_requests_rewards.append(self.timeout_penalty)

    def _handle_finish(self, cur_time: int, cur_event: Dict):
        """Handle request completion."""
        self.success_request_num += 1
        self.processing_server_num -= 1

        request = cur_event["request"]

        # Calculate actual latency if the request waited
        if "arrival_time" in request:
            actual_latency = request.get("processing_start_time", cur_time) - request["arrival_time"]
        else:
            actual_latency = 0

        # Update metrics
        score = request["score"]
        self.total_score += score
        self.score_count += 1

        # Update recent scores
        self.recent_scores.append(score)
        if len(self.recent_scores) > 100:
            self.recent_scores.pop(0)

        self.tot_energy += request["energy"]
        self.tot_energy_cost += request["energy"] * request["energy_price"]

        # Calculate completion reward based on actual outcomes
        if hasattr(self, 'completed_requests_rewards'):
            reward = self._calculate_completion_reward(request, actual_latency)
            self.completed_requests_rewards.append(reward)

        # Process next waiting request if any
        if len(self.waiting_queue) > 0:
            arrival_time, waiting_request = heappop(self.waiting_queue)

            # Calculate actual latency
            latency = cur_time - arrival_time
            self.total_latency += latency
            self.latency_count += 1

            # Update recent latencies
            self.recent_latencies.append(latency)
            if len(self.recent_latencies) > 100:
                self.recent_latencies.pop(0)

            # Start processing
            waiting_request["processing_start_time"] = cur_time
            self._start_processing(waiting_request)

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

        recent_score = 0.5  # Default
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
        total_requests = self.success_request_num + self.failed_request_num + self.denied_request_num + self.timeout_request_num

        info = {
            'success_requests': self.success_request_num,
            'failed_requests': self.failed_request_num,
            'denied_requests': self.denied_request_num,
            'timeout_requests': self.timeout_request_num,
            'total_energy': self.tot_energy,
            'total_energy_cost': self.tot_energy_cost,
            'current_time': self.current_time,
            'episode_reward': self.episode_reward,
            'episode_length': self.episode_length,
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
            print(f"\n--- Time: {self.current_time / 1000:.1f}s ---")
            print(f"Processing: {self.processing_server_num}/{self.server_num}")
            print(f"Waiting: {len(self.waiting_queue)}")
            print(
                f"Success/Failed/Denied/Timeout: {self.success_request_num}/{self.failed_request_num}/{self.denied_request_num}/{self.timeout_request_num}")
            if self.recent_scores:
                print(f"Recent Score: {sum(self.recent_scores) / len(self.recent_scores):.3f}")
            print(f"Energy Price: ${self._get_cur_energy_price(self.current_time):.2f}")
            if self.current_action is not None:
                print(f"Current Action: {self.action_names[self.current_action]}")

    def get_final_summary(self) -> Dict:
        """Get final summary statistics."""
        total_requests = self.success_request_num + self.failed_request_num + self.denied_request_num + self.timeout_request_num

        summary = {"Total Success Requests": self.success_request_num, "Total Failed Requests": self.failed_request_num,
                   "Total Denied Requests": self.denied_request_num, "Total Timeout Requests": self.timeout_request_num,
                   "Success Rate": self.success_request_num / total_requests if total_requests > 0 else 0,
                   "Denial Rate": self.denied_request_num / total_requests if total_requests > 0 else 0,
                   "Timeout Rate": self.timeout_request_num / total_requests if total_requests > 0 else 0,
                   "Average Latency (s)": (self.total_latency / self.latency_count) / 1000 if self.latency_count > 0 else 0,
                   "Average Score": (self.total_score / self.score_count) * 100 if self.score_count > 0 else 0,
                   "Total Energy (J)": self.tot_energy,
                   "Total Energy Cost": self.tot_energy_cost * self.pue / self.j2mwh,
                   "Total Profit": self.success_request_num / 1000 * 0.02, "Action Statistics": {}}

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