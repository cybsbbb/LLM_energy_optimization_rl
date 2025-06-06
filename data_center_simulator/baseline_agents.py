"""
Baseline agents for LLM Data Center environment.
These agents implement simple rule-based policies for comparison with RL agents.
"""

import numpy as np
from typing import Dict


class BaselineAgent:
    """Base class for baseline agents."""
    
    def __init__(self, name: str, enable_deny: bool = True):
        self.name = name
        self.enable_deny = enable_deny

    def predict(self, observation: Dict, deterministic: bool = True) -> int:
        """Make a decision based on observation."""
        raise NotImplementedError

    def __str__(self):
        return self.name


class AllFullKVAgent(BaselineAgent):
    """Agent that always uses full KV cache (best quality, highest energy)."""

    def __init__(self, enable_deny: bool = True):
        super().__init__("All FullKV", enable_deny)

    def predict(self, observation: Dict, deterministic: bool = True) -> int:
        return 0  # fullkv


class AllSnapKV64Agent(BaselineAgent):
    """Agent that always uses maximum compression (lowest energy, reduced quality)."""

    def __init__(self, enable_deny: bool = True):
        super().__init__("All SnapKV-64", enable_deny)

    def predict(self, observation: Dict, deterministic: bool = True) -> int:
        return 1  # snapkv_64


class RuleBasedPriceAgent(BaselineAgent):
    """Agent that switches KV cache based on energy price thresholds."""

    def __init__(self, enable_deny: bool = True):
        super().__init__("Rule-based (Price)", enable_deny)

    def predict(self, observation: Dict, deterministic: bool = True) -> int:
        # Extract energy price (already normalized)
        energy_price = observation['energy_price'][0] * 100  # Denormalize

        if energy_price < 20:
            return 0  # fullkv
        elif energy_price < 40:
            return 5  # snapkv_1024
        elif energy_price < 60:
            return 3  # snapkv_256
        else:
            return 1  # snapkv_64


class SmartDenyAgent(BaselineAgent):
    """Agent that uses deny strategically when system is overloaded."""

    def __init__(self, enable_deny: bool = True):
        super().__init__("Smart Deny", enable_deny)

    def predict(self, observation: Dict, deterministic: bool = True) -> int:
        processing_ratio = observation['processing_ratio'][0]
        waiting_ratio = observation['waiting_ratio'][0]
        energy_price = observation['energy_price'][0] * 100  # Denormalize

        # Only deny if enabled
        if self.enable_deny:
            # Deny if system is heavily loaded
            if processing_ratio > 0.95 and waiting_ratio > 0.5:
                return 6  # deny
            elif processing_ratio > 0.9 and waiting_ratio > 0.8:
                return 6  # deny

        # Otherwise use price-based strategy
        if energy_price < 20:
            return 0  # fullkv
        elif energy_price < 40:
            return 5  # snapkv_1024
        elif energy_price < 60:
            return 3  # snapkv_256
        else:
            return 1  # snapkv_64


class AdaptiveLoadAgent(BaselineAgent):
    """Agent that adapts both compression and denial based on system load."""

    def __init__(self, enable_deny: bool = True):
        super().__init__("Adaptive Load", enable_deny)

    def predict(self, observation: Dict, deterministic: bool = True) -> int:
        processing_ratio = observation['processing_ratio'][0]
        waiting_ratio = observation['waiting_ratio'][0]
        energy_price = observation['energy_price'][0] * 100  # Denormalize
        recent_latency = observation['recent_latency'][0]

        # Only use deny if enabled
        if self.enable_deny:
            # Aggressive denial if latency is high
            if recent_latency > 0.7:
                if processing_ratio > 0.8 or waiting_ratio > 0.5:
                    return 6  # deny

            # Standard denial thresholds
            if processing_ratio > 0.95 and waiting_ratio > 0.5:
                return 6  # deny

        # Adapt compression based on load and price
        load_factor = processing_ratio * 0.7 + waiting_ratio * 0.3

        if load_factor < 0.5:
            # Low load - prioritize quality
            if energy_price < 80:
                return 0  # fullkv
            else:
                return 3  # snapkv_256
        elif load_factor < 0.8:
            # Medium load - balance
            if energy_price < 60:
                return 3  # snapkv_256
            else:
                return 2  # snapkv_128
        else:
            # High load - prioritize throughput
            if energy_price < 50:
                return 2  # snapkv_128
            else:
                return 1  # snapkv_64


class RandomAgent(BaselineAgent):
    """Agent that takes random actions (for comparison)."""

    def __init__(self, enable_deny: bool = True):
        super().__init__("Random", enable_deny)
        self.n_actions = 7 if enable_deny else 6

    def predict(self, observation: Dict, deterministic: bool = True) -> int:
        if deterministic:
            # Use a fixed seed for reproducibility in evaluation
            # np.random.seed(42)
            pass
        return np.random.randint(0, self.n_actions)


# Factory function to get agent by name
def get_baseline_agent(name: str, enable_deny: bool = True) -> BaselineAgent:
    """Get a baseline agent by name."""
    agents = {
        "all_fullkv": AllFullKVAgent,
        "all_snapkv_64": AllSnapKV64Agent,
        "rule_based_price": RuleBasedPriceAgent,
        "rule_based_deny": SmartDenyAgent,
        "adaptive_load": AdaptiveLoadAgent,
        "random": RandomAgent,
    }

    if name not in agents:
        raise ValueError(f"Unknown agent: {name}. Available: {list(agents.keys())}")

    return agents[name](enable_deny=enable_deny)