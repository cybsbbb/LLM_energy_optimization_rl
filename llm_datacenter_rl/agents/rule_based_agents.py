import numpy as np
from typing import Dict, Tuple, Optional, Any

from .base_agent import BaseAgent


class AllFullKVAgent(BaseAgent):
    """Agent that always uses full KV cache."""

    def __init__(self, enable_deny: bool = True):
        super().__init__("All FullKV", enable_deny)

    def predict(self, observation: Dict, deterministic: bool = True) -> int:
        return 0  # fullkv


class AllSnapKV64Agent(BaseAgent):
    """Agent that always uses maximum compression."""

    def __init__(self, enable_deny: bool = True):
        super().__init__("All SnapKV-64", enable_deny)

    def predict(self, observation: Dict, deterministic: bool = True) -> int:
        return 1  # snapkv_64


class RuleBasedPriceAgent(BaseAgent):
    """Agent that switches KV cache based on energy price."""

    def __init__(self, enable_deny: bool = True):
        super().__init__("Rule-based (Price)", enable_deny)

    def predict(self, observation: Dict, deterministic: bool = True) -> int:
        energy_price = observation['energy_price'][0]
        approx_price = energy_price * 100.0  # Denormalize
        if approx_price < 20:
            return 0  # fullkv
        elif approx_price < 40:
            return 5  # snapkv_1024
        elif approx_price < 60:
            return 3  # snapkv_256
        else:
            return 1  # snapkv_64


class SmartDenyAgent(BaseAgent):
    """Agent that uses deny strategically when system is overloaded."""

    def __init__(self, enable_deny: bool = True):
        super().__init__("Smart Deny", enable_deny)

    def predict(self, observation: Dict, deterministic: bool = True) -> int:
        processing_ratio = observation['processing_ratio'][0]
        waiting_ratio = observation['waiting_ratio'][0]
        energy_price = observation['energy_price'][0]

        # Only deny if enabled
        if self.enable_deny:
            if waiting_ratio > 0.6:
                return 6  # deny

        # Otherwise use price-based strategy
        approx_price = energy_price * 100.0  # Denormalize
        if approx_price < 20:
            return 0  # fullkv
        elif approx_price < 40:
            return 5  # snapkv_1024
        elif approx_price < 60:
            return 3  # snapkv_256
        else:
            return 1  # snapkv_64


class RandomAgent(BaseAgent):
    """Agent that takes random actions."""

    def __init__(self, enable_deny: bool = True):
        super().__init__("Random", enable_deny)
        self.n_actions = 7 if enable_deny else 6

    def predict(self, observation: Dict, deterministic: bool = True) -> int:
        return np.random.randint(0, self.n_actions)


# Factory function for baseline agents
def get_baseline_agent(name: str, enable_deny: bool = True) -> BaseAgent:
    """Get a baseline agent by name."""
    agents = {
        "all_fullkv": AllFullKVAgent,
        "all_snapkv_64": AllSnapKV64Agent,
        "rule_based_price": RuleBasedPriceAgent,
        "smart_deny": SmartDenyAgent,
        "random": RandomAgent,
    }

    if name not in agents:
        raise ValueError(f"Unknown baseline agent: {name}. Available: {list(agents.keys())}")

    return agents[name](enable_deny=enable_deny)