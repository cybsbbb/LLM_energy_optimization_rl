from abc import ABC, abstractmethod
from typing import Dict
from ..utils.logger import get_logger


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, name: str, enable_deny: bool = True):
        self.name = name
        self.enable_deny = enable_deny
        self.logger = get_logger(__name__)

    @abstractmethod
    def predict(self, observation: Dict, deterministic: bool = True) -> int:
        """Make a decision based on observation."""
        pass

    def __str__(self):
        return self.name
