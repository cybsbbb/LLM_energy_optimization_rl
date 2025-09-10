import bisect
import random
import numpy as np
import pandas as pd
from typing import List
from abc import ABC, abstractmethod
from datetime import datetime

from ..utils.logger import get_logger
from ..config.config import EnvironmentConfig


class BaseDataProvider(ABC):
    """Abstract base class for data providers."""

    @abstractmethod
    def get_energy_price(self, current_time: datetime) -> float:
        """Get energy price at current time."""
        pass

    @abstractmethod
    def should_generate_request(self, current_time: datetime) -> bool:
        """Determine if a request should be generated. May reply on get_request_rate"""
        pass

    def get_request_rate(self, current_time: datetime) -> float:
        """Get request rate at current time."""
        return -1.0

    def get_energy_price_trend(self, current_time: datetime, lookback_minutes: int = 30) -> float:
        """Get energy price trend at current time."""
        return 0.0

    def get_request_rate_trend(self, current_time: datetime, lookback_minutes: int = 30) -> float:
        """Get request rate trend at current time."""
        return 0.0


class SyntheticDataProvider(BaseDataProvider):
    """Synthetic data provider using predefined patterns."""

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        if self.config.use_sample_energy_price:
            self.energy_price_df = pd.read_csv(self.config.sample_energy_price_file)
            self.energy_data = self.energy_price_df["VALUE"].tolist()
        else:
            self.energy_data = self._generate_energy_pattern()

    def _generate_energy_pattern(self) -> List[float]:
        """Generate synthetic energy price pattern."""
        # 24-hour pattern with higher prices during peak hours
        hours = np.arange(0, 24, 0.25)  # 15-minute intervals
        base_price = 50.0
        peak_hours = [8, 9, 10, 11, 18, 19, 20, 21]  # Peak hours

        prices = []
        for hour in hours:
            price = base_price
            if int(hour) in peak_hours:
                price *= 1.5
            # Add some randomness
            price *= (0.8 + 0.4 * random.random())
            prices.append(price)
        return prices

    def get_energy_price(self, current_time: datetime) -> float:
        """Get synthetic energy price."""
        hour = current_time.hour + current_time.minute / 60.0
        index = int((hour / 24.0) * len(self.energy_data))
        return self.energy_data[index % len(self.energy_data)]

    def should_generate_request(self, current_time: datetime) -> bool:
        """Bernoulli process for request generation."""
        return random.random() < self.config.bernoulli_prob


class RealisticDataProvider(BaseDataProvider):
    """Realistic data provider using real-world data files."""

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self._load_data()
        self._create_optimized_lookups()

    def _load_data(self):
        """Load real-world data files."""
        # Load invocations data
        self.invocations_df = pd.read_csv(self.config.invocations_file)
        self.invocations_df['Time'] = pd.to_datetime(self.invocations_df['Time'])
        self.invocations_df.set_index('Time', inplace=True)
        self.invocations_df.sort_index(inplace=True)

        # Load energy price data
        self.energy_df = pd.read_csv(self.config.energy_price_file)
        self.energy_df['INTERVALSTARTTIME_GMT'] = pd.to_datetime(
            self.energy_df['INTERVALSTARTTIME_GMT']
        )
        self.energy_df.set_index('INTERVALSTARTTIME_GMT', inplace=True)
        self.energy_df.sort_index(inplace=True)

        self.logger.info(f"Loaded {len(self.invocations_df)} invocation records")
        self.logger.info(f"Loaded {len(self.energy_df)} energy price records")

    def _create_optimized_lookups(self):
        """Create optimized lookup structures for O(log n) access."""
        # Energy price lookup
        self.energy_timestamps_ms = []
        self.energy_values = []

        for timestamp, row in self.energy_df.iterrows():
            timestamp_ms = int(timestamp.timestamp() * 1000)
            self.energy_timestamps_ms.append(timestamp_ms)
            self.energy_values.append(row['VALUE'])

        # Invocations lookup
        self.invocation_timestamps_ms = []
        self.invocation_values = []

        for timestamp, row in self.invocations_df.iterrows():
            timestamp_ms = int(timestamp.timestamp() * 1000)
            self.invocation_timestamps_ms.append(timestamp_ms)
            self.invocation_values.append(row['TIMESTAMP'])

        self.logger.info("Created optimized lookup structures")

    def get_energy_price(self, current_time: datetime) -> float:
        """Get energy price using optimized binary search."""
        current_time_ms = int(current_time.timestamp() * 1000)
        idx = bisect.bisect_right(self.energy_timestamps_ms, current_time_ms) - 1

        if idx >= 0:
            return self.energy_values[idx]
        else:
            return self.energy_values[0]

    def get_energy_price_trend(self, current_time: datetime, lookback_minutes: int = 30) -> float:
        """Calculate energy price trend over recent period."""
        current_time_ms = int(current_time.timestamp() * 1000)
        lookback_ms = lookback_minutes * 60 * 1000
        lookback_start_ms = current_time_ms - lookback_ms

        # Use binary search to find the range of relevant timestamps
        start_idx = bisect.bisect_left(self.energy_timestamps_ms, lookback_start_ms)
        end_idx = bisect.bisect_right(self.energy_timestamps_ms, current_time_ms) - 1

        if end_idx - start_idx < 2:
            return 0.0

        # Get the relevant data points
        times = np.arange(end_idx - start_idx)
        prices = np.array(self.energy_values[start_idx:end_idx])

        # Calculate linear trend
        trend = np.polyfit(times, prices, 1)[0]  # Slope of linear fit
        return np.clip(trend, -1, 1).item()  # Normalize

    def get_request_rate(self, current_time: datetime) -> float:
        """Get request rate using optimized binary search."""
        current_time_ms = int(current_time.timestamp() * 1000)
        idx = bisect.bisect_right(self.invocation_timestamps_ms, current_time_ms) - 1

        if idx >= 0:
            return self.invocation_values[idx] * self.config.request_scaling_factor
        else:
            return self.invocation_values[0] * self.config.request_scaling_factor

    def get_request_rate_trend(self, current_time: datetime, lookback_minutes: int = 30) -> float:
        """Calculate request rate trend over recent period."""
        current_time_ms = int(current_time.timestamp() * 1000)
        lookback_ms = lookback_minutes * 60 * 1000
        lookback_start_ms = current_time_ms - lookback_ms

        # Use binary search to find the range of relevant timestamps
        start_idx = bisect.bisect_left(self.invocation_timestamps_ms, lookback_start_ms)
        end_idx = bisect.bisect_right(self.invocation_timestamps_ms, current_time_ms) - 1

        if end_idx - start_idx < 2:
            return 0.0

        # Get the relevant data points
        times = np.arange(end_idx - start_idx)
        rates = np.array(self.invocation_values[start_idx: end_idx])

        # Calculate linear trend
        trend = np.polyfit(times, rates, 1)[0]  # Slope of linear fit
        trend = trend / 200.0
        return np.clip(trend, -1.0, 1.0).item()  # Normalize

    def should_generate_request(self, current_time: datetime) -> bool:
        """Poisson process based on real request rates."""
        request_rate = self.get_request_rate(current_time)
        requests_per_second = request_rate / 900.0  # just a normalization
        prob_request = min(1.0, requests_per_second / self.config.time_acceleration_factor)
        return random.random() < prob_request
