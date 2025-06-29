import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from typing import Optional, Union
import bisect


class EnergyPriceProcessor:
    """
    A class to preprocess energy price data and provide efficient timestamp-based queries.
    """

    def __init__(self):
        self.data = None
        self.timestamp_index = None
        self.price_values = None
        self.start_times = None
        self.end_times = None

    def preprocess_csv(self, csv_file_path: str, save_preprocessed: bool = True,
                       preprocessed_file_path: str = "energy_price_data.pkl"):
        """
        Preprocess the CSV file for efficient querying.

        Args:
            csv_file_path: Path to the input CSV file
            save_preprocessed: Whether to save preprocessed data to disk
            preprocessed_file_path: Path to save preprocessed data
        """
        print("Loading CSV file...")

        # Read the CSV file
        df = pd.read_csv(csv_file_path)

        print(f"Loaded {len(df)} records")

        # Filter to only include LMP_PRC records
        df = df[df['XML_DATA_ITEM'] == 'LMP_PRC'].copy()

        print(f"After filtering for LMP_PRC: {len(df)} records")

        if len(df) == 0:
            raise ValueError("No LMP_PRC records found in the CSV file. Please check the XML_DATA_ITEM column.")

        # Convert timestamp columns to datetime
        df['INTERVALSTARTTIME_GMT'] = pd.to_datetime(df['INTERVALSTARTTIME_GMT'])
        df['INTERVALENDTIME_GMT'] = pd.to_datetime(df['INTERVALENDTIME_GMT'])

        # Sort by start time to ensure chronological order
        df = df.sort_values('INTERVALSTARTTIME_GMT').reset_index(drop=True)

        # Store the relevant data
        self.data = df[['INTERVALSTARTTIME_GMT', 'INTERVALENDTIME_GMT', 'VALUE']].copy()

        # Create efficient lookup structures
        self.start_times = self.data['INTERVALSTARTTIME_GMT'].values
        self.end_times = self.data['INTERVALENDTIME_GMT'].values
        self.price_values = self.data['VALUE'].values

        # Convert to timestamps (seconds since epoch) for faster binary search
        # Handle both pandas Timestamp and numpy datetime64 objects
        self.timestamp_index = []
        for ts in self.start_times:
            if hasattr(ts, 'timestamp'):
                # pandas Timestamp
                self.timestamp_index.append(ts.timestamp())
            else:
                # numpy datetime64 - convert to pandas Timestamp first
                self.timestamp_index.append(pd.Timestamp(ts).timestamp())

        print(f"Preprocessed data covering {self.start_times[0]} to {self.end_times[-1]}")

        if save_preprocessed:
            self.save_preprocessed_data(preprocessed_file_path)

    def save_preprocessed_data(self, file_path: str):
        """Save preprocessed data to disk."""
        data_to_save = {
            'data': self.data,
            'timestamp_index': self.timestamp_index,
            'price_values': self.price_values,
            'start_times': self.start_times,
            'end_times': self.end_times
        }

        with open(file_path, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"Preprocessed data saved to {file_path}")

    def load_preprocessed_data(self, file_path: str):
        """Load preprocessed data from disk."""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            self.data = data['data']
            self.timestamp_index = data['timestamp_index']
            self.price_values = data['price_values']
            self.start_times = data['start_times']
            self.end_times = data['end_times']

            print(f"Preprocessed data loaded from {file_path}")
            print(f"Data covers {self.start_times[0]} to {self.end_times[-1]}")
            return True

        except FileNotFoundError:
            print(f"Preprocessed file {file_path} not found. Please preprocess the CSV first.")
            return False
        except Exception as e:
            print(f"Error loading preprocessed data: {e}")
            return False

    def query_price_by_timestamp(self, timestamp: Union[str, datetime, pd.Timestamp]) -> Optional[float]:
        """
        Query price for a specific timestamp using binary search for O(log n) complexity.

        Args:
            timestamp: Timestamp to query (string, datetime, or pandas Timestamp)

        Returns:
            Price value if found, None if not found
        """
        if self.timestamp_index is None:
            raise ValueError("Data not loaded. Please preprocess CSV or load preprocessed data first.")

        # Convert input to pandas Timestamp for consistency
        if isinstance(timestamp, str):
            query_ts = pd.to_datetime(timestamp)
        elif isinstance(timestamp, datetime):
            query_ts = pd.Timestamp(timestamp)
        else:
            query_ts = pd.Timestamp(timestamp)

        query_ts_float = query_ts.timestamp()

        # Binary search to find the appropriate interval
        # We want the interval where start_time <= query_time < end_time
        idx = bisect.bisect_right(self.timestamp_index, query_ts_float) - 1

        if idx >= 0 and idx < len(self.start_times):
            # Check if the timestamp falls within this interval
            start_ts = pd.Timestamp(self.start_times[idx])
            end_ts = pd.Timestamp(self.end_times[idx])
            if (start_ts <= query_ts < end_ts):
                return self.price_values[idx]

        return None

    def query_price_range(self, start_time: Union[str, datetime, pd.Timestamp],
                          end_time: Union[str, datetime, pd.Timestamp]) -> pd.DataFrame:
        """
        Query prices for a time range.

        Args:
            start_time: Start of the query range
            end_time: End of the query range

        Returns:
            DataFrame with timestamps and prices in the specified range
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please preprocess CSV or load preprocessed data first.")

        # Convert inputs to pandas Timestamps
        start_ts = pd.to_datetime(start_time)
        end_ts = pd.to_datetime(end_time)

        # Filter data within the range
        mask = (self.data['INTERVALSTARTTIME_GMT'] >= start_ts) & \
               (self.data['INTERVALSTARTTIME_GMT'] < end_ts)

        return self.data[mask][['INTERVALSTARTTIME_GMT', 'INTERVALENDTIME_GMT', 'VALUE']].copy()

    def get_statistics(self) -> dict:
        """Get basic statistics about the loaded data."""
        if self.price_values is None:
            raise ValueError("Data not loaded.")

        return {
            'total_records': len(self.price_values),
            'min_price': np.min(self.price_values),
            'max_price': np.max(self.price_values),
            'mean_price': np.mean(self.price_values),
            'std_price': np.std(self.price_values),
            'start_date': self.start_times[0],
            'end_date': self.end_times[-1]
        }

if __name__ == "__main__":
    # Initialize the processor
    processor = EnergyPriceProcessor()

    # Option 1: Preprocess from CSV
    processor.preprocess_csv(csv_file_path='../data/energy_price/raw/CAISO_20240509_20240521.csv',
                             save_preprocessed=True,
                             preprocessed_file_path='../data/energy_price/processed/CAISO_20240509_20240521.pkl')

    # Option 2: Load preprocessed data (faster for subsequent runs)
    processor.load_preprocessed_data('../data/energy_price/processed/CAISO_20240509_20240521.pkl')

    # Example queries (uncomment after loading data)
    print("\n=== Example Queries ===")

    # Query specific timestamp
    price = processor.query_price_by_timestamp('2024-05-09 07:02:23.009930')
    print(f"Price at 2024-05-09 07:02:23: {price}")

    # Get statistics
    stats = processor.get_statistics()
    print(f"\nData Statistics:\n{stats}")

