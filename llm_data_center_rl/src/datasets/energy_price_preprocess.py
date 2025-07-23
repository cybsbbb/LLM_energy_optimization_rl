import pandas as pd
import matplotlib.pyplot as plt


def preprocess_csv(csv_file_path: str,
                   preprocessed_file_path: str = "energy_price_data.csv",
                   plot_file_path: str = "energy_price_data.png"):
    """
        Preprocess the CSV file.
    Args:
        csv_file_path: Path to the input CSV file
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
    data = df[['INTERVALSTARTTIME_GMT', 'INTERVALENDTIME_GMT', 'VALUE']].copy()

    data.to_csv(preprocessed_file_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
    # save plot
    data.plot(x='INTERVALSTARTTIME_GMT', y='VALUE', grid=True, figsize=(15, 5))
    plt.ylabel("Energy Price ($)")
    plt.tight_layout()
    plt.savefig(plot_file_path)
    plt.close()


if __name__ == "__main__":
    # 2023
    preprocess_csv(csv_file_path='../../data/energy_price/raw/CAISO_20231115_20231117.csv',
                   preprocessed_file_path='../../data/energy_price/processed/CAISO_20231115_20231117.csv',
                   plot_file_path='../../data/energy_price/processed/CAISO_20231115_20231117.png')
    # 2024
    preprocess_csv(csv_file_path='../../data/energy_price/raw/CAISO_20240509_20240521.csv',
                   preprocessed_file_path='../../data/energy_price/processed/CAISO_20240509_20240521.csv',
                   plot_file_path='../../data/energy_price/processed/CAISO_20240509_20240521.png')
