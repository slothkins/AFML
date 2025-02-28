import pandas as pd
from tqdm import tqdm
#
# # Save dollar bars to HDF5
# dollar_bars_filename = f"tick_data/bars/dollar_bars_{dollar_threshold}.h5"
# dollar_bars_df.to_hdf(dollar_bars_filename, key="dollar_bars", mode="w", format="table")

tick_threshold = 5_000
volume_threshold = 50_000
dollar_threshold = 250_000_000  # 50000 contracts is approximately 50,000* spx level e.g. 5000 = 250,000,000


# Generate dollar bars
def create_dollar_bars(df, dollar_threshold=250_000_000):
    """
    Creates dollar bars from tick data based on a fixed cumulative dollar value per bar.

    Parameters:
        df (pd.DataFrame): Tick data with columns ['time', 'type', 'price', 'size']
        dollar_threshold (float): Cumulative dollar value per bar

    Returns:
        pd.DataFrame: Dollar bars with OHLC and volume
    """
    bars = []
    cumulative_dollar_value = 0
    ohlc = {"open": None, "high": None, "low": None, "close": None, "volume": 0}

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Creating Dollar Bars"):
        price = row["price"]
        volume = row["size"]
        dollar_value = price * volume

        if cumulative_dollar_value == 0:
            ohlc["open"] = price  # Set Open price

        # Update OHLC values
        ohlc["high"] = max(ohlc["high"], price) if ohlc["high"] is not None else price
        ohlc["low"] = min(ohlc["low"], price) if ohlc["low"] is not None else price
        ohlc["close"] = price  # Update Close price
        ohlc["volume"] += volume  # Accumulate volume

        cumulative_dollar_value += dollar_value

        # If threshold is met, finalize the bar
        if cumulative_dollar_value >= dollar_threshold:
            bars.append([row["time"], ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"], ohlc["volume"]])
            cumulative_dollar_value = 0  # Reset counter
            ohlc = {"open": None, "high": None, "low": None, "close": None, "volume": 0}  # Reset OHLC data

    # Convert to DataFrame
    dollar_bars_df = pd.DataFrame(bars, columns=["time", "open", "high", "low", "close", "volume"])
    return dollar_bars_df


# Generate tick bars
def create_tick_bars(df, tick_threshold=10000):
    """
    Creates tick bars from tick data based on a fixed number of ticks per bar.

    Parameters:
        df (pd.DataFrame): Tick data with columns ['time', 'type', 'price', 'size']
        tick_threshold (int): Number of ticks per bar

    Returns:
        pd.DataFrame: Tick bars with OHLC and volume
    """
    bars = []
    tick_count = 0
    ohlc = {"open": None, "high": None, "low": None, "close": None, "volume": 0}

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Creating Tick Bars"):
        price = row["price"]
        volume = row["size"]

        if tick_count == 0:
            ohlc["open"] = price  # Set Open price

        # Update OHLC values
        ohlc["high"] = max(ohlc["high"], price) if ohlc["high"] is not None else price
        ohlc["low"] = min(ohlc["low"], price) if ohlc["low"] is not None else price
        ohlc["close"] = price  # Update Close price
        ohlc["volume"] += volume  # Accumulate volume

        tick_count += 1

        # If threshold is met, finalize the bar
        if tick_count >= tick_threshold:
            bars.append([row["time"], ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"], ohlc["volume"]])
            tick_count = 0  # Reset counter
            ohlc = {"open": None, "high": None, "low": None, "close": None, "volume": 0}  # Reset OHLC data

    # Convert to DataFrame
    tick_bars_df = pd.DataFrame(bars, columns=["time", "open", "high", "low", "close", "volume"])
    return tick_bars_df

# Generate volume bars
def create_volume_bars(df, volume_threshold=200000):
    """
    Creates volume bars from tick data based on a fixed cumulative volume per bar.

    Parameters:
        df (pd.DataFrame): Tick data with columns ['time', 'type', 'price', 'size']
        volume_threshold (int): Cumulative volume per bar

    Returns:
        pd.DataFrame: Volume bars with OHLC and volume
    """
    bars = []
    cumulative_volume = 0
    ohlc = {"open": None, "high": None, "low": None, "close": None, "volume": 0}

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Creating Volume Bars"):
        price = row["price"]
        volume = row["size"]

        if cumulative_volume == 0:
            ohlc["open"] = price  # Set Open price

        # Update OHLC values
        ohlc["high"] = max(ohlc["high"], price) if ohlc["high"] is not None else price
        ohlc["low"] = min(ohlc["low"], price) if ohlc["low"] is not None else price
        ohlc["close"] = price  # Update Close price
        ohlc["volume"] += volume  # Accumulate volume

        cumulative_volume += volume

        # If threshold is met, finalize the bar
        if cumulative_volume >= volume_threshold:
            bars.append([row["time"], ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"], ohlc["volume"]])
            cumulative_volume = 0  # Reset counter
            ohlc = {"open": None, "high": None, "low": None, "close": None, "volume": 0}  # Reset OHLC data

    # Convert to DataFrame
    volume_bars_df = pd.DataFrame(bars, columns=["time", "open", "high", "low", "close", "volume"])
    return volume_bars_df






# Load sorted tick data
df = pd.read_hdf("tick_data/concat_sorted.h5", key="concat_sorted", mode="r")

# Generate tick bars from tick data
tick_bars_df = create_tick_bars(df, tick_threshold=tick_threshold)

# Save tick bars to HDF5
tick_bars_filename = f"tick_data/bars/tick_bars_{tick_threshold}.h5"
tick_bars_df.to_hdf(tick_bars_filename, key="tick_bars", mode="w", format="table")

# Generate volume bars from tick data
volume_bars_df = create_volume_bars(df, volume_threshold=volume_threshold)

# Generate dollar bars from tick data
dollar_bars_df = create_dollar_bars(df, dollar_threshold=dollar_threshold)

# Save volume bars to HDF5
volume_bars_filename = f"tick_data/bars/volume_bars_{volume_threshold}.h5"
volume_bars_df.to_hdf(volume_bars_filename, key="volume_bars", mode="w", format="table")
