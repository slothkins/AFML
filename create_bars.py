import pandas as pd
from tqdm import tqdm


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

# Load tick data
df = pd.read_hdf("tick_data/concat_sorted.h5", key="concat_sorted", mode="r")
# Generate tick bars
tick_bars_df = create_tick_bars(df, tick_threshold=10000)

# Save tick bars to HDF5
tick_bars_df.to_hdf("tick_data/bars/tick_bars.h5", key="tick_bars", mode="w", format="table")

# Generate volume bars
volume_bars_df = create_volume_bars(df, volume_threshold=100000)

# Save volume bars to HDF5
volume_bars_df.to_hdf("tick_data/bars/volume_bars.h5", key="volume_bars", mode="w", format="table")