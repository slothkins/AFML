import pandas as pd


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

    for i, row in df.iterrows():
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


# Load tick data
df = pd.read_hdf("tick_data.h5", key="tick_data")

# Generate tick bars
tick_bars_df = create_tick_bars(df, tick_threshold=100)

# Save tick bars to HDF5
tick_bars_df.to_hdf("tick_bars.h5", key="tick_bars", mode="w", format="table")

# Display the tick bars
import ace_tools_open as tools

tools.display_dataframe_to_user(name="Tick Bars Data", dataframe=tick_bars_df)
