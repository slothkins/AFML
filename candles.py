import mplfinance as mpf
import pandas as pd

# Load the tick bars data (assumed to be saved as HDF5 previously)
tick_bars_df = pd.read_hdf("tick_bars.h5", key="tick_bars")

# Convert the "time" column to datetime if it's not already
tick_bars_df["time"] = pd.to_datetime(tick_bars_df["time"])

# Set the "time" column as the index (required by mplfinance)
tick_bars_df.set_index("time", inplace=True)

# Plot the candlestick chart
mpf.plot(
    tick_bars_df,
    type="candle",  # Use candlestick style
    style="yahoo",  # Predefined style
    title="Tick Bars Candlestick Chart",
    mav=(5, 10),  # Add moving averages with window sizes (5 and 10)
    volume=True  # Show volume plot
)