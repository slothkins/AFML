import mplfinance as mpf
import pandas as pd


def load_and_plot_bars(file_path, key, title):
    df = pd.read_hdf(file_path, key=key)
    print(f"{key.replace('_', ' ').title()} DataFrame size: {df.shape}")
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    mpf.plot(
        df,
        type="candle",
        style="yahoo",
        title=title,
        mav=(5, 10),
        volume=True
    )


# Load and plot tick bars
load_and_plot_bars(
    file_path="tick_data/bars/tick_bars.h5",
    key="tick_bars",
    title="Tick Bars Candlestick Chart"
)

# Load and plot volume bars
load_and_plot_bars(
    file_path="tick_data/bars/volume_bars.h5",
    key="volume_bars",
    title="Volume Bars Candlestick Chart"
)
