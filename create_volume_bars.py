import pandas as pd


def create_volume_bars(data, volume_per_bar):
    """
    Creates volume bars from tick data.

    :param data: DataFrame containing tick data with 'time', 'price', and 'size' (volume for each trade).
    :param volume_per_bar: Total volume threshold per bar (e.g., 1000).
    :return: DataFrame with volume bars.
    """
    # Ensure the data is sorted by time
    data = data.sort_values(by="time").reset_index(drop=True)

    # Initialize variables
    volume_accumulated = 0
    price_sum_weighted = 0
    bar_start_time = None
    bars = []

    for i, row in data.iterrows():
        trade_volume = row['size']  # Size of the trade
        trade_price = row['price']  # Trade price
        trade_time = row['time']  # Trade timestamp

        # If first trade in the bar, set start time
        if volume_accumulated == 0:
            bar_start_time = trade_time

        # Accumulate weighted prices and volume
        price_sum_weighted += trade_price * trade_volume
        volume_accumulated += trade_volume

        # Check if volume threshold is reached
        if volume_accumulated >= volume_per_bar:
            # Calculate average price for the bar
            avg_price = price_sum_weighted / volume_accumulated

            # Create the volume bar
            bars.append({
                "start_time": bar_start_time,
                "end_time": trade_time,
                "volume": volume_accumulated,
                "average_price": avg_price
            })

            # Reset bar variables for the next bar
            volume_accumulated = 0
            price_sum_weighted = 0

    # Convert the list of bars into a DataFrame
    bars_df = pd.DataFrame(bars)
    return bars_df


# Main code to process tick_data.h5
if __name__ == "__main__":
    # Load tick data from HDF5 file
    filename = "tick_data.h5"
    key = "tick_data"  # Adjust if the key in the HDF5 file is different

    try:
        print(f"Loading data from {filename}...")
        tick_data = pd.read_hdf(filename, key=key)

        # Check that the required columns are present
        required_columns = ['time', 'price', 'size']
        if not all(column in tick_data.columns for column in required_columns):
            print(f"Error: The file {filename} does not contain required columns: {required_columns}")
        else:
            # Ensure the 'time' column is in datetime format
            tick_data['time'] = pd.to_datetime(tick_data['time'])

            # Define the volume threshold per bar
            volume_per_bar = 20000  # Example: 1000 units of volume per bar

            # Create volume bars
            print(f"Creating volume bars with {volume_per_bar} units per bar...")
            volume_bars = create_volume_bars(tick_data, volume_per_bar)

            # Save the volume bars to an HDF5 file
            output_filename = "volume_bars.h5"
            volume_bars.to_hdf(output_filename, key="volume_bars", mode="w", format="table", data_columns=True)
            print(f"Volume bars saved to {output_filename}.")

            # Optional: Display a preview of the volume bars
            print(volume_bars.head())
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")