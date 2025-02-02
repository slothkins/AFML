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
    total_volume = 0
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
        total_volume += trade_volume

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


# Example usage
if __name__ == "__main__":
    # Example tick data
    tick_data = pd.DataFrame({
        "time": ["2024-12-19 09:00:01", "2024-12-19 09:00:03", "2024-12-19 09:00:05",
                 "2024-12-19 09:00:06", "2024-12-19 09:00:08", "2024-12-19 09:00:10"],
        "price": [100.5, 101.0, 100.8, 101.2, 100.7, 101.5],
        "size": [200, 300, 500, 100, 400, 600]
    })
    tick_data['time'] = pd.to_datetime(tick_data['time'])  # Convert to datetime format

    # Create volume bars with a threshold of 1000 units per bar
    volume_per_bar = 1000
    volume_bars = create_volume_bars(tick_data, volume_per_bar)

    # Display the result
    print(volume_bars)