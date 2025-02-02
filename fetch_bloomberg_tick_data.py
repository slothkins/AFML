import blpapi
from datetime import datetime, timedelta
import pandas as pd


# Bloomberg API session setup
def get_bloomberg_session():
    session_options = blpapi.SessionOptions()
    session_options.setServerHost("localhost")  # Bloomberg API host
    session_options.setServerPort(8194)  # Bloomberg API port
    session = blpapi.Session(session_options)

    if not session.start():
        print("Failed to start Bloomberg session.")
        return None

    if not session.openService("//blp/refdata"):
        print("Failed to open Bloomberg refdata service.")
        return None

    return session


# Function to fetch trade tick data for a single day
def get_tick_data(security, date, event_types=["TRADE"]):
    session = get_bloomberg_session()
    if session is None:
        return None

    # Access refdata service
    refDataService = session.getService("//blp/refdata")

    # Create IntradayTickRequest
    request = refDataService.createRequest("IntradayTickRequest")

    # Set request parameters
    request.set("security", security)
    request.set("startDateTime", datetime(date.year, date.month, date.day, 0, 0).strftime("%Y-%m-%dT%H:%M:%S"))
    request.set("endDateTime", datetime(date.year, date.month, date.day, 23, 59, 59).strftime("%Y-%m-%dT%H:%M:%S"))

    # Add requested event types (e.g., TRADE)
    event_type_element = request.getElement("eventTypes")
    for event in event_types:
        event_type_element.appendValue(event)

    # Send request and process response
    session.sendRequest(request)
    data = []

    while True:
        event = session.nextEvent()
        for msg in event:
            if msg.messageType() == blpapi.Name("IntradayTickResponse"):
                # Extract each tick
                tick_data = msg.getElement("tickData").getElement("tickData")
                for tick in tick_data.values():
                    # Collect relevant fields: time, type, size, and price
                    time = tick.getElementAsDatetime("time")
                    type_ = tick.getElementAsString("type")
                    size = tick.getElementAsInteger("size") if tick.hasElement("size") else None
                    value = tick.getElementAsFloat("value") if tick.hasElement("value") else None

                    data.append({
                        "time": time,
                        "type": type_,
                        "price": value,
                        "size": size
                    })

        if event.eventType() == blpapi.Event.RESPONSE:  # Final event
            break

    # Stop the session
    session.stop()

    # Convert tick data into a DataFrame
    if data:
        return pd.DataFrame(data)
    else:
        print(f"No tick data received for {date}.")
        return None


def fetch_tick_data_between_dates(security, start_date, end_date, event_types=["TRADE"]):
    """Fetch tick data for each day in the date range."""
    current_date = start_date
    all_data = []

    while current_date <= end_date:
        print(f"Fetching data for {current_date.strftime('%Y-%m-%d')}...")
        daily_data = get_tick_data(security, current_date, event_types)
        if daily_data is not None:
            all_data.append(daily_data)
        current_date += timedelta(days=1)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        print("No tick data fetched for the specified date range.")
        return None


def save_to_hdf5(df, filename="tick_data.h5", key="tick_data"):
    """Saves DataFrame to HDF5 format"""
    df.to_hdf(filename, key=key, mode="w", format="table", data_columns=True)
    print(f"Data saved to {filename}")


# Example usage
if __name__ == "__main__":
    security = "ESZ4 Index"  # Example ticker
    start_date = datetime(2024, 12, 15)  # Start date
    end_date = datetime(2024, 12, 19)  # End date

    # Fetch tick data (trade prices and sizes) for the date range
    df = fetch_tick_data_between_dates(security, start_date, end_date)
    if df is not None:
        print(df)
        df.to_csv("ESZ4_tick_data.csv", index=False)  # Save to CSV for further analysis
        save_to_hdf5(df)
    else:
        print("Failed to fetch tick data.")