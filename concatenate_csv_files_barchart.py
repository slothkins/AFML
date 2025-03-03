import pandas as pd
import os
import tables
from pygments.util import duplicates_removed


def concat_csv_files(directory):
    """
    Reads all CSV files in a directory and concatenates them into a single DataFrame.

    :param directory: Path to the directory containing the CSV files
    :return: Concatenated DataFrame
    """
    # List to hold data from each file
    all_data = []

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a CSV file
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            print(f"Reading file: {filename}")

            # Read the CSV file into a DataFrame
            data = pd.read_csv(filepath,header=0)
            data.Date = pd.to_datetime(data.Date,dayfirst=True)
            data.Symbol = data.Symbol.astype(str)

            # Append the DataFrame to the list
            all_data.append(data)

    # Concatenate all DataFrames
    if all_data:
        merged_data = pd.concat(all_data, ignore_index=True)
        merged_data = merged_data.sort_values(by="Date").reset_index(drop=True)
        print("All files concatenated successfully.")
        return merged_data
    else:
        print("No CSV files found in the directory.")
        return None


# Example usage
if __name__ == "__main__":
    # Directory containing the CSV files
    csv_directory = "./tick_data/barchart.com/raw"  # Replace with your directory path

    # Concatenate the CSV files
    concatenated_df = concat_csv_files(csv_directory)

    file_name_end= "2019_2024"

    if concatenated_df is not None:
        # Print some information about the concatenated DataFrame
        print(concatenated_df.head())
        print(concatenated_df.dtypes)
        print(f"Total rows: {len(concatenated_df)}")

        concatenated_df = concatenated_df.drop_duplicates()

        # Save the merged DataFrame to a new CSV file
        concatenated_df.to_csv(f"./tick_data/barchart.com/concatenated_barchart_{file_name_end}.csv", index=False)
        print(f"Concatenated data saved to 'concatenated_barchart_{file_name_end}.csv'.")

        # Save the merged DataFrame to an HDF5 file
        concatenated_df.to_hdf(f"./tick_data/barchart.com/concatenated_barchart_{file_name_end}.h5", key="data", mode="w")
        print(f"Concatenated data saved to 'concatenated_barchart_{file_name_end}.h5'.")
