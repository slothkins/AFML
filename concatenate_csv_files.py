import pandas as pd
import os
import tables

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
            data = pd.read_csv(filepath)

            # Append the DataFrame to the list
            all_data.append(data)

    # Concatenate all DataFrames
    if all_data:
        merged_data = pd.concat(all_data, ignore_index=True)
        print("All files concatenated successfully.")
        return merged_data
    else:
        print("No CSV files found in the directory.")
        return None


# Example usage
if __name__ == "__main__":
    # Directory containing the CSV files
    csv_directory = "./"  # Replace with your directory path

    # Concatenate the CSV files
    concatenated_df = concat_csv_files(csv_directory)

    if concatenated_df is not None:
        # Print some information about the concatenated DataFrame
        print(concatenated_df.head())
        print(f"Total rows: {len(concatenated_df)}")

        # Save the merged DataFrame to a new CSV file
        concatenated_df.to_csv("concatenated_data.csv", index=False)
        print("Concatenated data saved to 'concatenated_data.csv'.")

        # Save the merged DataFrame to an HDF5 file
        concatenated_df.to_hdf("concatenated_data.h5", key="data", mode="w")
        print("Concatenated data saved to 'concatenated_data.h5'.")
