import pandas as pd
import glob
import csv

def merge_csv_files(folder_path):
    # Use glob to get all CSV files in the folder
    csv_files = glob.glob(f'{folder_path}/*.csv')

    # Initialize an empty list to hold DataFrames
    data_frames = []

    # Process each CSV file
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        data_frames.append(df)

    # Concatenate all DataFrames into a single DataFrame
    merged_df = pd.concat(data_frames, ignore_index=True)

    folder_path = 'csvs/clean_csvs'
    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(f'{folder_path}/nsatts_2020_codebook.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)

    # Return the merged DataFrame
    return merged_df

# Path to the folder containing CSV files
folder_path = 'csvs/parsed_csvs'

# Merge the CSV files
merged_df = merge_csv_files(folder_path)

# Optional: Print the merged DataFrame for verification
print(merged_df)
