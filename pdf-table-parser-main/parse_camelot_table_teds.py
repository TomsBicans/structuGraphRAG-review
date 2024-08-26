import pandas as pd
import glob
from io import StringIO
import csv  # Importing csv for quoting
import sys

def parse_csv_data(csv_content):
    # Read the CSV into a DataFrame
    df = pd.read_csv(StringIO(csv_content), header=None)
    
    # Extract the variable name and short description from the first row
    variable_name = None
    short_description = None
    first_row = df.iloc[0].dropna().tolist()
    if first_row and len(first_row[0].split(": ")) == 2:
        variable_name, short_description = first_row[0].split(": ")
    
    # If variable name and short description are not found, skip this table
    if variable_name is None or short_description is None:
        print("Skipping CSV, variable name and short description not found in the first row.")
        return []

    print("variable name: ", variable_name)
    print("short description: ", short_description)
    
    # Extract the long description from the second row
    second_row = df.iloc[1].dropna().tolist()
    long_description = ' '.join(second_row)
    
    # Initialize lists to store answer codes and descriptions
    answer_codes = []
    answer_descriptions = []

    # Flag to start processing after encountering "Value" and "Label"
    long_description_parts = []
    start_processing = False
    value_col_index = None
    label_col_index = None

    # Iterate over the rows to extract answer codes and descriptions
    for index, row in df.iterrows():
        if index > 0:
            if not start_processing:
                if "Value" in row.values and "Label" in row.values:
                    value_col_index = row[row == "Value"].index[0]
                    label_col_index = row[row == "Label"].index[0]
                    start_processing = True
                else:
                    long_description_parts.extend(row.dropna().astype(str).tolist())
                continue

            if pd.isna(row.iloc[0]) and row.iloc[1] == "Total":
                break
            if pd.notna(row.iloc[value_col_index]) and pd.notna(row.iloc[label_col_index]):
                answer_codes.append(row.iloc[value_col_index])
                answer_descriptions.append(row.iloc[label_col_index])

    long_description = ' '.join(long_description_parts)
    # Prepare the result
    results = []
    for code, description in zip(answer_codes, answer_descriptions):
        results.append({
            "Question_Code": variable_name,
            "Short_Description": short_description,
            "Long_Description": long_description,
            "Answer_code": code,
            "Answer_meaning": description
        })
    return results

# Path to the folder containing CSV files
folder_path = 'csvs'
csv_files = glob.glob(f'{folder_path}/*.csv')

try:
    # Process each CSV file and save the parsed results
    for index, file_path in enumerate(csv_files, start=1):
        with open(file_path, 'r') as file:
            csv_content = file.read()
            parsed_results = parse_csv_data(csv_content)

            # Skip saving if the parsed results are empty
            if not parsed_results:
                continue

        # Convert parsed results to a DataFrame
        parsed_df = pd.DataFrame(parsed_results)
        
        # Save the DataFrame to a new CSV file
        parsed_df.to_csv(f'teds_d_parsed_table_{index}.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)

    # Optional: Print the last DataFrame for verification
    print(parsed_df)

except ValueError as e:
    print(e)
    sys.exit(1)
