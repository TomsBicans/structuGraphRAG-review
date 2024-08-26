import pandas as pd
import glob
from io import StringIO
import csv  # Importing csv for quoting
import sys

def parse_csv_data(csv_content):
    # Read the CSV into a DataFrame
    df = pd.read_csv(StringIO(csv_content), header=None)
    
    # Initialize the variable_name and short_description
    variable_name = None
    short_description = ""
    
    # Find the variable name and initial part of the short description
    for col in df.iloc[0]:
        if pd.notna(col) and ':' in col:
            try:
                variable_name, part_description = col.split(": ", 1)
                short_description += part_description
                break
            except ValueError as e:
                print(f"Error splitting column '{col}': {e}")
    
    if variable_name is None or short_description is None:
        print("Variable name and short description not found in the first row. Skipping CSV.")
        return []
    
    # Concatenate additional parts of the short description until the row with "Value" and "Label" is found
    row_index = 1
    while row_index < len(df):
        row = df.iloc[row_index]
        if "Value" in row.values and "Label" in row.values:
            value_col_index = row[row == "Value"].index[0]
            label_col_index = row[row == "Label"].index[0]
            break
        if pd.notna(row.iloc[0]):
            short_description += " " + row.iloc[0]
        row_index += 1
    
    print("Variable name: ", variable_name)
    print("Short description: ", short_description)

    # Initialize lists to store answer codes and descriptions
    answer_codes = []
    answer_descriptions = []

    # Iterate over the rows to extract answer codes and descriptions
    for index, row in df.iterrows():
        if index <= row_index:
            continue
        if pd.isna(row.iloc[value_col_index]) and row.iloc[label_col_index] == "Total":
            break
        if pd.notna(row.iloc[value_col_index]) and pd.notna(row.iloc[label_col_index]):
            if row.iloc[value_col_index] != "Value" and row.iloc[label_col_index] != "Label":
                answer_codes.append(row.iloc[value_col_index])
                answer_descriptions.append(row.iloc[label_col_index])

    # Prepare the result
    results = []
    for code, description in zip(answer_codes, answer_descriptions):
        results.append({
            "Variable_name": variable_name,
            "Short_description": short_description,
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
        parsed_df.to_csv(f'nsatts_parsed_table_{index}.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)

    # Optional: Print the last DataFrame for verification
    print(parsed_df)

except ValueError as e:
    print(e)
    sys.exit(1)