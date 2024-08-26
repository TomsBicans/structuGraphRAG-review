import pandas as pd
from io import StringIO
import csv
import sys
import re

def parse_csv_data(csv_content):
    # Read the CSV into a DataFrame
    df = pd.read_csv(StringIO(csv_content), header=None)
    
    results = []
    current_question_code = ""
    current_description = ""
    
    for index, row in df.iterrows():
        # Extract Question_Code and Description
        if pd.notna(row[0]) and "-" in row[0]:
            match = re.match(r"(V\d+) - (.+)", row[0])
            if match:
                if current_question_code and current_description:
                    # Add current question code and description with empty answer code and meaning
                    results.append({
                        "Question_Code": current_question_code,
                        "Description": current_description,
                        "Answer_Code": "",
                        "Answer_Meaning": ""
                    })
                current_question_code = match.group(1)
                current_description = match.group(2) 
                answer_idx = -1   

        for idx, col in enumerate(row):
            if "Label" == col:
                answer_idx = idx
        # Extract Answer_Code and Answer_Meaning
        if answer_idx != -1:
            if len(row) > 1 and pd.notna(row[answer_idx]) and "=" in row[answer_idx]:
                answer_code, answer_meaning = row[answer_idx].split("=", 1)
                answer_code = answer_code.strip()
                answer_meaning = answer_meaning.strip()
                results.append({
                    "Question_Code": current_question_code,
                    "Description": current_description,
                    "Answer_Code": answer_code,
                    "Answer_Meaning": answer_meaning
                })
    
    # Add the last Question_Code and Description if no answer code follows
    if current_question_code and current_description:
        results.append({
            "Question_Code": current_question_code,
            "Description": current_description,
            "Answer_Code": "",
            "Answer_Meaning": ""
        })
    
    return results

def post_process_results(results):
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(results)
    
    # Iterate through rows and check for empty Answer_Code and Answer_Meaning
    rows_to_drop = []
    for i in range(1, len(df)):
        if df.loc[i, 'Answer_Code'] == "" and df.loc[i, 'Answer_Meaning'] == "":
            if df.loc[i, 'Question_Code'] == df.loc[i - 1, 'Question_Code']:
                rows_to_drop.append(i)
    
    # Drop the rows identified
    df.drop(rows_to_drop, inplace=True)
    
    return df

# Path to the single CSV file
file_path = 'csvs/raw_csvs/spi_mega_table.csv'

try:
    # Open and read the CSV file
    with open(file_path, 'r') as file:
        csv_content = file.read()
        parsed_results = parse_csv_data(csv_content)

        # Skip saving if the parsed results are empty
        if not parsed_results:
            print("No data parsed from the CSV file.")
            sys.exit(0)

        # Post-process the parsed results
        processed_df = post_process_results(parsed_results)
        
        # Save the DataFrame to a new CSV file
        output_file_path = 'csvs/clean_csvs/spi_2020_codebook.csv'
        processed_df.to_csv(output_file_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

        print(f"Parsed data has been written to {output_file_path}")

except ValueError as e:
    print(e)
    sys.exit(1)
