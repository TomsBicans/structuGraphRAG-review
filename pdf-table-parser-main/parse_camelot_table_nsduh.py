import pandas as pd
import re
import glob
import csv  # Importing csv for quoting

# Function to remove superscript/subscript text
def remove_super_sub_scripts(text):
    """Remove text in the form of <s>#</s> where # is any string."""
    return re.sub(r'<s>.*?</s>', '', text)

# Function to extract the short description
def extract_short_description(description):
    """Extract the short description after the pattern 'Len : #'."""
    if isinstance(description, str):
        match = re.search(r'Len : \d+(.+)', description)
        if match:
            return remove_super_sub_scripts(match.group(1).strip())
    return ''

# Function to parse camelot table CSV files
def parse_camelot_table_v6(camelot_table):
    # Drop rows with all NaN values
    camelot_table.dropna(how='all', inplace=True)

    # Initialize lists to store extracted data
    question_codes = []
    short_descriptions = []
    related_variables = []
    answer_codes = []
    answer_meanings = []

    current_question_code = None
    current_short_description = None
    current_related_variables = None

    # Iterate through the rows of the camelot table
    for index, row in camelot_table.iterrows():
        if len(row) > 1 and pd.notna(row.iloc[0]) and not row.iloc[0].startswith('(') and not row.iloc[0].startswith('\\n'):
            current_question_code = re.sub(r'\s+', '', row.iloc[0])  # Remove spaces and newlines
            current_question_code = remove_super_sub_scripts(current_question_code)  # Remove superscripts/subscripts
            current_short_description = extract_short_description(row.iloc[1] if pd.notna(row.iloc[1]) else '')
        elif len(row) > 1 and pd.notna(row.iloc[0]) and row.iloc[0].startswith('('):
            current_related_variables = remove_super_sub_scripts(row.iloc[0])
        elif len(row) > 1 and pd.notna(row.iloc[1]) and '=' in str(row.iloc[1]):
            code_meaning_split = str(row.iloc[1]).split('=')
            code = remove_super_sub_scripts(code_meaning_split[0].strip())
            if code:  # Skip if answer_code is empty
                meaning = remove_super_sub_scripts(code_meaning_split[1].split('.....')[0].strip())
                question_codes.append(current_question_code)
                short_descriptions.append(current_short_description)
                related_variables.append(current_related_variables)
                answer_codes.append(code)
                answer_meanings.append(meaning)

    # Create a DataFrame with the extracted data
    parsed_data = pd.DataFrame({
        'Question_Code': question_codes,
        'Short_Description': short_descriptions,
        'Related_Variables': related_variables,
        'Answer_code': answer_codes,
        'Answer_meaning': answer_meanings
    })

    return parsed_data

# Use glob to dynamically load all camelot CSV files
file_paths = glob.glob('csvs/nsduh_table_*.csv')

# Parse all camelot tables
parsed_data_frames = [parse_camelot_table_v6(pd.read_csv(file_path, dtype=str)) for file_path in file_paths]

# Combine all parsed data into a single DataFrame
consolidated_data_v6 = pd.concat(parsed_data_frames, ignore_index=True)

# Save the consolidated DataFrame to a CSV file
consolidated_data_v6.to_csv('consolidated_nsduh_data.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)

# Display the consolidated DataFrame
print(consolidated_data_v6)

# Print the specific cell in the DataFrame
print(consolidated_data_v6.iloc[90:91, 4:5])
