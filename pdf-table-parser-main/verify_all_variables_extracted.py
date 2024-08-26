import pandas as pd

# Function to read the variables from a text file
def read_variables_from_text_file(file_path):
    with open(file_path, 'r') as file:
        variables = file.read().splitlines()
    return variables

# Function to verify the variables in the CSV file
def verify_variables_in_csv(text_file_path, csv_file_path, column_name):
    # Read variables from the text file
    variables = read_variables_from_text_file(text_file_path)

    # Read the CSV file
    csv_data = pd.read_csv(csv_file_path)

    # Get the unique values from the specified column
    question_codes = csv_data[column_name].unique()

    # Check which variables are missing in the CSV file
    missing_variables = [var for var in variables if var not in question_codes]

    # Print the result
    if missing_variables:
        print("The following variables are missing in the CSV file:")
        for var in missing_variables:
            print(var)
        print("Count: ", len(missing_variables))
    else:
        print("All variables from the text file are present in the CSV file.")

# Specify the paths to the text file and the CSV file
text_file_path = 'spi_variables.txt'  # Path to your text file with variables
csv_file_path = 'csvs/clean_csvs/spi_2016_codebook.csv'  # Path to your CSV file
column_name = 'Question_Code'  # Name of the column to check in the CSV file

# Verify the variables
verify_variables_in_csv(text_file_path, csv_file_path, column_name)
