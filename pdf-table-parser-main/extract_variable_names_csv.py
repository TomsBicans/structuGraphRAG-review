import pandas as pd

def extract_variable_names():
    # Hardcoded paths for the CSV file and the output file
    csv_file_path = 'datasets/PFI_2019_Dataset.csv'
    output_file_path = 'NHES_PFI_2019_variable_names.txt'

    # Load the CSV file with a specific encoding
    try:
        data = pd.read_csv(csv_file_path, encoding='latin1')
    except UnicodeDecodeError as e:
        print(f"Error reading {csv_file_path}: {e}")
        return

    # Extract the column names (variable names)
    variable_names = data.columns.tolist()

    # Save the variable names to a text file
    with open(output_file_path, 'w') as file:
        for name in variable_names:
            file.write(name + '\n')

if __name__ == "__main__":
    extract_variable_names()
