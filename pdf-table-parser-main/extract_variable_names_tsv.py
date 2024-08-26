import pandas as pd
import argparse

def extract_variable_names(tsv_file_path, output_file_path):
    # Load the TSV file
    data = pd.read_csv(tsv_file_path, sep='\t')

    # Extract the column names (variable names)
    variable_names = data.columns.tolist()

    # Save the variable names to a text file
    with open(output_file_path, 'w') as file:
        for name in variable_names:
            file.write(name + '\n')

def main():
    parser = argparse.ArgumentParser(description="Extract variable names from a TSV file and save to a text file.")
    parser.add_argument('tsv_file_path', type=str, help="Path to the TSV file")
    parser.add_argument('output_file_path', type=str, help="Path to the output text file")

    args = parser.parse_args()

    extract_variable_names(args.tsv_file_path, args.output_file_path)

if __name__ == "__main__":
    tsv_file = "datasets/37692-0001-Data.tsv"
    output_file = "spi_variables.txt"
    extract_variable_names(tsv_file, output_file)
