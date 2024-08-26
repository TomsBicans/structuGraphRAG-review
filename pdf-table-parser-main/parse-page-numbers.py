import pandas as pd
import re

for i in range(0, 12):
    # Load the CSV file
    file_path = f'csvs/nsduh_table_{i}.csv'  # Update with your actual file path
    df = pd.read_csv(file_path)

    # Define a function to parse the cell content
    def parse_cell(cell):
        if pd.isna(cell):
            return None
        match = re.search(r'(.*?)\.+\s*(\d+)', cell)
        if match:
            variable_name = match.group(1).strip()
            page_number = match.group(2).strip()
            return variable_name, page_number
        return None

    # Apply the function to each cell and filter out None values
    parsed_data = df.applymap(parse_cell)

    # Create a list to hold the valid rows
    valid_rows = []

    # Iterate over the DataFrame and collect valid rows
    for row in parsed_data.values.flatten():
        if row is not None:
            valid_rows.append(row)

    # Convert the list of valid rows to a DataFrame
    combined_df = pd.DataFrame(valid_rows, columns=['variable_name', 'page_number'])

    # Display the combined DataFrame
    print("Parsed Variables and Pages:")
    print(combined_df)

    # Save the combined DataFrame to a new CSV file
    output_file_path = f'csvs/parsed_variables_pages_{i}.csv'
    combined_df.to_csv(output_file_path, index=False)
