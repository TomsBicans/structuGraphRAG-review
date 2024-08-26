import pandas as pd

# Load the first CSV file
csv1 = pd.read_csv('variable_descriptions.csv')

# Load the second CSV file
csv2 = pd.read_csv('csvs/variable_descriptions_andres_11.csv')

combined_csv = pd.concat([csv1, csv2], ignore_index=True)

# Save the combined CSV to a new file
combined_csv.to_csv('variable_descriptions.csv', index=False)
