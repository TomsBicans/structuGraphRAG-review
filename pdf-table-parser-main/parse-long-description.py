import pandas as pd
import re
import pdfplumber

# Load the CSV file with variable names and page numbers
csv_file_path = 'parsed_variables_pages.csv'
print(f"Loading CSV file from {csv_file_path}")
df = pd.read_csv(csv_file_path)
print(f"CSV file loaded successfully with {len(df)} records")

# Load the PDF file
pdf_file_path = 'pdfs/NSDUH-2022-DS0001-info-codebook (1).pdf'
print(f"Loading PDF file from {pdf_file_path}")

# Function to find the page number based on logical page number
def find_page_by_logical_number(pdf, logical_page_number):
    print(f"Searching for logical page number: {logical_page_number}")
    for i, page in enumerate(pdf.pages):
        print(i)
        text = page.extract_text()
        if text:
            match = re.search(r'Codebook Creation Date:.*?\.{5,}\s*(\d+)', text)
            if match and int(match.group(1)) == logical_page_number:
                print(f"Logical page number {logical_page_number} found at actual page number {i + 1}")
                return i + 1  # Return the actual page number (1-based index)
    print(f"Logical page number {logical_page_number} not found")
    return None

# Extract descriptions for each variable
descriptions = []

with pdfplumber.open(pdf_file_path) as pdf:
    print(f"PDF file loaded successfully with {len(pdf.pages)} pages")
    for index, row in df.iterrows():
        variable_name = row['variable_name']
        logical_page_number = int(row['page_number'])
        print(f"Processing variable '{variable_name}' on logical page number {logical_page_number}")
        actual_page_number = find_page_by_logical_number(pdf, logical_page_number)
        
        if actual_page_number:
            page_text = pdf.pages[actual_page_number - 1].extract_text()
            descriptions.append({
                'variable_name': variable_name,
                'page_number': logical_page_number,
                'description': page_text.strip() if page_text else 'No text found'
            })
            print(f"Description for '{variable_name}' added successfully")
        else:
            descriptions.append({
                'variable_name': variable_name,
                'page_number': logical_page_number,
                'description': 'Page not found'
            })
            print(f"Page for '{variable_name}' not found")

# Convert the descriptions to a DataFrame
descriptions_df = pd.DataFrame(descriptions)
print("Descriptions extracted and DataFrame created")

# Save the descriptions to a new CSV file
output_file_path = 'csvs/variable_descriptions.csv'
print(f"Saving descriptions to {output_file_path}")
descriptions_df.to_csv(output_file_path, index=False)
print("Descriptions saved successfully")

# Print the descriptions DataFrame
print(descriptions_df)
