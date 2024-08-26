import pandas as pd
import re
import pdfplumber



# Load the PDF file
pdf_file_path = 'pdfs/NSDUH-2022-DS0001-info-codebook (1).pdf'
print(f"Loading PDF file from {pdf_file_path}")

# Function to find the page number based on logical page number
def find_page_by_logical_number(pdf, logical_page_number):
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            match = re.search(r'Codebook Creation Date:.*?\.{5,}\s*(\d+)', text)
            if match and int(match.group(1)) == logical_page_number:
                return i + 1  # Return the actual page number (1-based index)
    return None

# Function to extract the description of a variable from the page text
def extract_description(page_text, variable_name):
    # Define a pattern to identify the start of the variable line, considering the possible footnote numbers
    variable_line_pattern = rf"{re.escape(variable_name)}\d*\s+Len\s*:\s*\d+"
    
    # Split the page text into lines
    lines = page_text.split('\n')
    
    # Find the line where the variable is defined
    variable_line_index = None
    for i, line in enumerate(lines):
        if re.search(variable_line_pattern, line):
            variable_line_index = i
            break
    
    if variable_line_index is not None:
        # Extract the description, which is all text before the variable line
        description_lines = lines[:variable_line_index]
        
        # Detect the end of the previous variable's text
        end_of_previous_text = None
        for j, desc_line in enumerate(reversed(description_lines)):
            if re.search(r'\.{3,}\s*\d+', desc_line) or (j > 0 and re.search(r'\s+Len\s*:\s*\d+', description_lines[-j-1])):
                end_of_previous_text = len(description_lines) - j
                break
        
        if end_of_previous_text is not None:
            description_lines = description_lines[end_of_previous_text:]
        
        description = ' '.join(description_lines).strip()
        # Remove any footer text like "Codebook Creation Date..."
        description = re.sub(r'Codebook Creation Date:.*?\d+', '', description)
        return description.strip()
    
    return 'Description not found'

# Function to clean the text and remove unwanted parts like titles and footers
def clean_text(text):
    # Remove footers
    text = re.sub(r'Codebook Creation Date:.*?\d+', '', text)
    # Remove lines that are in all uppercase (likely titles) and that are short enough to be titles
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Check if the line is likely a title
        if not (line.isupper() and len(line) < 50):
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines).strip()


for i in range(0, 12):

    # Load the CSV file with variable names and page numbers
    csv_file_path = f'csvs/parsed_variables_pages_{i}.csv'
    print(f"Loading CSV file from {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    print(f"CSV file loaded successfully with {len(df)} records")

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
                page_text = clean_text(page_text)  # Clean the text to remove titles and footers
                description = extract_description(page_text, variable_name)
                if 'Description not found' in description:
                    description = ''  # Set empty if no description is found
                descriptions.append({
                    'variable_name': variable_name,
                    'description': description
                })
                print(f"Description for '{variable_name}' added successfully")
            else:
                descriptions.append({
                    'variable_name': variable_name,
                    'description': ''
                })
                print(f"Page for '{variable_name}' not found")

    # Convert the descriptions to a DataFrame
    descriptions_df = pd.DataFrame(descriptions)
    print("Descriptions extracted and DataFrame created")

    # Save the descriptions to a new CSV file
    output_file_path = f'csvs/variable_descriptions_andres_{i}.csv'
    print(f"Saving descriptions to {output_file_path}")
    descriptions_df.to_csv(output_file_path, index=False)
    print("Descriptions saved successfully")

    # Print the descriptions DataFrame
    print(descriptions_df)
