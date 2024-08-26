# PDF Table Parser

A Python repository for parsing and processing tables from PDF files and manipulating CSV/TSV files. This repository contains scripts to handle various tasks such as combining CSV files, extracting tables from PDF files, extracting variable names from TSV files, and parsing long descriptions and page numbers from PDF files.

## Files and Scripts

### [append_csv.py](append_csv.py)

This script combines two CSV files into one. It reads the content of the first CSV file and appends the content of the second CSV file to it. The combined content is then saved back into the first CSV file.

### [camelot-pdf-extract-all-tables.py](camelot-pdf-extract-all-tables.py)

This script extracts tables from a specified PDF file and saves them as CSV files.

**Usage:**

* Ensure you have Ghostscript installed on your system

* Specify the path to your PDF file in the script

```bash
python3 camelot-pdf-extract-all-tables.py
```

### [extract_variable_names.py](extract_variable_names.py)

This script extracts variable names (column names) from a TSV file and saves them to a text file.

**Usage:**

* Run the script with the paths to the input TSV file and output text file as arguments

```bash
python3 extract_variable_names.py <tsv_file_path> <output_file_path>
```

### [merge_csvs.py](merge_csvs.py)

This script merges two CSV files based on specific columns and saves the merged data to a new CSV file.

### [parse_long_descriptions.py](parse-long-descriptions-andres.py)

This script extracts long descriptions of variables from a specified PDF file and saves them to CSV files.

### [parse-page-numbers.py](parse-page-numbers.py)

This script extracts variable names and their corresponding page numbers from a series of CSV files and saves the parsed data to new CSV files.

### [verify_all_variables_extracted.py](verify_all_variables_extracted.py)

This script verifies that all variables listed in a text file are present in a specified column of a CSV file.

Getting Started 
---------------

### Prerequisites

Ensure you have the following installed:

* Python3.x
* pandas library
* camelot-py library
* pdfplumber library
* Ghostscript

Install the required libraries using pip:

```bash
pip install pandas camelot-py[cv] pdfplumber
```

Install Ghostscript using Homebrew (macOS):

```bash
brew install ghostscript
```

