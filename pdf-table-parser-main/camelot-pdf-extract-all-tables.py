import camelot
import matplotlib.pyplot as plt
from ctypes.util import find_library
print(find_library("gs")) # Make sure you have ghostscript installed (thru homebrew or other)

# Specify the path to your PDF file
pdf_path = "pdfs/PFI_2019.pdf"

# Extract tables using stream mode (for tables without borders)
tables = camelot.read_pdf(
    pdf_path, 
    pages='all', # or all
    flavor='stream', # or lattice
    edge_tol=1000, 
    # table_regions=['55,702,300,52'], # x1, y1, x2, y2 (top left corner, bottom right corner)
    flag_size=True
)

# camelot.plot(tables[0], kind='textedge').show()

# Function to convert specific columns to strings to avoid automatic date conversion
def convert_columns_to_string(df):
    for col in df.columns:
        df[col] = df[col].astype(str)
    return df


for i, table in enumerate(tables):
    print(f"Table {i}")
    print(table.parsing_report)
    table.df = convert_columns_to_string(table.df)
    table.to_csv(f"csvs/raw_csvs/nhes_pfi_table_{i}.csv")

    # plt.show()