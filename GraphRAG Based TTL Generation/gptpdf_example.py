from gptpdf import parse_pdf
import os
import dotenv
dotenv.load_dotenv()
api_key=os.environ.get("OPENAI_API_KEY")
pdf_path = "/Users/saillab/Shengting/CAST/example_doc/(1013) 2023 E Lan.pdf"

content, image_paths = parse_pdf(
	api_key=api_key,
    pdf_path=pdf_path,
    output_dir='../Processed/(1013) 2023 E Lan.pdf',
    model="gpt-4o",
    verbose=True,
)

print(content)
