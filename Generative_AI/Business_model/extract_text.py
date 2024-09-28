from PyPDF2 import PdfReader

pdf_path = "/Users/admin/Desktop/Generative_AI/business_model/Business_gm.pdf"
output_txt = "/Users/admin/Desktop/Generative_AI/business_model/dataset.txt"

# Extract text from PDF
with open(output_txt, 'w') as f:
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text = page.extract_text()
        f.write(text)
