from fpdf import FPDF
import os

pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", size=12)
pdf.cell(200, 10, text="This is a test paper about artificial intelligence.", ln=True)
pdf.cell(200, 10, text="RAG systems combine retrieval with generation.", ln=True)

# Save at test/sample.pdf
output_path = os.path.join(os.path.dirname(__file__), "sample.pdf")
pdf.output(output_path)

print(f"Generated sample PDF at: {output_path}")
