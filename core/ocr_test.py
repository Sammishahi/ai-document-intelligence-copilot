import pytesseract
import fitz

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

pdf_path = "data/documents/sample_report.pdf"

doc = fitz.open(pdf_path)

print(f"Total pages: {len(doc)}")

for page_number, page in enumerate(doc, start=1):

    # Render PDF page as image
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

    # Convert Pixmap to PIL image
    from PIL import Image
    import io

    image = Image.open(io.BytesIO(pix.tobytes("png")))

    # OCR
    text = pytesseract.image_to_string(image)

    print("\n" + "=" * 60)
    print(f"PAGE {page_number}")
    print("=" * 60)

    print(text[:1000])