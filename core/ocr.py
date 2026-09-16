from pathlib import Path

import fitz
import pytesseract
from PIL import Image


class OCRProcessor:
    """
    Handles OCR extraction from scanned PDF pages.
    """

    def __init__(
        self,
        tesseract_path: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        dpi_scale: float = 2.0,
    ):
        self.tesseract_path = tesseract_path
        self.dpi_scale = dpi_scale

        pytesseract.pytesseract.tesseract_cmd = self.tesseract_path

    def extract_page_text(self, page) -> str:
        """
        Convert one PDF page into an image and extract text using Tesseract.
        """

        matrix = fitz.Matrix(self.dpi_scale, self.dpi_scale)
        pix = page.get_pixmap(matrix=matrix)

        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples,
        )

        text = pytesseract.image_to_string(image)

        return text.strip()

    def extract_pdf_text(self, pdf_path: str) -> list[dict]:
        """
        OCR all pages of a PDF.

        Returns:
            [
                {
                    "page": 1,
                    "text": "..."
                },
                ...
            ]
        """

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        document = fitz.open(pdf_path)

        results = []

        for page_number, page in enumerate(document, start=1):

            text = self.extract_page_text(page)

            if text:
                results.append(
                    {
                        "page": page_number,
                        "text": text,
                    }
                )

        document.close()

        return results