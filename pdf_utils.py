from PyPDF2 import PdfReader
import io

def sample_pdf_pages(file_bytes: bytes, pages_to_sample=3):
    reader = PdfReader(io.BytesIO(file_bytes))
    total_pages = len(reader.pages)
    sample_texts = []

    indices = [0]
    if total_pages > 1:
        indices += list(set([min(i, total_pages-1) for i in [total_pages//3, 2*total_pages//3]]))

    for i in indices[:pages_to_sample]:
        page = reader.pages[i]
        sample_texts.append(page.extract_text() or "")

    return "\n".join(sample_texts)

def extract_all_pages(file_bytes: bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    return [(i, page.extract_text() or "") for i, page in enumerate(reader.pages)]