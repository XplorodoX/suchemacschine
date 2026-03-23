from typing import Dict, List, Optional

"""PDF extraction module for scraping PDF documents from HS Aalen website."""

import os
import tempfile
from urllib.parse import urljoin, urlparse

import requests

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from pdfplumber import open as pdf_open
except ImportError:
    pdf_open = None

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None


def extract_pdf_text_with_ocr(file_path: str) -> Optional[str]:
    """Try OCR on PDF to extract text from scans."""
    if pytesseract is None or Image is None:
        return None
    
    try:
        from pdf2image import convert_from_path
        
        images = convert_from_path(file_path, first_page=1, last_page=5)  # First 5 pages max
        text_parts = []
        
        for img in images:
            try:
                text = pytesseract.image_to_string(img, lang='deu+eng')
                if text.strip():
                    text_parts.append(text)
            except Exception:
                continue
        
        if text_parts:
            return "\n".join(text_parts)
    except Exception as e:
        print(f"    OCR failed: {e}")
    
    return None


def extract_pdf_text(file_path: str) -> Optional[str]:
    """Extract text from a PDF file using available libraries."""
    if not os.path.exists(file_path):
        return None

    text_content = []

    # Try pdfplumber first (more reliable for complex PDFs)
    if pdf_open is not None:
        try:
            with pdf_open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
            if text_content:
                full_text = "\n".join(text_content)
                if len(full_text.strip()) > 100:
                    return full_text
        except Exception as e:
            print(f"    pdfplumber failed: {e}")

    # Fallback to pypdf
    if PdfReader is not None:
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
            if text_content:
                full_text = "\n".join(text_content)
                if len(full_text.strip()) > 100:
                    return full_text
        except Exception as e:
            print(f"    pypdf failed: {e}")

    # Try OCR if text extraction failed (PDF might be a scan)
    print("    Trying OCR for scanned PDF...")
    ocr_text = extract_pdf_text_with_ocr(file_path)
    if ocr_text and len(ocr_text.strip()) > 100:
        return ocr_text
    
    return None


def download_and_extract_pdf(pdf_url: str, session: requests.Session, timeout: int = 30) -> Optional[str]:
    """Download a PDF and extract its text content."""
    try:
        response = session.get(pdf_url, timeout=timeout)
        response.raise_for_status()

        # Check if response is actually a PDF
        content_type = response.headers.get("content-type", "").lower()
        if not ("pdf" in content_type or pdf_url.lower().endswith(".pdf")):
            return None

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        try:
            # Extract text
            text = extract_pdf_text(tmp_path)
            return text
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    except Exception as e:
        print(f"Failed to download/extract PDF {pdf_url}: {e}")
        return None


def find_pdf_links(html_content: str, page_url: str) -> List[str]:
    """Find all PDF links in HTML content."""
    from bs4 import BeautifulSoup

    pdf_links = []
    soup = BeautifulSoup(html_content, "html.parser")

    # Find all links with href ending in .pdf
    for link in soup.find_all("a", href=True):
        href = link.get("href", "").strip()
        if href.lower().endswith(".pdf"):
            # Convert relative URLs to absolute
            absolute_url = urljoin(page_url, href)
            pdf_links.append(absolute_url)

    # Also look for links with data-href attributes (some sites use these)
    for link in soup.find_all("a", attrs={"data-href": True}):
        href = link.get("data-href", "").strip()
        if href.lower().endswith(".pdf"):
            absolute_url = urljoin(page_url, href)
            pdf_links.append(absolute_url)

    # Remove duplicates while preserving order
    seen = set()
    unique_links = []
    for link in pdf_links:
        if link not in seen:
            seen.add(link)
            unique_links.append(link)

    return unique_links


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks for better vector search results."""
    if not text:
        return []
    
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        # Try to find a space near the end to avoid splitting words
        if end < len(text):
            space_pos = text.rfind(' ', start, end)
            if space_pos > start + (max_chars // 2):
                end = space_pos
        
        chunks.append(text[start:end].strip())
        start = end - overlap
        if start >= len(text) - overlap:
            break
            
    return chunks


def extract_pdfs_from_page(html_content: str, page_url: str, session: requests.Session) -> List[Dict]:
    """
    Extract all PDFs from a page and return their content with metadata.
    
    Returns:
        List of dicts with keys: url, text, chunks, filename
    """
    pdf_links = find_pdf_links(html_content, page_url)
    pdf_data = []

    for pdf_url in pdf_links:
        print(f"  Extracting PDF: {pdf_url}")
        text = download_and_extract_pdf(pdf_url, session)

        if text is None:
            print(f"    ✗ Skipping dead or unreadable PDF: {pdf_url}")
            continue

        filename = urlparse(pdf_url).path.split("/")[-1]
        chunks = chunk_text(text)
        
        pdf_data.append({
            "url": pdf_url,
            "filename": filename,
            "text": text,
            "chunks": chunks
        })
        print(f"    ✓ Added PDF ({len(text)} chars, {len(chunks)} chunks from {filename})")

    return pdf_data

import re # Ensure re is imported
