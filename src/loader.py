# loader.py
# Updated: Modified _list_pdfs to avoid duplicates on case-insensitive systems
from __future__ import annotations
from typing import List
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

def _list_pdfs(directory: str, recursive: bool = True) -> List[Path]:
    """
    List all PDF files in a given directory (case-insensitive).
    """
    base = Path(directory)
    if not base.exists():
        raise FileNotFoundError(f"[loader] Directory does not exist: {base}")
    
    pdfs = list(base.glob(f'{"**/" if recursive else ""}*.[pP][dD][fF]'))
    return pdfs

def load_pdfs(
    directory: str,
    recursive: bool = True,
    skip_empty_pages: bool = True,
    min_chars_per_page: int = 10,
    verbose: bool = True
) -> List[Document]:
    """
    Loads PDFs from a directory into a list of LangChain Documents.

    Each page becomes one Document with metadata: source, page_number, total_pages.
    """
    pdf_paths = _list_pdfs(directory, recursive=recursive)

    if verbose:
        print(f"[loader] Found {len(pdf_paths)} PDF(s) in {directory}")

    documents: List[Document] = []
    for path in pdf_paths:
        if verbose:
            print(f"[loader] Loading: {path.name}")

        try:
            loader = PyPDFLoader(str(path))
            pages = loader.load()
            total_pages = len(pages)
            valid_pages = 0

            for i, page in enumerate(pages):
                content = (page.page_content or "").strip()
                if skip_empty_pages and (len(content) < min_chars_per_page):
                    continue

                # enrich metadata
                page.metadata.update({
                    "source": path.name,
                    "file_path": str(path),
                    "page_number": i + 1,
                    "total_pages": total_pages
                })
                documents.append(page)
                valid_pages += 1

            if verbose:
                print(f"  → Loaded {valid_pages}/{total_pages} page(s) from {path.name}")

        except Exception as e:
            print(f"[loader][ERROR] Failed to load {path.name}: {e}")

    if verbose:
        print(f"\n[loader] Total pages after filtering: {len(documents)}")
    return documents