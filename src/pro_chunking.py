# pro_chunking.py
# No changes needed
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# --- Step 1: Boilerplate detection & cleanup ---
BOILERPLATE_PATTERNS = [
    r"This sample PDF file is provided by Sample-Files\.com.*?Visit us for more sample files and resource\.",
    r"https://sample-files\.com/"
]

def clean_boilerplate(text: str) -> str:
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r'\n{2,}', '\n\n', text).strip()

# --- Step 2: High-level section splitting ---
SECTION_PATTERNS = [
    r"(?=\n\s*\d+\.\s[A-Z])",   # numbered headings like '1. Abstract'
    r"(?=\n\s*##?)",            # markdown-like headings
    r"(?=\n[A-Z][^\n]{1,80}\n)" # ALL CAPS or bold-like titles
]

def hierarchical_split(text: str) -> list[str]:
    """Splits a document into sections based on headings and patterns."""
    for pattern in SECTION_PATTERNS:
        parts = re.split(pattern, text)
        if len(parts) > 1:
            return [p.strip() for p in parts if p.strip()]
    return [text.strip()]  # fallback: whole text

# --- Step 3: Chunking pipeline ---
def professional_chunk_documents(docs: list[Document], chunk_size=1000, chunk_overlap=150):
    """
    Professional multi-stage chunking:
    1. Cleans boilerplate
    2. Splits by sections/headings
    3. Splits sections into chunks with overlap
    4. Enriches metadata
    """
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )

    for doc in docs:
        base_meta = {
            "source": doc.metadata.get("source"),
            "page_number": doc.metadata.get("page", doc.metadata.get("page_number", None))
        }

        cleaned_text = clean_boilerplate(doc.page_content)
        sections = hierarchical_split(cleaned_text)

        for section_idx, section_text in enumerate(sections):
            section_chunks = splitter.split_text(section_text)
            for idx, chunk_text in enumerate(section_chunks):
                all_chunks.append(Document(
                    page_content=chunk_text,
                    metadata={
                        **base_meta,
                        "section_id": section_idx,
                        "chunk_id": f"{base_meta['source']}_sec{section_idx}_ch{idx}"
                    }
                ))

    print(f"[CHUNKING] Created {len(all_chunks)} high-quality chunks")
    if all_chunks:
        print("\nSample professional chunk:\n")
        print(all_chunks[0].page_content[:500])
        print("\nMetadata:", all_chunks[0].metadata)
    return all_chunks