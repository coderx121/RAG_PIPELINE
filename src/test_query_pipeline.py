# test_query_pipeline.py
# Minor updates: Use professional chunking, update paths if needed
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import shutil
import chromadb
import gc  # For forcing garbage collection
import time  # For adding delays
import stat  # For handling file permissions
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Import from query_pipeline (updated)
from src.query_pipeline import process_documents, build_vector_store, truncate_context
from src.embedding_pipeline import EmbeddingPipeline
from src.loader import load_pdfs
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Test config
TEST_DATA_DIR = "D:/RAG-PIPELINE/test_data"
TEST_VECTOR_DIR = "D:/RAG-PIPELINE/test_vector_db"
TEST_COLLECTION = "query_pipeline_test"
os.makedirs(TEST_DATA_DIR, exist_ok=True)
os.makedirs(TEST_VECTOR_DIR, exist_ok=True)

# Mock text for testing
MOCK_TEXT = "Fast language models are critical for efficient NLP tasks. They reduce latency."

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def handle_remove_readonly(func, path, exc):
    excvalue = exc[1]
    if excvalue.errno == 13 or "WinError 32" in str(excvalue):  # Permission denied or file in use
        os.chmod(path, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
        func(path)
    else:
        raise

def safe_rmtree(path):
    for attempt in range(5):
        try:
            shutil.rmtree(path, onerror=handle_remove_readonly)
            break
        except PermissionError as e:
            print(f"Retry {attempt+1}/5: PermissionError during rmtree: {e}")
            gc.collect()
            time.sleep(2)  # Increased delay for lock release
    else:
        raise RuntimeError(f"Failed to delete {path} after 5 attempts.")

class TestQueryPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Clean up test vector db
        if os.path.exists(TEST_VECTOR_DIR):
            client = chromadb.PersistentClient(path=str(TEST_VECTOR_DIR))
            try:
                client.delete_collection(TEST_COLLECTION)
            except chromadb.errors.NotFoundError:
                pass  # Collection doesn't exist
            del client  # Explicitly delete client reference
            gc.collect()  # Force garbage collection to release file locks
            time.sleep(2)  # Initial delay
            safe_rmtree(TEST_VECTOR_DIR)
            os.makedirs(TEST_VECTOR_DIR, exist_ok=True)

        # Create a mock PDF if none exist
        if not any(f.lower().endswith('.pdf') for f in os.listdir(TEST_DATA_DIR)):
            mock_pdf_path = os.path.join(TEST_DATA_DIR, "mock.pdf")
            pdf_content = (
                b'%PDF-1.3\n'
                b'%\xe2\xe3\xcf\xd3\n'
                b'1 0 obj\n'
                b'<< /Type /Catalog /Pages 2 0 R >>\n'
                b'endobj\n'
                b'2 0 obj\n'
                b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n'
                b'endobj\n'
                b'3 0 obj\n'
                b'<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >>\n'
                b'endobj\n'
                b'4 0 obj\n'
                b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\n'
                b'endobj\n'
                b'5 0 obj\n'
                b'<< /Length 100 >>\n'
                b'stream\n'
                b'BT /F1 12 Tf 100 700 Td (' + MOCK_TEXT.encode('utf-8') + b') Tj ET\n'
                b'endstream\n'
                b'endobj\n'
                b'xref\n'
                b'0 6\n'
                b'0000000000 65535 f \n'
                b'0000000015 00000 n \n'
                b'0000000060 00000 n \n'
                b'0000000111 00000 n \n'
                b'0000000257 00000 n \n'
                b'0000000336 00000 n \n'
                b'trailer << /Size 6 /Root 1 0 R >>\n'
                b'startxref\n'
                b'460\n'
                b'%%EOF'
            )
            with open(mock_pdf_path, "wb") as f:
                f.write(pdf_content)

    def test_process_documents(self):
        """Test document loading and chunking."""
        chunks = process_documents(TEST_DATA_DIR)
        if not chunks:
            self.skipTest("No valid PDFs found in test_data directory")
        self.assertGreater(len(chunks), 0, "No chunks created")
        self.assertIn("source", chunks[0].metadata)
        self.assertIn("page_number", chunks[0].metadata)
        self.assertGreater(len(chunks[0].page_content), 0)

    def test_build_vector_store(self):
        """Test building vector store from chunks."""
        mock_docs = [Document(page_content=MOCK_TEXT, metadata={"source": "mock.pdf", "page_number": 1})]
        vector_store = build_vector_store(mock_docs)
        self.assertIsInstance(vector_store, Chroma)
        self.assertGreater(len(vector_store.get()["ids"]), 0, "No documents in vector store")

    def test_truncate_context(self):
        """Test context truncation."""
        long_context = "A" * 9000
        truncated = truncate_context(long_context, max_length=8000)
        self.assertEqual(len(truncated), 8003)  # 8000 + "... [truncated]"
        self.assertTrue(truncated.endswith("... [truncated]"))
        short_context = "Short text"
        self.assertEqual(truncate_context(short_context), short_context)

    def test_end_to_end_query(self):
        """Test full query pipeline with Groq LLM."""
        if not GROQ_API_KEY:
            self.skipTest("GROQ_API_KEY not set")

        # Build index
        chunks = process_documents(TEST_DATA_DIR)
        if not chunks:
            chunks = [Document(page_content=MOCK_TEXT, metadata={"source": "mock.pdf", "page_number": 1})]
        vector_store = build_vector_store(chunks)

        # Initialize LLM
        try:
            llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model_name="mixtral-8x7b-32768",  # Use a known Groq model
                temperature=0.5,
                max_tokens=1024
            )
        except Exception as e:
            self.skipTest(f"Failed to initialize Groq LLM: {e}")

        # Set up RetrievalQA
        retriever = vector_store.as_retriever(search_kwargs={"k": 1, "mmr": True, "mmr_diversity": 0.7})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        # Run query
        query = "What are fast language models?"
        response = qa_chain({"query": query})
        self.assertIn("result", response)
        self.assertGreater(len(response["result"]), 0, "No answer returned")
        self.assertGreater(len(response["source_documents"]), 0, "No source documents returned")
        self.assertIn("source", response["source_documents"][0].metadata)
        self.assertTrue(any("fast language models" in doc.page_content.lower() for doc in response["source_documents"]))

    @classmethod
    def tearDownClass(cls):
        # Clean up
        if os.path.exists(TEST_VECTOR_DIR):
            client = chromadb.PersistentClient(path=str(TEST_VECTOR_DIR))
            try:
                client.delete_collection(TEST_COLLECTION)
            except chromadb.errors.NotFoundError:
                pass
            del client  # Explicitly delete client reference
            gc.collect()  # Force garbage collection to release file locks
            time.sleep(2)  # Initial delay
            safe_rmtree(TEST_VECTOR_DIR)

if __name__ == "__main__":
    unittest.main()