import os
import PyPDF2
from docx import Document
import numpy as np
from langchain_community.vectorstores import FAISS
from typing import List, Dict, Any, Optional, Tuple
from models.embeddings import EmbeddingModel

# --- All of your helper functions below are unchanged ---

def load_document(file_path: str) -> str:
    """Load document content from various file formats."""
    try:
        _, file_extension = os.path.splitext(file_path)
        if file_extension.lower() == '.pdf':
            return load_pdf(file_path)
        elif file_extension.lower() == '.docx':
            return load_docx(file_path)
        elif file_extension.lower() in ['.txt', '.md', '.csv']:
            return load_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        raise Exception(f"Error loading document: {e}")

def load_pdf(file_path: str) -> str:
    """Load text from PDF file."""
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def load_docx(file_path: str) -> str:
    """Load text from DOCX file."""
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def load_text(file_path: str) -> str:
    """Load text from TXT, MD, or CSV file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            while end > start and not text[end].isspace():
                end -= 1
            if end == start:
                end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        start = end - overlap if end < text_len else text_len
    return chunks

# --- The VectorStore class is now corrected ---

class VectorStore:
    def __init__(self, embedding_model: EmbeddingModel, persist_dir: Optional[str] = None):
        """Initialize the vector store using FAISS."""
        self.embedding_model = embedding_model
        # REMOVED the incorrect line that was causing the error.
        # self.langchain_embeddings = self.embedding_model.get_embedding_function() 
        self.vector_store: Optional[FAISS] = None
        self.processed_docs = set()

    def document_exists(self, document_id: str) -> bool:
        """Check if a document has already been processed."""
        return document_id in self.processed_docs

    def add_documents(self, documents: List[str], document_id: str) -> None:
        """Add documents to the FAISS vector store."""
        try:
            metadatas = [{"source": document_id} for _ in documents]

            if self.vector_store is None:
                # Create a new FAISS index
                # FIXED: Pass the entire embedding_model object directly
                self.vector_store = FAISS.from_texts(
                    texts=documents, 
                    embedding=self.embedding_model, 
                    metadatas=metadatas
                )
            else:
                # Add new documents to the existing index
                self.vector_store.add_texts(
                    texts=documents, 
                    metadatas=metadatas
                )
            
            self.processed_docs.add(document_id)

        except Exception as e:
            raise Exception(f"Error adding documents to FAISS vector store: {e}")
    
    def search(self, query: str, top_k: int = 5) -> List[str]:
        """Search for relevant document chunks in the FAISS index."""
        if self.vector_store is None:
            return []
        try:
            results = self.vector_store.similarity_search(query, k=top_k)
            
            formatted_results = []
            for doc in results:
                source = doc.metadata.get("source", "unknown")
                formatted_results.append(f"From {source}: {doc.page_content}")
                
            return formatted_results
        except Exception as e:
            raise Exception(f"Error searching FAISS vector store: {e}")