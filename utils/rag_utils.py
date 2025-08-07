import os
import PyPDF2
from docx import Document
import numpy as np
# FAISS is the replacement for ChromaDB
from langchain_community.vectorstores import FAISS
from typing import List, Dict, Any, Optional, Tuple
from models.embeddings import EmbeddingModel
# These are no longer needed as they were Chroma-specific
# from config.config import COLLECTION_NAME, VECTOR_DIMENSION

# --- All of your helper functions below are unchanged ---

def load_document(file_path: str) -> str:
    """Load document content from various file formats.
    
    Args:
        file_path: Path to the document
        
    Returns:
        Document content as a string
    """
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
    """Load text from PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        PDF content as a string
    """
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def load_docx(file_path: str) -> str:
    """Load text from DOCX file.
    
    Args:
        file_path: Path to the DOCX file
        
    Returns:
        DOCX content as a string
    """
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def load_text(file_path: str) -> str:
    """Load text from TXT, MD, or CSV file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        File content as a string
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks.
    
    Args:
        text: Text to split
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
        
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # Adjust end to avoid cutting words
        if end < text_len:
            while end > start and not text[end].isspace():
                end -= 1
            if end == start:  # If no space found, use the original end
                end = min(start + chunk_size, text_len)
        
        # Add chunk
        chunks.append(text[start:end])
        
        # Move start position considering overlap
        start = end - overlap if end < text_len else text_len
        
    return chunks

# --- The VectorStore class is now updated to use FAISS ---

class VectorStore:
    def __init__(self, embedding_model: EmbeddingModel, persist_dir: Optional[str] = None):
        """Initialize the vector store using FAISS.
        
        Args:
            embedding_model: Model for generating embeddings
            persist_dir: Optional directory for persistence (not used by this in-memory version)
        """
        self.embedding_model = embedding_model
        self.langchain_embeddings = self.embedding_model.get_embedding_function()
        self.vector_store: Optional[FAISS] = None # Will hold the FAISS index
        self.processed_docs = set() # To track existing documents

    def document_exists(self, document_id: str) -> bool:
        """Check if a document has already been processed."""
        return document_id in self.processed_docs

    def add_documents(self, documents: List[str], document_id: str) -> None:
        """Add documents to the FAISS vector store.
        
        Args:
            documents: List of document chunks
            document_id: ID of the source document
        """
        try:
            # Create metadata for each chunk, linking it back to the source document
            metadatas = [{"source": document_id} for _ in documents]

            if self.vector_store is None:
                # Create a new FAISS index if one doesn't exist
                self.vector_store = FAISS.from_texts(
                    texts=documents, 
                    embedding=self.langchain_embeddings, 
                    metadatas=metadatas
                )
            else:
                # Add new documents to the existing index
                self.vector_store.add_texts(
                    texts=documents, 
                    metadatas=metadatas
                )
            
            # Mark this document as processed
            self.processed_docs.add(document_id)

        except Exception as e:
            raise Exception(f"Error adding documents to FAISS vector store: {e}")
    
    def search(self, query: str, top_k: int = 5) -> List[str]:
        """Search for relevant document chunks in the FAISS index.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant document chunks with source information
        """
        if self.vector_store is None:
            return []

        try:
            # FAISS similarity search returns LangChain Document objects
            results = self.vector_store.similarity_search(query, k=top_k)
            
            # Format results to match your original output style
            formatted_results = []
            for doc in results:
                source = doc.metadata.get("source", "unknown")
                formatted_results.append(f"From {source}: {doc.page_content}")
                
            return formatted_results
        except Exception as e:
            raise Exception(f"Error searching FAISS vector store: {e}")