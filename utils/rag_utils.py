import os
import PyPDF2
from docx import Document
import numpy as np
import chromadb
from typing import List, Dict, Any, Optional, Tuple
from models.embeddings import EmbeddingModel
from config.config import COLLECTION_NAME, VECTOR_DIMENSION


def document_exists(self, document_id: str) -> bool:
    """Check if a document already exists in the vector store.
    
    Args:
        document_id: ID of the document to check
        
    Returns:
        True if document exists, False otherwise
    """
    try:
        # Query for IDs that start with the document ID
        results = self.collection.query(
            query_texts=[""],  # Empty query text
            where={"source": document_id},
            n_results=1
        )
        
        # If there are any results, the document exists
        return len(results["ids"][0]) > 0
    except Exception as e:
        logger.error(f"Error checking if document exists: {str(e)}")
        return False

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

class VectorStore:
    def __init__(self, embedding_model: EmbeddingModel, persist_dir: Optional[str] = None):
        """Initialize the vector store.
        
        Args:
            embedding_model: Model for generating embeddings
            persist_dir: Optional directory for persistence
        """
        self.embedding_model = embedding_model
        self.persist_path = os.path.join(persist_dir, "chroma_db") if persist_dir else None
        
        try:
            # Initialize ChromaDB client with persistence if specified
            if self.persist_path:
                os.makedirs(self.persist_path, exist_ok=True)
                self.client = chromadb.PersistentClient(path=self.persist_path)
            else:
                self.client = chromadb.Client()
            
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            raise Exception(f"Failed to initialize vector store: {e}")
    
    def add_documents(self, documents: List[str], document_id: str) -> None:
        """Add documents to the vector store.
        
        Args:
            documents: List of document chunks
            document_id: ID of the document
        """
        try:
            # Generate IDs for chunks
            ids = [f"{document_id}_{i}" for i in range(len(documents))]
            
            # Generate embeddings
            embeddings = self.embedding_model.get_embeddings(documents)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=documents,
                ids=ids,
                metadatas=[{"source": document_id} for _ in range(len(documents))]
            )
        except Exception as e:
            raise Exception(f"Error adding documents to vector store: {e}")
    
    def search(self, query: str, top_k: int = 5) -> List[str]:
        """Search for relevant document chunks.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant document chunks with source information
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.get_embeddings(query)
            
            # Search collection
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k,
                include=["documents", "metadatas"]
            )
            
            if not results or not results.get("documents") or len(results["documents"][0]) == 0:
                return []
                
            # Extract documents and add source information
            formatted_results = []
            for i, doc in enumerate(results["documents"][0]):
                source = results["metadatas"][0][i]["source"] if i < len(results["metadatas"][0]) else "unknown"
                formatted_results.append(f"From {source}: {doc}")
                
            return formatted_results
        except Exception as e:
            raise Exception(f"Error searching vector store: {e}")