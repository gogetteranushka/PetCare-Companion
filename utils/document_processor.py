import os
import PyPDF2
import numpy as np
from typing import List, Dict, Any, Tuple

def extract_text(file_path: str) -> str:
    """Extract text from PDF or text files"""
    try:
        if file_path.endswith('.pdf'):
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                # Only process first 10 pages to reduce memory usage
                for i in range(min(10, len(reader.pages))):
                    text += reader.pages[i].extract_text() + "\n"
                return text
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return ""
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks (optimized)"""
    try:
        # Limit total text size for memory efficiency
        text = text[:100000]  # Limit to 100K characters
        
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - overlap
        return chunks
    except Exception as e:
        print(f"Error chunking text: {e}")
        return []

def create_document_index(document_dir: str, model) -> Tuple[List[Dict], np.ndarray]:
    """Create searchable index from documents (optimized)"""
    try:
        chunks_with_metadata = []
        chunk_texts = []
        
        file_count = 0
        for file in os.listdir(document_dir):
            if file.endswith(('.pdf', '.txt')):
                # Limit number of files processed
                file_count += 1
                if file_count > 5:  # Process max 5 files
                    break
                    
                file_path = os.path.join(document_dir, file)
                text = extract_text(file_path)
                chunks = chunk_text(text)
                
                for i, chunk in enumerate(chunks):
                    chunks_with_metadata.append({
                        'text': chunk,
                        'source': file,
                        'chunk_id': i
                    })
                    chunk_texts.append(chunk)
        
        # Create embeddings for all chunks
        from models.embeddings import get_embeddings
        embeddings = get_embeddings(chunk_texts, model)
        
        return chunks_with_metadata, embeddings
    except Exception as e:
        print(f"Error creating document index: {e}")
        return [], np.array([])

def search_documents(query: str, chunks_with_metadata: List[Dict], 
                   embeddings: np.ndarray, model, top_k: int = 2) -> List[Dict]:
    """Search for relevant document chunks (optimized)"""
    try:
        from models.embeddings import get_embeddings
        query_embedding = get_embeddings([query], model)[0]
        
        # For TF-IDF vectorizer with sparse matrices
        if hasattr(embeddings, 'toarray'):
            embeddings = embeddings.toarray()
            
        # Calculate similarity (simplified)
        similarities = np.dot(embeddings, query_embedding.T).flatten()
        
        # Get top matches (limit to 2 for faster processing)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        
        for idx in top_indices:
            results.append({
                'text': chunks_with_metadata[idx]['text'],
                'source': chunks_with_metadata[idx]['source'],
                'similarity': float(similarities[idx])
            })
            
        return results
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []