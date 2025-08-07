# test_kb_processing.py
import os
import logging
from models.embeddings import EmbeddingModel
from utils.rag_utils import VectorStore, load_document, chunk_text

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_knowledge_base():
    """Process all text files in the knowledge_base directory."""
    try:
        # Initialize embedding model
        embedding_model = EmbeddingModel()
        
        # Initialize vector store
        persist_dir = os.path.join(os.getcwd(), "vector_store")
        vector_store = VectorStore(embedding_model, persist_dir=persist_dir)
        
        # Get knowledge_base directory path
        kb_dir = os.path.join(os.getcwd(), "knowledge_base")
        
        # Check if directory exists
        if not os.path.exists(kb_dir):
            logger.error(f"Knowledge base directory not found: {kb_dir}")
            return
        
        # Get all text files
        text_files = [f for f in os.listdir(kb_dir) if f.endswith('.txt')]
        
        if not text_files:
            logger.warning("No text files found in knowledge base directory")
            return
        
        logger.info(f"Found {len(text_files)} text files in knowledge base directory")
        
        # Process each file
        for file_name in text_files:
            file_path = os.path.join(kb_dir, file_name)
            
            # Load document content
            logger.info(f"Loading document: {file_name}")
            document_text = load_document(file_path)
            
            # Chunk document
            logger.info(f"Chunking document: {file_name}")
            chunks = chunk_text(document_text)
            
            # Add to vector store
            logger.info(f"Adding {len(chunks)} chunks to vector store for document: {file_name}")
            vector_store.add_documents(chunks, file_name)
            
        logger.info("Knowledge base processing completed successfully")
    except Exception as e:
        logger.error(f"Error processing knowledge base: {str(e)}")

if __name__ == "__main__":
    process_knowledge_base()