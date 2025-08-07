import streamlit as st
import os
import tempfile
import logging
from config.config import APP_TITLE, RESPONSE_MODES, PET_SPECIES, validate_together_api_key, TOGETHER_API_KEY
from models.embeddings import EmbeddingModel
from models.llm import TogetherModel
from utils.rag_utils import VectorStore, load_document, chunk_text
from typing import List
import requests
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Initialize session state
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Custom CSS with soft brown theme and ULTRA-AGGRESSIVE black bar targeting
def custom_css():
    return """
    <style>
    /* Soft Brown Theme */
    :root {
        --bg-light-brown: #F5F0E6;     /* Light beige for main background */
        --sidebar-brown: #E6DCD0;      /* Darker beige for sidebar */
        --text-dark: #4A3B27;          /* Dark brown for text */
        --bubble-white: #FFFFFF;       /* White for chat bubbles */
        --accent-brown: #A67C52;       /* Medium brown for accents */
        --accent-brown-hover: #8B6841; /* Darker brown for hover states */
        --border-light: #DFD3C3;       /* Light beige for borders */
        --header-gradient-1: #A67C52;  /* Start of header gradient */
        --header-gradient-2: #C19A6B;  /* End of header gradient */
        --dropdown-bg: #F9F6F0;        /* Light beige for dropdown background */
    }

    /* Apply background to main content and sidebar */
    .main, [data-testid="stAppViewContainer"], .stApp {
        background-color: var(--bg-light-brown) !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: var(--sidebar-brown) !important;
    }

    /* Make ALL text dark brown */
    body, p, li, label, h1, h2, h3, .stMarkdown, .stText, .stCaption, div {
        color: var(--text-dark) !important;
    }
    
    /* Welcome header styling */
    .welcome-header {
        background: linear-gradient(90deg, var(--header-gradient-1), var(--header-gradient-2));
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .welcome-header h1 {
        color: white !important;
        font-weight: 700;
        margin: 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    .welcome-header p {
        color: white !important;
        opacity: 0.9;
        margin: 5px 0 0 0;
    }
    
    /* Override text in sidebar to be dark brown */
    section[data-testid="stSidebar"] *, section[data-testid="stSidebar"] div, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span {
        color: var(--text-dark) !important;
    }

    /* FIX 2: Make pet type dropdown LIGHT COLORED with dark text */
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div,
    div.st-emotion-cache-1aumxhk {
        background-color: var(--dropdown-bg) !important;
        border-radius: 8px !important;
        border: 1px solid var(--border-light) !important;
    }
    
    /* FIX 2: Ensure all dropdown list items have proper background and visible text */
    [data-baseweb="select"] ul,
    [data-baseweb="select"] li,
    [data-baseweb="menu"],
    [data-baseweb="list"],
    [data-baseweb="popover"],
    div[role="listbox"],
    ul[role="listbox"],
    li[role="option"],
    div.st-emotion-cache-1aumxhk * {
        background-color: var(--dropdown-bg) !important;
        color: var(--text-dark) !important;
    }

    /* FIX 2: Ensure the dropdown text is visible */
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] *,
    div.st-emotion-cache-1aumxhk * {
        color: var(--text-dark) !important;
    }
    
    /* Make the file uploader white */
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background-color: white !important;
        border-radius: 8px !important;
        padding: 10px !important;
        border: 1px solid var(--border-light) !important;
    }
    
    /* Make sure the drag and drop area is white */
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
        background-color: white !important;
        color: var(--text-dark) !important;
    }
    
    /* Target the specific black elements */
    section[data-testid="stSidebar"] .st-emotion-cache-1e10r4t,
    section[data-testid="stSidebar"] .st-emotion-cache-1x8hxfn,
    section[data-testid="stSidebar"] .st-emotion-cache-7ym5gk,
    section[data-testid="stSidebar"] .st-emotion-cache-dz5oo9 {
        background-color: white !important;
        color: var(--text-dark) !important;
    }
    
    /* Browse files button */
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] button,
    button.css-firdtp,
    button.st-emotion-cache-firdtp,
    .stButton button {
        background-color: var(--accent-brown) !important;
        color: white !important;
        border: none !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover,
    button.css-firdtp:hover,
    button.st-emotion-cache-firdtp:hover,
    .stButton button:hover {
        background-color: var(--accent-brown-hover) !important;
    }

    /* Chat Bubbles */
    .user-message, .assistant-message {
        background-color: var(--bubble-white);
        color: var(--text-dark);
        border: 1px solid var(--border-light);
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        display: inline-block;
        max-width: 80%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Align user messages to the right */
    .user-message-container {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 1rem;
    }
    
    .assistant-message-container {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 1rem;
    }
    
    /* Feature card styling */
    .feature-card {
        background-color: #F9F6F0;
        border-radius: 8px;
        padding: 15px;
        border-left: 3px solid var(--accent-brown);
        margin-bottom: 10px;
    }

    /* Radio buttons in white background */
    .stRadio [role="radiogroup"] {
        background-color: white !important;
        border-radius: 8px !important;
        padding: 10px !important;
        border: 1px solid var(--border-light) !important;
    }
    
    /* Checkbox in white background */
    .stCheckbox {
        background-color: white !important;
        border-radius: 8px !important;
        padding: 10px !important;
        border: 1px solid var(--border-light) !important;
    }
    
    /* Style the chat elements at the bottom */
    .stChatInput, div[data-testid="stChatInput"], footer, footer div, footer form, footer section, footer div > div, footer section > div, [data-testid="stChatInput"] > div, div.st-emotion-cache-16j374b, div.st-emotion-cache-90vs21, div.st-emotion-cache-1h2q2s2, .main > div:last-child, .css-90vs21, .css-1h2q2s2, .st-emotion-cache-1y4p8pa, div.css-1y4p8pa, [data-testid="chatInputFooterContainer"], div.st-emotion-cache-a3vehb, [data-baseweb="input"], [data-baseweb="base-input"], .st-emotion-cache-uq7eal, .st-emotion-cache-1vbkxwb, div.css-1vbkxwb, .st-emotion-cache-qcpyf5, .css-qcpyf5 {
        background-color: var(--bg-light-brown) !important;
        border-color: var(--border-light) !important;
    }
    
    /* FIX 1: Style the input field itself - ensure text is dark */
    .stChatInput input, 
    div[data-testid="stChatInput"] input, 
    [data-testid="stChatInput"] input, 
    input.st-emotion-cache-1ln6ewj, 
    div.st-emotion-cache-w6ut8h, 
    footer input, 
    footer input[type="text"], 
    [data-baseweb="input"] input, 
    [data-baseweb="base-input"] input, 
    div.st-emotion-cache-1n76uvr input, 
    div.st-emotion-cache-1n76uvr > input, 
    div.st-emotion-cache-qbxlf4,
    div[data-testid="stChatInputContainer"] input,
    section[data-testid="stChatMessageFooter"] input {
        background-color: white !important;
        border: 1px solid var(--border-light) !important;
        border-radius: 20px !important;
        color: var(--text-dark) !important;
        padding-left: 15px !important;
    }
    
    /* FIX 1: Ensure placeholder text in chat input is visible */
    ::placeholder,
    ::-webkit-input-placeholder,
    :-ms-input-placeholder,
    ::-moz-placeholder,
    :-moz-placeholder,
    div[data-testid="stChatInputContainer"] input::placeholder,
    section[data-testid="stChatMessageFooter"] input::placeholder {
        color: var(--text-dark) !important;
        opacity: 0.7 !important;
    }
    
    /* Fix button color at bottom */
    .stChatInput button, div[data-testid="stChatInput"] button, [data-testid="stChatInput"] button, footer button, button[data-testid="baseButton-secondary"], button.st-emotion-cache-1rg185l, button.css-1rg185l, div.st-emotion-cache-ocqkz7 button, [data-testid="SendButtonIcon"] {
        background-color: var(--accent-brown) !important;
        color: white !important;
    }
    
    /* All SVG elements for buttons */
    button svg, button svg path, [data-testid="SendButtonIcon"] svg, [data-testid="SendButtonIcon"] svg path, div.st-emotion-cache-ocqkz7 button svg, button.st-emotion-cache-1rg185l svg, div[data-testid="stChatInput"] button svg, div[data-testid="stChatInput"] button svg path {
        fill: white !important;
        color: white !important;
    }
    
    /* Change color of radio button icons */
    .st-emotion-cache-1k5z2r9 {
        background-color: var(--accent-brown) !important;
    }
    
    /* Change color of checkbox background when checked */
    .st-emotion-cache-j5hnr6 {
        background-color: var(--accent-brown) !important;
        border-color: var(--accent-brown) !important;
    }
    
    /* Change success banner to match theme */
    div[data-baseweb="notification"] {
        background-color: #D8C9B7 !important;
    }
    
    div[data-baseweb="notification"] [data-testid="stNotificationIcon"] {
        color: var(--accent-brown) !important;
    }
    
    /* Fix black background at the bottom completely */
    footer, footer section, footer section > div, footer form {
        background-color: var(--bg-light-brown) !important;
    }
    
    footer input, footer input[type="text"] {
        background-color: white !important;
    }
    
    footer button {
        background-color: var(--accent-brown) !important;
    }
    
    /* Target specifically the bottom search bar */
    .st-emotion-cache-183lzff, .css-183lzff, section[data-testid="stChatMessageFooter"], div.st-emotion-cache-1v0mbdj, div.css-1v0mbdj, div.st-emotion-cache-90vs21 {
        background-color: var(--bg-light-brown) !important;
    }
    
    .st-emotion-cache-183lzff input, .css-183lzff input, section[data-testid="stChatMessageFooter"] input {
        background-color: white !important;
        color: var(--text-dark) !important;
    }
    
    /* Even more specific targeting */
    footer.st-emotion-cache-1l21srw, footer.css-1l21srw,
    section.st-emotion-cache-6qob1r, section.css-6qob1r,
    div.st-emotion-cache-90vs21, div.css-90vs21 {
        background-color: var(--bg-light-brown) !important;
    }
    
    /* Target any remaining black areas */
    [data-testid="baseButton-secondary"], [data-testid="chatInputFooterContainer"],
    [data-testid="stChatFloatingInputContainer"], 
    div.st-emotion-cache-1qg05tj, div.css-1qg05tj,
    div.st-emotion-cache-1erivf3, div.css-1erivf3 {
        background-color: var(--bg-light-brown) !important;
    }

    /* ULTRA-SPECIFIC TARGETING FOR THE BLACK BOTTOM BAR */
    footer, 
    footer *, 
    footer div, 
    footer section, 
    footer form, 
    div[data-testid="stChatMessageFooter"],
    [data-testid="stChatMessageFooter"] *,
    [data-testid="stChatMessageFooter"],
    [data-testid="chatInputFooterContainer"],
    [data-testid="chatInputFooterContainer"] *,
    .stChatFloatingInputContainer,
    .stChatFloatingInputContainer *,
    div.st-emotion-cache-16j374b,
    div.st-emotion-cache-bl5ch0,
    div.st-emotion-cache-r421ms,
    div.st-emotion-cache-90vs21,
    div.st-emotion-cache-qfj7io,
    div.st-emotion-cache-19rxjzo,
    div.st-emotion-cache-cgxjdg,
    .css-16j374b,
    .css-bl5ch0,
    .css-r421ms,
    .css-90vs21,
    .css-qfj7io,
    .css-19rxjzo,
    .css-cgxjdg,
    .css-12ozfss,
    .main > div:last-child,
    .main div[data-testid="block-container"] > div:last-child,
    div[data-testid="stDecoration"],
    div[data-testid="stDecoration"] *,
    .css-5rimss,
    .st-emotion-cache-5rimss,
    .css-15tx938,
    .st-emotion-cache-15tx938,
    .css-9ycgxx,
    .st-emotion-cache-9ycgxx,
    .css-1dp5vir,
    .st-emotion-cache-1dp5vir,
    .css-1dx1gwv,
    .st-emotion-cache-1dx1gwv,
    .css-1x8cf1d,
    .st-emotion-cache-1x8cf1d,
    .css-36ebgw,
    .st-emotion-cache-36ebgw,
    .st-emotion-cache-eczf16,
    .css-eczf16,
    .st-emotion-cache-jkl5yg,
    .css-jkl5yg,
    .st-emotion-cache-16idsys,
    .css-16idsys {
        background-color: var(--bg-light-brown) !important;
        border-color: var(--border-light) !important;
    }
    
    /* Add overlay for any elements we might have missed */
    body::after {
        content: "";
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        height: 60px; /* Adjust height as needed to cover the black bar */
        background-color: var(--bg-light-brown) !important;
        z-index: -1;
        pointer-events: none;
    }
    
    /* Target the bottom message container specifically */
    section[data-testid="stChatMessageFooter"] {
        position: relative;
        z-index: 1000;
        background-color: var(--bg-light-brown) !important;
    }
    
    /* Target any inline styles */
    [style*="background-color: rgb(17, 17, 17)"],
    [style*="background-color: #111111"],
    [style*="background-color: black"],
    [style*="background-color: rgba(0, 0, 0"],
    [style*="background: rgb(17, 17, 17)"],
    [style*="background: #111111"],
    [style*="background: black"],
    [style*="background: rgba(0, 0, 0"] {
        background-color: var(--bg-light-brown) !important;
        background: var(--bg-light-brown) !important;
    }
    </style>
    """

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "api_key_validated" not in st.session_state:
        st.session_state.api_key_validated = False
    
    if "selected_pet" not in st.session_state:
        st.session_state.selected_pet = "All species"
    
    # Check API key validity
    is_valid, message = validate_together_api_key()
    if not is_valid:
        st.session_state.api_key_validated = False
        st.error(f"API Key Issue: {message}")
        st.session_state.api_key_error = message
        return
    
    # If API key is valid, initialize models
    if "embedding_model" not in st.session_state:
        try:
            st.session_state.embedding_model = EmbeddingModel()
            logger.info("Embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            st.error(f"Error initializing embedding model: {e}")
            st.session_state.embedding_model = None
    
    if "llm" not in st.session_state:
        try:
            st.session_state.llm = TogetherModel()
            # Test the API key with a simple request
            test_response = st.session_state.llm.simple_response("Hello")
            if "Error" in test_response:
                st.error("Together API key validation failed. Please check your .env file for a valid key.")
                st.session_state.api_key_validated = False
                return
            st.session_state.api_key_validated = True
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            st.error(f"Error initializing LLM: {e}")
            st.session_state.llm = None
            st.session_state.api_key_validated = False
            return
    
    if "vector_store" not in st.session_state:
        try:
            if st.session_state.embedding_model:
                # Create a vector store with persistence
                persist_dir = os.path.join(os.getcwd(), "vector_store")
                st.session_state.vector_store = VectorStore(
                    st.session_state.embedding_model,
                    persist_dir=persist_dir
                )
                logger.info("Vector store initialized successfully")
                
                # Load and process knowledge base documents
                if "knowledge_base_loaded" not in st.session_state:
                    with st.spinner("Loading knowledge base..."):
                        num_processed = load_knowledge_base_documents()
                        if num_processed > 0:
                            st.success(f"Loaded {num_processed} documents from knowledge base")
                        st.session_state.knowledge_base_loaded = True
            else:
                st.session_state.vector_store = None
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            st.error(f"Error initializing vector store: {e}")
            st.session_state.vector_store = None

def api_key_form():
    """Form for entering API key manually."""
    st.header("API Key Configuration")
    with st.form("api_key_form"):
        api_key = st.text_input("Enter your Together API Key:", type="password")
        submitted = st.form_submit_button("Submit")
        
        if submitted and api_key:
            # Test the API key
            try:
                os.environ["TOGETHER_API_KEY"] = api_key
                test_model = TogetherModel(api_key=api_key)
                test_response = test_model.simple_response("Hello")
                
                if "Error" in test_response:
                    st.error("Invalid API key. Please check and try again.")
                else:
                    st.success("API key validated successfully!")
                    st.session_state.api_key_validated = True
                    st.session_state.manual_api_key = api_key
                    st.rerun()
            except Exception as e:
                st.error(f"Error validating API key: {str(e)}")

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary location and return the path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        st.error(f"Error saving uploaded file: {e}")
        return None

def process_document(file_path, file_name):
    """Process document and add to vector store."""
    try:
        with st.spinner("Processing document..."):
            # Load document
            document_text = load_document(file_path)
            
            # Chunk document
            chunks = chunk_text(document_text)
            
            # Add to vector store
            st.session_state.vector_store.add_documents(chunks, file_name)
            
            return len(chunks)
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        st.error(f"Error processing document: {e}")
        return 0

def search_documents(query, top_k=5):
    """Search for relevant document chunks."""
    try:
        if st.session_state.vector_store:
            return st.session_state.vector_store.search(query, top_k=top_k)
        return []
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        st.error(f"Error searching documents: {e}")
        return []

def web_search(query: str, num_results: int = 5):
    """Perform a web search using Tavily API."""
    try:
        from utils.web_search import tavily_search
        
        # Use Tavily for search
        search_results = tavily_search(
            query=query, 
            search_depth="basic", 
            max_results=num_results
        )
        
        return search_results
    except Exception as e:
        logger.error(f"Error performing web search: {str(e)}")
        return []

def fetch_webpage_content(url: str, max_length: int = 3000):
    """Fetch and extract content from a webpage."""
    try:
        from bs4 import BeautifulSoup
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        # Get text
        text = soup.get_text(separator="\n")
        
        # Clean text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)
        
        # Truncate if needed
        if len(text) > max_length:
            text = text[:max_length] + "..."
            
        return text
    except Exception as e:
        logger.error(f"Error fetching webpage content: {str(e)}")
        return None

def generate_response(query, response_mode, selected_pet, use_web_search=True):
    """Generate response using RAG and/or web search."""
    try:
        # First search local documents
        # If a specific pet is selected, add it to the query for better retrieval
        search_query = query
        if selected_pet != "All species":
            search_query = f"{selected_pet} {query}"
            
        context = search_documents(search_query)
        
        # If web search is enabled and we don't have enough context, search the web
        web_results = []
        if use_web_search and (len(context) < 2):
            with st.spinner("Searching the web for additional information..."):
                search_query = query
                if selected_pet != "All species":
                    search_query = f"{selected_pet} {query}"
                
                search_results = web_search(search_query)
                
                # Process search results
                if search_results:
                    for result in search_results[:2]:  # Limit to top 2 results
                        # No need to fetch content as Tavily already provides it
                        snippet = result.get("snippet", "")
                        if snippet:
                            web_results.append(f"From {result['title']} ({result['link']}):\n{snippet}")
        
        # Combine local and web context
        all_context = context + web_results
        
        # Generate response
        system_message = f"You are a helpful pet care assistant providing accurate information about pets."
        
        if selected_pet != "All species":
            system_message += f" The user is specifically asking about {selected_pet}, so focus your response on that species."
        
        # Generate response
        if all_context:
            response = st.session_state.llm.generate_response(
                query, 
                context=all_context, 
                response_mode=response_mode,
                system_message=system_message
            )
        else:
            response = st.session_state.llm.generate_response(
                query,
                response_mode=response_mode,
                system_message=system_message
            )
            
        return response
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"I encountered an error: {str(e)}"

def display_feature_card(title, description):
    """Display a feature card using native Streamlit components."""
    with st.container():
        # Apply custom styling with markdown
        st.markdown(f"##### {title}")
        st.write(description)

def main():
    """Main application."""
    st.set_page_config(
        page_title=APP_TITLE, 
        page_icon="🐾", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # --- All your CSS is applied here ---
    st.markdown(custom_css(), unsafe_allow_html=True)
    # This includes your first aggressive styling block
    st.markdown("""
    <style>
        /* Aggressive possible fix for chat input text color */
        input, textarea, [contenteditable="true"] {
            color: black !important;
            -webkit-text-fill-color: black !important;
            font-weight: 600 !important;
        }
        /* Target specific Streamlit elements */
        .stChatInput input, 
        div[data-testid="stChatInput"] input, 
        [data-testid="stChatInput"] input, 
        input.st-emotion-cache-1ln6ewj, 
        div.st-emotion-cache-w6ut8h, 
        footer input, 
        footer input[type="text"], 
        [data-baseweb="input"] input, 
        [data-baseweb="base-input"] input, 
        div.st-emotion-cache-1n76uvr input, 
        div.st-emotion-cache-1n76uvr > input, 
        div.st-emotion-cache-qbxlf4,
        div[data-testid="stChatInputContainer"] input,
        section[data-testid="stChatMessageFooter"] input {
            color: black !important;
            -webkit-text-fill-color: black !important;
            text-shadow: 0 0 0 black !important;
        }
        /* FORCEFUL APPROACH: Use high-contrast color black on white */
        section[data-testid="stChatMessageFooter"] input,
        [data-testid="stChatInputContainer"] input {
            background-color: white !important;
            color: black !important;
            -webkit-text-fill-color: black !important;
            font-weight: 600 !important;
            text-shadow: none !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # --- Step 1: Initialize Session State ---
    if "system_ready" not in st.session_state:
        st.session_state.system_ready = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_pet" not in st.session_state:
        st.session_state.selected_pet = "All species"
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # --- Step 2: Check API Key and Initialize Backend ONCE ---
    if TOGETHER_API_KEY:
        if not st.session_state.system_ready:
            with st.spinner("Initializing models, please wait..."):
                try:
                    embedding_model = EmbeddingModel()
                    st.session_state.vector_store = VectorStore(embedding_model=embedding_model)
                    st.session_state.llm = TogetherModel()
                    st.session_state.system_ready = True
                    st.toast("System ready!")
                except Exception as e:
                    st.error(f"Failed to initialize. Check API key in secrets. Error: {e}")
                    st.session_state.system_ready = False
    else:
        st.session_state.system_ready = False

    # --- Step 3: Render UI Based on State ---
    
    # Define variables for sidebar controls to be used later
    selected_pet = st.session_state.selected_pet
    response_mode = "concise"
    use_web_search = True

    # Sidebar
    with st.sidebar:
        st.title("PetCare Companion")
        st.write("Your pet care assistant")
        st.markdown("---")
        
        if st.session_state.system_ready:
            st.subheader("Pet Type")
            selected_pet = st.selectbox(
                "Select pet type:", PET_SPECIES, 
                index=PET_SPECIES.index(st.session_state.selected_pet),
                label_visibility="collapsed"
            )
            st.session_state.selected_pet = selected_pet
            
            st.subheader("Response Style")
            response_mode = st.radio(
                "Select style:", list(RESPONSE_MODES.keys()), 
                format_func=lambda x: "Concise" if x == "concise" else "Detailed",
                horizontal=True, label_visibility="collapsed"
            )
            
            st.subheader("Web Search")
            use_web_search = st.checkbox("Enable web search", value=True)
            
            st.markdown("---")
            
            st.subheader("Knowledge Base")
            uploaded_file = st.file_uploader(
                "Upload documents", type=["pdf", "docx", "txt", "md", "csv"], label_visibility="collapsed"
            )
            
            if uploaded_file:
                st.write(f"File: {uploaded_file.name}")
                if st.button("Process Document"):
                    if st.session_state.vector_store:
                        file_path = save_uploaded_file(uploaded_file)
                        if file_path:
                            num_chunks = process_document(file_path, uploaded_file.name)
                            if num_chunks > 0:
                                st.success(f"Added {num_chunks} chunks")

    # Main content area
    if not st.session_state.system_ready:
        st.title("Welcome to PetCare Companion")
        st.warning("Please configure a valid Together API key to use this application.")
        st.info("The app owner must set the API key in the Streamlit Cloud secrets.")
        st.markdown("""
        ### How to get a Together API Key:
        1. Go to the [Together AI website](https://api.together.xyz/settings/api-keys)
        2. Sign in or create an account
        3. Create an API key 
        4. Add it to this app's secrets in the Streamlit Cloud dashboard
        """)
    else:
        # Main Chat Interface
        current_date = datetime.now().strftime("%B %d, %Y")
        st.caption(f"Today: {current_date}")
        
        st.markdown("""
        <div class="welcome-header">
            <h1>Welcome to PetCare Companion!</h1>
            <p>Your trusted source for pet care information</p>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.messages:
            st.write("Try asking things like: 'What can my cat eat?, 'Why is my dog barking at night?'")
            col1, col2 = st.columns(2)
            with col1:
                with st.container(): st.markdown("### Nutrition"); st.write("Dietary advice and recommendations")
                with st.container(): st.markdown("### Behavior"); st.write("Training tips and interpreting behavior")
            with col2:
                with st.container(): st.markdown("### Health & Wellness"); st.write("Understanding symptoms and care")
                with st.container(): st.markdown("### Seasonal Care"); st.write("Safety and seasonal advice")
        else:
            for message in st.session_state.messages:
                # Your original code used custom markdown for chat messages, 
                # but st.chat_message is the modern, recommended way.
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # All of your aggressive CSS and JS for the chat input is placed here
        st.markdown("""
        <style>
            /* CRITICAL TEXT COLOR FIX - Make text in input field BLACK */
            [data-testid="stChatInput"] input,
            div[data-testid="stChatInputContainer"] input,
            section[data-testid="stChatMessageFooter"] input,
            .stChatInput input,
            input[type="text"] {
                color: #000000 !important;
                font-weight: 500 !important;
            }
            /* Make placeholder text dark and visible */
            [data-testid="stChatInput"] input::placeholder,
            div[data-testid="stChatInputContainer"] input::placeholder,
            section[data-testid="stChatMessageFooter"] input::placeholder,
            .stChatInput input::placeholder,
            input[type="text"]::placeholder {
                color: #4A3B27 !important;
                opacity: 0.7 !important;
            }
        </style>
        <script>
            // Wait for page to load
            setTimeout(function() {
                var inputs = document.querySelectorAll('input');
                inputs.forEach(function(input) {
                    if(input.getAttribute('data-testid') === 'stChatInput') {
                        input.style.color = '#000000 !important';
                    }
                });
            }, 500);
        </script>
        """, unsafe_allow_html=True)
        
        # Handle new chat input
        if prompt := st.chat_input(f"Ask about {st.session_state.selected_pet.lower()} care..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = generate_response(prompt, response_mode, selected_pet, use_web_search)
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

def load_knowledge_base_documents():
    """Load and process all text files from the knowledge_base directory."""
    try:
        # Get the knowledge_base directory path
        kb_dir = os.path.join(os.getcwd(), "knowledge_base")
        
        # Check if directory exists
        if not os.path.exists(kb_dir):
            logger.warning(f"Knowledge base directory not found: {kb_dir}")
            return 0
        
        # Get all text files
        text_files = [f for f in os.listdir(kb_dir) if f.endswith('.txt')]
        
        if not text_files:
            logger.warning("No text files found in knowledge base directory")
            return 0
        
        # Count processed files
        processed_count = 0
        
        # Process each file
        for file_name in text_files:
            file_path = os.path.join(kb_dir, file_name)
            
            # Load document content
            document_text = load_document(file_path)
            
            # Chunk document
            chunks = chunk_text(document_text)
            
            if chunks:
                # Add to vector store
                st.session_state.vector_store.add_documents(chunks, file_name)
                processed_count += 1
                logger.info(f"Processed knowledge base document: {file_name} ({len(chunks)} chunks)")
        
        return processed_count
    except Exception as e:
        logger.error(f"Error loading knowledge base documents: {str(e)}")
        return 0

if __name__ == "__main__":
    main()