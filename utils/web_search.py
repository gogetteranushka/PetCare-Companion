import requests
import logging
from typing import List, Dict, Any, Optional
from config.config import TAVILY_API_KEY

logger = logging.getLogger(__name__)

def tavily_search(query: str, search_depth: str = "basic", max_results: int = 5) -> List[Dict[str, Any]]:
    """Perform a web search using Tavily API.
    
    Args:
        query: The search query
        search_depth: 'basic' or 'deep' (basic is faster, deep is more comprehensive)
        max_results: Maximum number of results to return
        
    Returns:
        List of search results
    """
    try:
        if not TAVILY_API_KEY:
            logger.warning("Tavily API key is missing. Web search is disabled.")
            return []
            
        # Add pet-specific terms to the query if not already present
        if not any(term in query.lower() for term in ["pet", "dog", "cat", "animal"]):
            query = f"pet care {query}"
            
        url = "https://api.tavily.com/search"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TAVILY_API_KEY}"
        }
        
        payload = {
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_domains": [
                "aspca.org", 
                "akc.org", 
                "avma.org", 
                "petmd.com", 
                "vet.cornell.edu", 
                "vetmed.ucdavis.edu",
                "merckvetmanual.com",
                "cdc.gov",
                "aav.org",
                "arav.org"
            ]
        }
        
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        if not data or "results" not in data:
            logger.warning("No search results found in Tavily response")
            return []
            
        results = []
        for item in data["results"]:
            results.append({
                "title": item.get("title", ""),
                "link": item.get("url", ""),
                "snippet": item.get("content", ""),
                "source": item.get("source", "")
            })
            
        return results
    except Exception as e:
        logger.error(f"Error performing Tavily search: {str(e)}")
        return []

def fetch_webpage_content(url: str, max_length: int = 3000) -> Optional[str]:
    """Fetch and extract content from a webpage.
    
    Args:
        url: URL to fetch
        max_length: Maximum content length to return
        
    Returns:
        Extracted text content or None if failed
    """
    try:
        from bs4 import BeautifulSoup
        import requests
        
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
        
        # Break into lines and remove leading and trailing space
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Remove blank lines
        text = "\n".join(chunk for chunk in chunks if chunk)
        
        # Truncate if needed
        if len(text) > max_length:
            text = text[:max_length] + "..."
            
        return text
    except Exception as e:
        logger.error(f"Error fetching webpage content: {str(e)}")
        return None