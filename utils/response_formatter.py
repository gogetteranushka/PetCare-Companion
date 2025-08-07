def format_response(content: str, mode: str = "concise") -> str:
    """Format response based on selected mode"""
    try:
        if mode.lower() == "concise":
            # Simple approach: For concise mode, limit to first couple sentences
            sentences = content.split('. ')
            if len(sentences) <= 3:
                return content
                
            # Take first 2-3 sentences for a quick summary
            return '. '.join(sentences[:3]) + '.'
        
        # For detailed mode, return the full content
        return content
    except Exception as e:
        print(f"Error formatting response: {e}")
        return content