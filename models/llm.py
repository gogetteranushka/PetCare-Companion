import logging
from typing import List, Dict, Any, Optional, Tuple
from config.config import TOGETHER_API_KEY, LLM_MODEL
from together import Together

logger = logging.getLogger(__name__)

class TogetherModel:
    def __init__(self, api_key: str = TOGETHER_API_KEY, model_name: str = LLM_MODEL):
        """Initialize the Together AI model.
        
        Args:
            api_key: Together API key
            model_name: Model name to use
        """
        if not api_key:
            raise ValueError("Together API key is missing. Please check your .env file.")
            
        try:
            self.client = Together(api_key=api_key)
            self.model = model_name
            logger.info(f"Successfully initialized Together AI model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Together AI model: {str(e)}")
            raise Exception(f"Failed to initialize Together AI model: {e}")
    
    @staticmethod
    def validate_api_key(api_key: str) -> Tuple[bool, str]:
        """Validate if the API key is properly formatted.
        
        Args:
            api_key: Together API key to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not api_key:
            return False, "API key is empty"
        
        return True, "API key format appears valid"
            
    def generate_response(self, 
                         prompt: str, 
                         context: Optional[List[str]] = None,
                         response_mode: str = "detailed",
                         system_message: str = None) -> str:
        """Generate a response based on the prompt and optional context.
        
        Args:
            prompt: User query
            context: Optional list of context strings retrieved from the vector database
            response_mode: Whether to generate a concise or detailed response
            system_message: Optional custom system message
            
        Returns:
            Generated response as a string
        """
        try:
            # Build system prompt with context and response mode instruction
            if not system_message:
                system_message = "You are a helpful assistant providing accurate information."
            
            if response_mode == "concise":
                system_message += " Keep your responses brief and to the point."
            else:
                system_message += " Provide detailed and comprehensive responses."
                
            # Add context if available
            context_text = ""
            if context and len(context) > 0:
                context_text = "Here's relevant information to help answer the question:\n"
                for i, ctx in enumerate(context):
                    context_text += f"{i+1}. {ctx}\n"
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_message},
            ]
            
            # Add context as assistant message if available
            if context_text:
                messages.append({"role": "assistant", "content": context_text})
                
            # Add user prompt
            messages.append({"role": "user", "content": prompt})
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I encountered an error generating a response: {str(e)}"
            
    def simple_response(self, prompt: str) -> str:
        """Generate a simple response without context or formatting.
        Used for testing API key validity.
        
        Args:
            prompt: User query
            
        Returns:
            Generated response as a string
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in simple response: {str(e)}")
            return f"Error: {str(e)}"