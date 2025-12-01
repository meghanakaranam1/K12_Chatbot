"""
AI service integration with OpenAI (v1.0+ compatible)
"""
import os
from openai import OpenAI
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
import re


class OpenAIService:
    """Service for interacting with OpenAI API (v1.0+)"""
    
    def __init__(self):
        raw_api_key = os.getenv("OPENAI_API_KEY")
        if not raw_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Clean the API key - remove any whitespace and handle cases where
        # the env var might contain "OPENAI_API_KEY = sk-..." format
        self.api_key = raw_api_key.strip()
        
        # If the key starts with "OPENAI_API_KEY", extract just the key part
        if self.api_key.startswith("OPENAI_API_KEY"):
            # Try to extract the key after "="
            parts = self.api_key.split("=", 1)
            if len(parts) > 1:
                self.api_key = parts[1].strip()
        
        # Remove any quotes that might be around the key
        self.api_key = self.api_key.strip('"\'')
        
        # Validate the key format
        if not self.api_key.startswith("sk-"):
            print(f"⚠️  [WARNING] API key doesn't start with 'sk-'. Got: {self.api_key[:10]}...")
            print(f"⚠️  [WARNING] Original value was: {raw_api_key[:50]}...")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is empty after cleaning")
        
        # Configure OpenAI client with timeout and retry settings
        # Try multiple approaches for better reliability
        try:
            import httpx
            
            # Create custom HTTP client with extended timeout and connection pooling
            http_client = httpx.Client(
                timeout=httpx.Timeout(120.0, connect=30.0),  # 120s total, 30s connect
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
                follow_redirects=True
            )
            
            self.client = OpenAI(
                api_key=self.api_key,
                http_client=http_client,
                max_retries=3,  # Retry up to 3 times on failure
                timeout=120.0  # Also set timeout on OpenAI client
            )
            print("✅ OpenAI client initialized with custom HTTP client (httpx)")
        except (ImportError, TypeError) as e:
            # Fallback if httpx not available or http_client parameter not supported
            print(f"⚠️  Using default HTTP client (httpx import/config failed: {e})")
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    timeout=120.0,  # 120 second timeout
                    max_retries=3  # Retry up to 3 times on failure
                )
                print("✅ OpenAI client initialized with default HTTP client")
            except Exception as init_e:
                print(f"❌ Failed to initialize OpenAI client: {init_e}")
                raise
        self.default_model = "gpt-4o-mini"
        
        self.call_count = 0
        self.token_count = {"input": 0, "output": 0}
    
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: Optional[int] = None, 
        temperature: Optional[float] = None, 
        model: Optional[str] = None, 
        response_format: Optional[Dict] = None
    ) -> str:
        """
        Make a chat completion request to OpenAI
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens in response (optional)
            temperature: Sampling temperature 0.0-1.0 (optional)
            model: Override default model (optional)
            response_format: Response format like {"type": "json_object"} (optional)
        
        Returns:
            Response content as string
        """
        if model is None:
            model = self.default_model
        
        self.call_count += 1
        
        print(f"🔍 [DEBUG] ai_services: OpenAI chat request with {len(messages)} messages")
        print(f"🔍 [DEBUG] ai_services: Using model: {model}")
        
        try:
            params = {
                'model': model,
                'messages': messages
            }
            
            if response_format is not None:
                params['response_format'] = response_format
            
            if max_tokens is not None:
                params['max_tokens'] = max_tokens
                
            if temperature is not None:
                params['temperature'] = temperature
            
            response = self.client.chat.completions.create(**params)
            
            content = response.choices[0].message.content
            
            if hasattr(response, 'usage'):
                self.token_count["input"] += response.usage.prompt_tokens
                self.token_count["output"] += response.usage.completion_tokens
            
            print(f"🔍 [DEBUG] ai_services: OpenAI response received, length: {len(content)}")
            
            return content
            
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            
            # Get full exception details
            import traceback
            full_trace = traceback.format_exc()
            
            print(f"💥 [ERROR] ai_services: OpenAI request failed: {error_type}: {error_msg}")
            print(f"🔍 [DEBUG] Full error trace:\n{full_trace}")
            
            # Check for specific error types
            if "Connection" in error_type or "APIConnectionError" in error_type:
                print("⚠️  [NETWORK] Connection error detected. Possible causes:")
                print("   1. Network connectivity issue")
                print("   2. Firewall blocking OpenAI API")
                print("   3. Proxy configuration needed")
                print("   4. DNS resolution problem")
                print("   5. VPN or network restrictions")
                
                # Try to diagnose network connectivity
                try:
                    import socket
                    import ssl
                    
                    # Test basic connectivity
                    print("\n🔍 [DIAGNOSTIC] Testing network connectivity...")
                    
                    # Test DNS resolution
                    try:
                        socket.gethostbyname("api.openai.com")
                        print("   ✅ DNS resolution: OK")
                    except Exception as dns_e:
                        print(f"   ❌ DNS resolution failed: {dns_e}")
                    
                    # Test HTTPS connection
                    try:
                        context = ssl.create_default_context()
                        with socket.create_connection(("api.openai.com", 443), timeout=5) as sock:
                            with context.wrap_socket(sock, server_hostname="api.openai.com") as ssock:
                                print("   ✅ HTTPS connection: OK")
                    except Exception as conn_e:
                        print(f"   ❌ HTTPS connection failed: {conn_e}")
                        
                except Exception as diag_e:
                    print(f"   ⚠️  Could not run diagnostics: {diag_e}")
                    
            elif "timeout" in error_msg.lower():
                print("⚠️  [TIMEOUT] Request timed out. The request will be retried automatically.")
            elif "rate_limit" in error_msg.lower() or "429" in error_msg:
                print("⚠️  [RATE_LIMIT] Rate limit hit. Please wait a moment and try again.")
            elif "authentication" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
                print("⚠️  [AUTH] Authentication error. Check your OPENAI_API_KEY environment variable.")
            
            raise
    
    def fetch_url_content(self, url: str, max_length: int = 10000) -> str:
        """
        Fetch and extract text content from a URL
        
        Args:
            url: URL to fetch
            max_length: Maximum content length
        
        Returns:
            Extracted text content
        """
        print(f"🔍 [DEBUG] ai_services: Fetching remote content from: {url}")
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            
            if 'text/html' in content_type:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                    element.decompose()
                
                text = soup.get_text(separator='\n', strip=True)
                
                text = re.sub(r'\n\s*\n', '\n\n', text)
                text = re.sub(r' +', ' ', text)
                
                if len(text) > max_length:
                    text = text[:max_length] + "..."
                
                return text
            
            else:
                text = response.text[:max_length]
                return text
        
        except Exception as e:
            print(f"💥 [ERROR] ai_services: Failed to fetch URL: {str(e)}")
            return ""
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "call_count": self.call_count,
            "tokens": self.token_count,
            "estimated_cost": (
                self.token_count["input"] * 0.15 / 1_000_000 +
                self.token_count["output"] * 0.60 / 1_000_000
            )
        }


# Global instance
openai_service = OpenAIService()