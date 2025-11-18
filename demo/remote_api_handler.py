"""
Remote API Handler for F1 Demo
Use this when models are too large for local hardware.
Supports Hugging Face Inference API, Azure, OpenAI, Anthropic, or custom endpoints.
"""

import requests
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RemoteAPIHandler:
    """Handles remote API calls instead of local model."""
    
    def __init__(self, api_type: str = "huggingface", api_key: str = "", api_url: str = "", model_id: str = ""):
        self.api_type = api_type.lower()
        self.api_key = api_key or os.getenv("HF_API_KEY", "") or os.getenv("HUGGINGFACE_API_KEY", "")
        self.api_url = api_url
        self.model_id = model_id or "mistralai/Mistral-7B-Instruct-v0.1"
        self.loaded = bool(self.api_key or self.api_url)
        
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 250,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate response using remote API."""
        if not self.loaded:
            return "Error: API not configured. Please set API key or URL in config.py"
        
        try:
            if self.api_type == "huggingface" or "huggingface" in (self.api_url or "").lower():
                return self._call_huggingface_api(prompt, system_prompt, max_new_tokens, temperature)
            elif self.api_type == "azure":
                return self._call_azure_api(prompt, system_prompt, max_new_tokens, temperature)
            elif self.api_type == "openai" or "openai" in (self.api_url or "").lower():
                return self._call_openai_api(prompt, system_prompt, max_new_tokens, temperature)
            elif self.api_type == "anthropic" or "anthropic" in (self.api_url or "").lower():
                return self._call_anthropic_api(prompt, system_prompt, max_new_tokens, temperature)
            elif self.api_url:
                return self._call_custom_api(prompt, system_prompt, max_new_tokens, temperature)
            else:
                return "Error: No valid API configuration found"
                
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return f"Error: {str(e)}"
    
    def _call_huggingface_api(self, prompt: str, system_prompt: Optional[str], max_tokens: int, temperature: float) -> str:
        """Call Hugging Face Inference API - FREE TIER AVAILABLE!"""
        # Format prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        else:
            full_prompt = prompt
        
        # Hugging Face Inference API endpoint
        api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False
            },
            "options": {
                "wait_for_model": True  # Wait if model is loading
            }
        }
        
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 503:
            # Model is loading, wait and retry
            logger.info("Model is loading, waiting...")
            import time
            time.sleep(10)
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        
        response.raise_for_status()
        data = response.json()
        
        # Handle different response formats
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("generated_text", "")
        elif isinstance(data, dict):
            return data.get("generated_text", data.get("text", ""))
        else:
            return str(data)
    
    def _call_azure_api(self, prompt: str, system_prompt: Optional[str], max_tokens: int, temperature: float) -> str:
        """Call Azure OpenAI or Azure ML endpoint."""
        if not self.api_url:
            return "Error: Azure API URL not configured"
        
        # Format prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        else:
            full_prompt = prompt
        
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "prompt": full_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        response.raise_for_status()
        data = response.json()
        return data.get("text", data.get("generated_text", data.get("response", "")))
    
    def _call_openai_api(self, prompt: str, system_prompt: Optional[str], max_tokens: int, temperature: float) -> str:
        """Call OpenAI API."""
        import openai
        
        if not self.api_key:
            return "Error: OpenAI API key not set"
        
        client = openai.OpenAI(api_key=self.api_key)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use cheaper model for demo
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content
    
    def _call_anthropic_api(self, prompt: str, system_prompt: Optional[str], max_tokens: int, temperature: float) -> str:
        """Call Anthropic API."""
        import anthropic
        
        if not self.api_key:
            return "Error: Anthropic API key not set"
        
        client = anthropic.Anthropic(api_key=self.api_key)
        
        system_msg = system_prompt or "You are a helpful Formula 1 assistant."
        
        response = client.messages.create(
            model="claude-3-haiku-20240307",  # Cheapest model
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_msg,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def _call_custom_api(self, prompt: str, system_prompt: Optional[str], max_tokens: int, temperature: float) -> str:
        """Call custom API endpoint."""
        payload = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        response = requests.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=30
        )
        
        response.raise_for_status()
        data = response.json()
        return data.get("response", data.get("text", "No response received"))

Remote API Handler for F1 Demo
Use this when models are too large for local hardware.
Supports Hugging Face Inference API, Azure, OpenAI, Anthropic, or custom endpoints.
"""

import requests
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RemoteAPIHandler:
    """Handles remote API calls instead of local model."""
    
    def __init__(self, api_type: str = "huggingface", api_key: str = "", api_url: str = "", model_id: str = ""):
        self.api_type = api_type.lower()
        self.api_key = api_key or os.getenv("HF_API_KEY", "") or os.getenv("HUGGINGFACE_API_KEY", "")
        self.api_url = api_url
        self.model_id = model_id or "mistralai/Mistral-7B-Instruct-v0.1"
        self.loaded = bool(self.api_key or self.api_url)
        
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 250,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate response using remote API."""
        if not self.loaded:
            return "Error: API not configured. Please set API key or URL in config.py"
        
        try:
            if self.api_type == "huggingface" or "huggingface" in (self.api_url or "").lower():
                return self._call_huggingface_api(prompt, system_prompt, max_new_tokens, temperature)
            elif self.api_type == "azure":
                return self._call_azure_api(prompt, system_prompt, max_new_tokens, temperature)
            elif self.api_type == "openai" or "openai" in (self.api_url or "").lower():
                return self._call_openai_api(prompt, system_prompt, max_new_tokens, temperature)
            elif self.api_type == "anthropic" or "anthropic" in (self.api_url or "").lower():
                return self._call_anthropic_api(prompt, system_prompt, max_new_tokens, temperature)
            elif self.api_url:
                return self._call_custom_api(prompt, system_prompt, max_new_tokens, temperature)
            else:
                return "Error: No valid API configuration found"
                
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return f"Error: {str(e)}"
    
    def _call_huggingface_api(self, prompt: str, system_prompt: Optional[str], max_tokens: int, temperature: float) -> str:
        """Call Hugging Face Inference API - FREE TIER AVAILABLE!"""
        # Format prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        else:
            full_prompt = prompt
        
        # Hugging Face Inference API endpoint
        api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False
            },
            "options": {
                "wait_for_model": True  # Wait if model is loading
            }
        }
        
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 503:
            # Model is loading, wait and retry
            logger.info("Model is loading, waiting...")
            import time
            time.sleep(10)
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        
        response.raise_for_status()
        data = response.json()
        
        # Handle different response formats
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("generated_text", "")
        elif isinstance(data, dict):
            return data.get("generated_text", data.get("text", ""))
        else:
            return str(data)
    
    def _call_azure_api(self, prompt: str, system_prompt: Optional[str], max_tokens: int, temperature: float) -> str:
        """Call Azure OpenAI or Azure ML endpoint."""
        if not self.api_url:
            return "Error: Azure API URL not configured"
        
        # Format prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        else:
            full_prompt = prompt
        
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "prompt": full_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        response.raise_for_status()
        data = response.json()
        return data.get("text", data.get("generated_text", data.get("response", "")))
    
    def _call_openai_api(self, prompt: str, system_prompt: Optional[str], max_tokens: int, temperature: float) -> str:
        """Call OpenAI API."""
        import openai
        
        if not self.api_key:
            return "Error: OpenAI API key not set"
        
        client = openai.OpenAI(api_key=self.api_key)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use cheaper model for demo
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content
    
    def _call_anthropic_api(self, prompt: str, system_prompt: Optional[str], max_tokens: int, temperature: float) -> str:
        """Call Anthropic API."""
        import anthropic
        
        if not self.api_key:
            return "Error: Anthropic API key not set"
        
        client = anthropic.Anthropic(api_key=self.api_key)
        
        system_msg = system_prompt or "You are a helpful Formula 1 assistant."
        
        response = client.messages.create(
            model="claude-3-haiku-20240307",  # Cheapest model
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_msg,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def _call_custom_api(self, prompt: str, system_prompt: Optional[str], max_tokens: int, temperature: float) -> str:
        """Call custom API endpoint."""
        payload = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        response = requests.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=30
        )
        
        response.raise_for_status()
        data = response.json()
        return data.get("response", data.get("text", "No response received"))

