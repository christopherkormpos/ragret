import requests
from openai import OpenAI
from anthropic import Anthropic

# LLMAdapter class for universal client initialization across tools
class LLMAdapter:
    def __init__(self, 
                 provider: str, 
                 api_key: str, 
                 ollama_url: str | None, 
                 model: str | None):
        # Default cheap and fast models
        self.default_models = {
            "openai": "gpt-4.1-nano-2025-04-14",
            "anthropic": "claude-haiku-4-5-20251001",
            "ollama": "gemma3:4b"
        }
        self.provider = provider
        self.model = model or self.default_models[self.provider]
        
        # Check for the provider name and make the client instance accordingly
        if self.provider == "openai":
            self.openai_client = OpenAI(api_key=api_key)

        elif self.provider == "anthropic":
            self.anthropic_client = Anthropic(api_key=api_key)

        elif self.provider == "ollama":
            if ollama_url:
                self.base_url = ollama_url
            else:
                self.base_url = "http://localhost:11434/api/generate"

# Function that takes as input the prompt and based on the self.provider and the client
# it creates a response object by making an API request to the self.provider
    def generate(self, prompt: str) -> str:
        # For OpenAI
        if self.provider == "openai":
            response = self.openai_client.responses.create(
                model=self.model,
                input=prompt
            )
            return response.output_text.strip()

        # For Anthropic
        elif self.provider == "anthropic":
            response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }
                ]
            )
            return "".join(block.text
                for block in response.content
                if block.type == "text"
            )

        # For local Ollama models
        elif self.provider == "ollama":
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0
                },
                timeout=60
            )
            return response.json()["response"].strip()
        # Mandatory else statement for correct class initialization
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for a single text"""
        # For OpenAI
        if self.provider == "openai":
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding

        # For local Ollama models
        elif self.provider == "ollama":
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.embedding_model,
                    "prompt": text
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["embedding"]
        
        # Anthropic doesn't support embeddings
        elif self.provider == "anthropic":
            raise ValueError("Anthropic does not support embeddings. Use OpenAI or Ollama.")
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

