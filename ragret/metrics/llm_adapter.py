import requests
from openai import OpenAI
from anthropic import Anthropic

# LLMAdapter class for universal client initialization across tools
class LLMAdapter:
    def __init__(self, 
                 provider: str, 
                 api_key: str, 
                 ollama_url: str | None, 
                 model: str | None,
                 embedding_model: str | None):
        # Default cheap and fast models
        self.default_llm_models = {
            "openai": "gpt-4.1-nano-2025-04-14",
            "ollama": "gemma3:4b"
        }
        self.default_embeddings_models = {
            "openai": "text-embedding-3-small",
            "ollama": "nomic-embed-text"
        }
        self.provider = provider
        self.model = model or self.default_llm_models[self.provider]
        self.embedding_model = embedding_model
        
        # Check for the provider name and make the client instance accordingly
        if self.provider == "openai":
            try:
                self.openai_client = OpenAI(api_key=api_key)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

        elif self.provider == "ollama":
            self.base_url = ollama_url or "http://localhost:11434"
            self.session = requests.Session()

# Function that takes as input the prompt and based on the self.provider and the client
# it creates a response object by making an API request to the self.provider
    def generate(self, prompt: str) -> str:
        # For OpenAI
        if self.provider == "openai":
            try:
                response = self.openai_client.responses.create(
                    model=self.model,
                    input=prompt,
                )
                return response.output_text.strip()

            except Exception as e:
                raise RuntimeError(f"OpenAI API generation request failed: {e}")

        # For local Ollama models
        elif self.provider == "ollama":
            try:
                response = self.session.post(f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0,
                    },
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
                return data["response"].strip()

            except Exception as e:
                raise RuntimeError(
f"""Ollama generation request failed.
1. Check if you provided the correct URL and port (eg ollama_url="http://127.0.0.0:11434")\n
2. Check if you have gemma3:4b model installed (DEFAULT MODEL). 
If not pull it using: ollama pull gemma3:4b or use a different model you have by stating it on class initialzation as below:\n
Metric(provider="ollama", model="gpt-oss:20b")""")
        # Mandatory else statement for correct class initialization
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def get_embedding(self, input: str) -> list[float]:
        # For OpenAI
        if self.provider == "openai":
            try:
                response = self.openai_client.embeddings.create(
                    model= self.embedding_model or self.default_embeddings_models["openai"],
                    input=input,
                )
                return response.data[0].embedding

            except Exception as e:
                raise RuntimeError(f"OpenAI embedding request failed: {e}")

        # For local Ollama models
        elif self.provider == "ollama":
            try:
                response = self.session.post(
                    f"{self.base_url}/api/embed",
                    json={
                        "model": self.embedding_model or self.default_embeddings_models["ollama"],
                        "input": input,
                    },
                    timeout=60,
                )

                response.raise_for_status()
                data = response.json()
                return data["embeddings"][0]

            except Exception as e:
                raise RuntimeError(
f"""Ollama embedding request failed.\n
1. Check if you provided the correct URL and port(eg ollama_url="http://127.0.0.0:11434")\n
2. Check if you have nomic-embed-text model installed (DEFAULT MODEL). 
If not pull it using: ollama pull nomic-embed-text or use a different model you have by stating it on class initialzation as below:\n
Metric(provider="ollama", embedding_model="embeddinggemma")""")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

