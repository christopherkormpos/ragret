import requests
from openai import OpenAI
import tiktoken
from google import genai
from google.genai import types
from ragret.utils.models import DEFAULT_LLM_MODELS, DEFAULT_EMBEDDING_MODELS

# LLMAdapter class for universal client initialization across tools
class LLMAdapter:
    def __init__(self, 
                 provider: str, 
                 api_key: str, 
                 ollama_url: str | None, 
                 model: str | None,
                 embedding_model: str | None) -> None:
        # Default cheap and fast models
        self.default_llm_models = DEFAULT_LLM_MODELS
        self.default_embeddings_models = DEFAULT_EMBEDDING_MODELS
        self.provider = provider
        self.model = model or self.default_llm_models[self.provider]
        self.embedding_model = embedding_model or self.default_embeddings_models[self.provider]
        
        # Check for the provider name and make the client instance accordingly
        if self.provider == "openai":
            try:
                self.openai_client = OpenAI(api_key=api_key)
                # The OpenAI SDK imports these sub-resources lazily on first access.
                # Touch them once here (main thread) so the imports complete before the
                # metrics call them concurrently from worker threads, which would otherwise
                # trigger a Python import deadlock (_DeadlockError) under ThreadPoolExecutor.
                _ = self.openai_client.responses
                _ = self.openai_client.embeddings
            except Exception as e:
                raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
        
        elif self.provider == "google":
            try:
                self.google_client = genai.Client(api_key=api_key)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Google client: {e}")
            
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
                    temperature=0,
                )
                return response.output_text.strip()

            except Exception as e:
                raise RuntimeError(f"OpenAI API generation request failed: {e}")
            
        # For Google
        elif self.provider == "google":
            try:
                response = self.google_client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0),
                )
                return response.text.strip() #type:ignore

            except Exception as e:
                raise RuntimeError(f"Google API generation request failed: {e}")
            
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
1. Check if you provided the correct URL and PORT (eg ollama_url="http://127.0.0.0:11434")
2. There is a possibility you have previous loaded models in memory and you dont have enough RAM/VRAM space for another model.
Please make sure to clear ollama memory using: ollama stop <model_name> on the machine running ollama
3. Check if you have gemma3:4b model installed (DEFAULT MODEL).
If not pull it using: ollama pull gemma3:4b or use a different model you have by stating it on class initialzation as below:
Metric(provider="ollama", model="gpt-oss:20b")""")
        # Mandatory else statement for correct class initialization
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

# Function that takes as input the prompt and based on the self.provider and the client
# it creates a vector embedding by making a request to the self.provider
    def get_embedding(self, input: str) -> list[float]:
        # For OpenAI
        if self.provider == "openai":
            try:
                response = self.openai_client.embeddings.create(
                    model= self.embedding_model,
                    input=input,
                )
                return response.data[0].embedding

            except Exception as e:
                raise RuntimeError(f"OpenAI embedding request failed: {e}")

        # For Google
        elif self.provider == "google":
            try:
                response = self.google_client.models.embed_content(
                    model= self.embedding_model,
                    contents=input,
                    config=types.EmbedContentConfig(output_dimensionality=1536),
                )
                return response.embeddings[0].values  # type: ignore

            except Exception as e:
                raise RuntimeError(f"Google embedding request failed: {e}")
            
        # For local Ollama models
        elif self.provider == "ollama":
            try:
                response = self.session.post(
                    f"{self.base_url}/api/embed",
                    json={
                        "model": self.embedding_model,
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
1. Check if you provided the correct URL and PORT (eg ollama_url="http://127.0.0.0:11434")
2. Check if you have nomic-embed-text model installed (DEFAULT MODEL). 
If not pull it using: ollama pull nomic-embed-text or use a different model you have by stating it on class initialzation as below:
Metric(provider="ollama", embedding_model="embeddinggemma")""")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def count_tokens(self, text:str)-> int:
        #OpenAI -> tiktoken
        if self.provider == "openai":
            try:
                enc = tiktoken.encoding_for_model(self.model)
            except KeyError:
                enc = tiktoken.get_encoding("o200k_base") # fallback for unmapped models
            except Exception as e:
                raise RuntimeError(f"OpenAI token count failed: {e}")
            return int(len(enc.encode(text)))
        
        #Google -> endpoint request
        elif self.provider == "google":
            try:
                response = self.google_client.models.count_tokens(
                    model=self.model,
                    contents=text,
                )
                return int(response.total_tokens) # type: ignore
            except Exception as e:
                raise RuntimeError(f"Google token count failed: {e}")
        
        #Ollama -> local api call
        elif self.provider == "ollama":
            try:
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model":self.model,
                        "prompt":text,
                        "stream": False,
                        "options": {"num_predict": 0}
                    },
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
                return int(data.get("prompt_eval_count",0))
            except Exception as e:
                raise RuntimeError(f"Ollama token counting failed: {e}")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")