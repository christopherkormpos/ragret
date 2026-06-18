import os
import logging
from ragret.utils.llm_adapter import LLMAdapter
from concurrent.futures import ThreadPoolExecutor

# Logging configuration for debugging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class TokenCounter:
    def __init__(self,
                 provider:str,
                 prompt_template: str,
                 api_key: str | None = None,
                 ollama_url: str | None = None,
                 model: str | None = None) -> None:
        supported_clients = ["openai","google","ollama"]
        self.provider = provider
        self.prompt_template = prompt_template
        self.api_key = api_key or os.getenv("API_KEY")
        
        #Optional: model, ollama_url (for ollama use)
        self.ollama_url = ollama_url
        self.model = model

        # self.provider must match any of the supported clients
        if self.provider not in supported_clients:
            raise ValueError(f"Incorrect LLM provider. Please choose from supported clients: {supported_clients}")
        if not self.api_key:
            raise ValueError("Please provide a valid API key")

        # Based on the provider. TokenCounter only counts tokens, so no embedding model is needed.
        self.llm = LLMAdapter(provider=self.provider,
                              api_key=self.api_key,
                              ollama_url=self.ollama_url,
                              model=self.model,
                              embedding_model=None)
    
# Function that, for a single dataset record, reconstructs the rag input from the prompt template
# and counts the input tokens (template + context + query) and the output tokens (llm_answer)
    def _count(self, record:dict) -> dict:
        try:
            context = "\n".join(record["retrieved_documents"])
            input_text = self.prompt_template.format(context=context, query=record["user_query"])
            input_tokens = self.llm.count_tokens(input_text)
            output_tokens = self.llm.count_tokens(record["llm_answer"])
            return {
                "input_tokens":input_tokens,
                "output_tokens":output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        except Exception as error:
            logging.error(f"Error on Token Counter |_count|: {error}")
            raise

# Main score token counter function. Calls _count function for each row of the dataset and after that completes
# it calculates the average tokens for input and output for the whole dataset.
    def score(self, dataset:list[dict]) -> dict:
        try:
            if not dataset:
                raise ValueError("Dataset cannot be empty")

            with ThreadPoolExecutor() as executor:
                rows = list(executor.map(self._count, dataset))

            n = len(rows)
            response = {
                "avg_input_tokens":sum(r["input_tokens"] for r in rows) / n,
                "avg_output_tokens":sum(r["output_tokens"] for r in rows) / n,
                "avg_total_tokens":sum(r["total_tokens"] for r in rows) / n,
            }
            return response
        except Exception as error:
            logging.error(f"Error on Token Counter |score|: {error}")
            raise