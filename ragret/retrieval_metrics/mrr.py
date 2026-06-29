import os
from concurrent.futures import ThreadPoolExecutor
from ragret.utils.llm_adapter import LLMAdapter
from ragret.utils.claim_utils import check_document_relevance
import logging

# Logging configuration for debugging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class MRR:
    def __init__(self,
                 provider:str,
                 api_key: str | None = None,
                 ollama_url: str | None = None,
                 model: str | None = None,
                 embedding_model: str| None = None) -> None:

        supported_clients = ["openai","google","ollama"]
        self.provider = provider
        self.api_key = api_key or os.getenv("API_KEY")

        #Optional: model, ollama_url (for ollama use)
        self.ollama_url = ollama_url
        self.model = model
        self.embedding_model = embedding_model

        # self.provider must match any of the supported clients
        if self.provider not in supported_clients:
            raise ValueError(f"Incorrect LLM provider. Please choose from supported clients: {supported_clients}")
        if not self.api_key:
            raise ValueError("Please provide a valid API key")

        # Based on the provider
        self.llm = LLMAdapter(provider=self.provider,
                              api_key=self.api_key,
                              ollama_url=self.ollama_url,
                              model=self.model,
                              embedding_model=self.embedding_model)

# Main score mrr function. Calculates the rr of retrieved_documents using the fomula: 
# rr = sum( 1 / position of first relevant answer).
    def score(self, retrieved_documents: list[str], user_query: str) -> dict:
        try:
            if not retrieved_documents:
                return {"score": 0.0, "first_relevant_rank": None, "relevance": []}
            with ThreadPoolExecutor() as executor:
                relevance = list(executor.map(
                    lambda doc: check_document_relevance(self.llm, doc, user_query),
                retrieved_documents))
            
            first_relevant_rank = None
            for index, is_relevant in enumerate(relevance):
                if is_relevant:
                    first_relevant_rank = index + 1
                    break
            
            reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
            return {
                "score": reciprocal_rank,
                "first_relevant_rank": first_relevant_rank,
                "relevance": relevance
            }
            
        except Exception as error:
            logging.error(f"Error on MRR: |score|: {error}")
            raise
    
    # mean function calculate the MRR (RR* (1 / len(user_queries))
    @staticmethod
    def mean(reciprocal_ranks: list[float]) -> dict:
        if not reciprocal_ranks:
            return {"score": 0.0}
        return {"score": sum(reciprocal_ranks)/ len(reciprocal_ranks)}