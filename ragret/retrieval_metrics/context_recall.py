import os
from concurrent.futures import ThreadPoolExecutor
from ragret.utils.llm_adapter import LLMAdapter
from ragret.utils.claim_utils import extract_claims_from_ground_truth, check_claim_against_context
import logging

# Logging configuration for debugging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class ContextRecall:
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

# Main score context recall function. Calculates the context recall
# of a response using the fomula: context recall = len(supported ground-truth claims) / len(total ground-truth claims).
# Supported claims refers to GROUND-TRUTH claims that are supported by the retrieved context.
    def score(self, retrieved_documents: list[str], ground_truth: str) -> dict:
        try:
            claims = extract_claims_from_ground_truth(self.llm, ground_truth)
            #logging.info(f"Claims: {claims}")
            # Avoid dividing with zero
            if not claims:
                context_recall_response = {
                    "score": 1.0,
                    "claims": [],
                    "supported_claims": [],
                    "unsupported_claims": [],
                }
                return context_recall_response

            supported_claims = []
            unsupported_claims = []

            # Create a pool of worker threads and using the .map() send each of the claims to a separate thread simultaneously
            with ThreadPoolExecutor() as executor:
                checked = executor.map(lambda claim: check_claim_against_context(self.llm, claim, retrieved_documents), claims)

            # for each claim and its coresponding result from the ThreadPoolExecutor
            for claim, is_supported in zip(claims, checked):
                if is_supported:
                    supported_claims.append(claim)
                else:
                    unsupported_claims.append(claim)

            # Calculation formula
            context_recall = len(supported_claims) / len(claims)

            context_recall_response = {
                "score": context_recall,
                "claims": claims,
                "supported_claims": supported_claims,
                "unsupported_claims": unsupported_claims,
            }
            #logging.info(context_recall_response)
            return context_recall_response

        except Exception as error:
            logging.error(f"Error on Context Recall: |score|: {error}")
            raise
