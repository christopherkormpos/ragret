# Context Precision measures how much of the retrieved context is actually useful for answering the user query.
# It ranges from 0 to 1, where higher scores indicate that the retrieved context is highly relevant to the query.
# 
# To calculate context precision:
# 1. Extract factual claims from the retrieved context using an LLM or other method.
# 2. Compare each claim against the user query to see if it is relevant for answering the query.
# 3. Compute Context Precision using the formula:
# Context Precision = (Number of supported claims) / (Total number of retrieved claims)
import os
from concurrent.futures import ThreadPoolExecutor
from ragret.utils.llm_adapter import LLMAdapter
import logging

# Logging configuration for debugging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class ContextPrecision:
    def __init__(self, 
                 provider:str, 
                 api_key: str | None = None, 
                 ollama_url: str | None = None, 
                 model: str | None = None,
                 embedding_model: str| None = None) -> None:
        
        supported_clients = ["openai","ollama"]
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
        
# Function that takes as input the CONTEXT retrieved from the vector db that is present in the data samples and returns 
# a list of strings that represent the claims that are stated in the context.
    def _claim_extractor(self, retrieved_documents: list[str]) -> list[str]:
        prompt = f"""
Extract explicit factual claims from retrieved context.  

Rules:
- Only extract claims explicitly stated in the text. Do NOT infer or assume.
- Split compound factual statements only if clearly separable.
- Each claim must be self-contained.
- Preserve the original language.
- Output a single string with claims separated by dollar sign ($).
- If no factual claims exist, output an empty string.

Context:
{retrieved_documents}
"""
        try:
            derived_response_claims = self.llm.generate(prompt)
            # Make the response a list that splits each sentence on '\n'
            claims = [c.strip() for c in derived_response_claims.split("$") if c.strip()]
            return claims
        except Exception as error:
            logging.error(f"Error on Context Precision: |_claim_extractor| : {error}")
            raise
        
# Function that takes the claim made from the _claim_extractor function and sees whether a retrieved context claim is 
# relevant for answering the user query
    def _claim_checker(self, claim: str, user_query: str) -> bool:
        prompt = f"""
You are checking whether a retrieved context claim is relevant to a user question.

Context Claim:
{claim}

User Question:
{user_query}

Decide if the claim is:
- SUPPORTED: clearly states or implies the claim
- NOT_SUPPORTED: does not mention or imply the claim

Answer with exactly one label:
SUPPORTED
NOT_SUPPORTED
"""
        try:
            check_result = self.llm.generate(prompt)
            return check_result.strip().upper() == "SUPPORTED"
        except Exception as error:
            logging.error(f"Error on Context Recall: |_claim_checker| : {error}")
            raise

# Main score context recall function. Calls both _claim_extractor and claim_checker and calculates the context precision 
# of a response using the fomula: context precision = len(supported context claims) / len(total context claims).
# Supported claims refers to supported CONTEXT claims
    def score(self, user_query: str, retrieved_documents: list[str]):
        try:
            claims = self._claim_extractor(retrieved_documents)
            #logging.info(f"Claims: {claims}")
            # Avoid dividing with zero
            if not claims:
                context_precision_response = {
                    "score": 1.0,
                    "claims": [],
                    "supported_claims": [],
                    "unsupported_claims": [],
                }
                return context_precision_response
            
            supported_claims = []
            unsupported_claims = []

            # Create a pool of worker threads and using the .map() send each of the claims to a separate thread simultaneously
            with ThreadPoolExecutor() as executor:
                checked = executor.map(lambda claim: self._claim_checker(claim, user_query), claims) #boolean values returned
            
            # for each claim and its coresponding result from the ThreadPoolExecutor
            for claim, is_supported in zip(claims, checked): 
                if is_supported:
                    supported_claims.append(claim)
                else:
                    unsupported_claims.append(claim)
            
            # Calculation formula
            context_precision = len(supported_claims) / len(claims)
            
            context_precision_response = {
                "score": context_precision,
                "claims": claims,
                "supported_claims": supported_claims,
                "unsupported_claims": unsupported_claims,
            }
            #logging.info(context_precision_response)
            return context_precision_response
        
        except Exception as error:
            logging.error(f"Error on Context Recall: |score|: {error}")
            raise