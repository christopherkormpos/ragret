# “Does the answer cover the important information present in the retrieved context?”

# Context Recall measures how many of the relevant documents (or pieces of information) were successfully retrieved. 
# It focuses on not missing important results. 
# Higher recall means fewer relevant documents were left out. 
# In short, recall is about not missing anything important.
# To calculate this: 
# 1. Extract claims from the context
# 2. Check whether each context claim is covered by the answer
# 3. Compute recall using the formula:
# Context Recall = (Number of claims in the retrieved context) / (Total number of claims in referece)
import os
from ragret.utils.llm_adapter import LLMAdapter
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
                 embedding_model: str| None = None):
        
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
    def _claim_extractor(self, context) -> list[str]:
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
{context}
"""
        try:
            derived_response_claims = self.llm.generate(prompt)
            # Make the response a list that splits each sentence on '\n'
            claims = [c.strip() for c in derived_response_claims.split("$") if c.strip()]
            return claims
        
        except Exception as error:
            logging.error(f"Error on Context Recall: |_claim_extractor| : {error}")
            raise
        
# Function that takes the claim made from the _claim_extractor function and sees wether it is supported by the ANSWER 
# or not. If yes it returns the output to be added on the supported_claims list.
    def _claim_checker(self, claim, llm_answer) -> bool:
        prompt = f"""
You are checking whether an answer covers a factual claim.

Claim
{claim}

Answer:
{llm_answer}

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

# Main score context recall function. Calls both _claim_extractor and claim_checker and calculates the context recall 
# of a response using the fomula: context recall = len(supported context claims) / len(total context claims). 
# Supported claims refers to supported CONTEXT claims
    def score(self, context, llm_answer) -> dict:
        try:
            claims = self._claim_extractor(context)
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

            # Check every claim in the list that was returned from _claim_extractor
            for claim in claims:
                if self._claim_checker(claim, llm_answer):
                    supported_claims.append(claim)
                    #logging.info(f"supported: {supported_claims}")
                else:
                    unsupported_claims.append(claim)
                    #logging.info(f"unsupported: {unsupported_claims}")
            
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
            logging.error(f"Error on Context Recall: |_calculate_context_recall|: {error}")
            raise
