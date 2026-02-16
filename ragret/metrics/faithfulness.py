# “Are the claims in the answer supported by the retrieved context?”

# The Faithfulness metric measures how factually consistent a response is with the retrieved context. 
# It ranges from 0 to 1, with higher scores indicating better consistency.
# A response is considered faithful if all its claims can be supported by the retrieved context.
# To calculate this: 
# 1. Identify all the claims in the response. 
# 2. Check each claim to see if it can be inferred from the retrieved context.
# 3. Compute the faithfulness score using the formula:
# Faithfulnes = (Number of claims in the response supported by the retrieved context) / (Total Number of claims in response)
import os
import logging
from ragret.utils.llm_adapter import LLMAdapter
from dotenv import load_dotenv
load_dotenv()

# Logging configuration for debugging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class Faithfulness:
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
        
# Function that takes as input the RESPONSE generated from the LLM
# Returns a list of strings that represent the claims that are stated in a response.
    def _claim_extractor(self, answer) -> list[str]:
        prompt = f"""
Extract explicit factual claims an answer.  

Rules:
- Only extract claims explicitly stated in the text. Do NOT infer or assume.
- Split compound factual statements only if clearly separable.
- Each claim must be self-contained.
- Preserve the original language.
- Output a single string with claims separated by dollar sign ($).
- If no factual claims exist, output an empty string.

Answer:
{answer}
"""
        try:
            derived_response_claims = self.llm.generate(prompt)
            # Make the response a list that splits each sentence on '\n'
            claims = [c.strip() for c in derived_response_claims.split("$") if c.strip()]
            return claims
        
        except Exception as error:
            logging.error(f"Error on Faithfulness: |_claim_extractor| : {error}")
            raise
        
# Function that takes the claim made from the _claim_extractor function 
# and decides wether it is supported from the retrieved CONTEXT or not. 
# If yes it returns the output to be added on the supported_claims list.
    def _claim_checker(self, claim, context) -> bool:
        prompt = f"""
You are checking whether a claim is supported by the given context.
Context:
{context}

Claim:
{claim}

Decide if the claim is:
- SUPPORTED: directly stated or clearly inferred
- NOT_SUPPORTED: not mentioned or insufficient information

Answer with exactly one label:
SUPPORTED
NOT_SUPPORTED
"""
        try:
            check_result = self.llm.generate(prompt)
            return check_result.strip().upper() == "SUPPORTED"
        except Exception as error:
            logging.error(f"Error on Faithfulness: |_claim_checker| : {error}")
            raise    
        
# Main score faithfulness function. Calls both _claim_extractor and claim_checker and calculates the faithfulness 
# of a response using the fomula: 
# faithfulness = len(supported llm_response claims) / len(llm_response claims).
# Supported claims refers to supported RESPONSE claims
    def score(self, context, llm_answer) -> dict:
        try:
            claims = self._claim_extractor(llm_answer)
            #logging.info(f"Claims: {claims}")
            # Avoid dividing with zero
            if not claims:
                faithfulness_response = {
                    "score": 1.0,
                    "claims": [],
                    "supported_claims": [],
                    "unsupported_claims": [],
                }
                return faithfulness_response
            
            supported_claims = []
            unsupported_claims = []
            
            # Check every claim in the list that was returned from _claim_extractor
            for claim in claims:
                if self._claim_checker(claim, context):
                    supported_claims.append(claim)
                    #logging.info(f"supported: {supported_claims}")
                else:
                    unsupported_claims.append(claim)
                    #logging.info(f"unsupported: {unsupported_claims}")
            
            # Calculation formula
            faithfulness = len(supported_claims) / len(claims)
            
            faithfulness_response = {
                "score": faithfulness,
                "claims": claims,
                "supported_claims": supported_claims,
                "unsupported_claims": unsupported_claims,
            }
            #logging.info(faithfulness_response)
            return faithfulness_response
        
        except Exception as error:
            logging.error(f"Error on Faithfulness: |_calculate_faithfulness|: {error}")
            raise
