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
from openai import OpenAI
import logging

# Logging configuration for debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class Faithfulness:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("TEST_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OpenAI API key")
        
        #OpenAI client. You can use any client you wish as long as the response object is handled accordingly.
        self.client = OpenAI(api_key=self.api_key)
        
# Function that takes as input the RESPONSE generated from the LLM that is present in the data samples and returns 
# a list of strings that represent the claims that are stated in a response.
    def _claim_extractor(self, answer) -> list[str]:
        prompt = f"""
You are extracting factual claims from an answer.

Rules:
- Extract ONLY factual, checkable claims.
- Split compound sentences into atomic claims.
- Output each claim on a new line.
- Ignore opinions, advice, or speculation.
- Each claim must be self-contained.
- Output a single string with claims separated by '\n'.
- Do NOT include explanations or numbering.

Answer:
{answer}
"""
        try:
            response = self.client.responses.create(model="gpt-4.1-nano-2025-04-14", input=prompt)
            derived_response_claims = response.output_text

            # Make the response a list that splits each sentence on '\n'
            claims = derived_response_claims.split("\n")
            for i in range(len(claims)):
                claims[i] = claims[i].rstrip()
            return claims
        except Exception as error:
            logging.error(f"Error on Faithfulness: |_claim_extractor| : {error}")
            raise
        
# Function that takes the claim made from the _claim_extractor function and sees wether it is supported from the 
# retrieved CONTEXT or not. If yes it returns the output to be added on the supported_claims list.
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
            response = self.client.responses.create(model="gpt-4.1-nano-2025-04-14", input=prompt)
            check_result = response.output_text
            return check_result == "SUPPORTED"
        except Exception as error:
            logging.error(f"Error on Faithfulness: |_claim_checker| : {error}")
            raise    
        
# Main calculate faithfulness function. Calls both _claim_extractor and claim_checker and calculates the faithfulness 
# of a response using the fomula: faithfulness = len(supported llm_response claims) / len(llm_response claims).
# Supported claims refers to supported RESPONSE claims
    def score(self, llm_answer, context) -> dict:
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
