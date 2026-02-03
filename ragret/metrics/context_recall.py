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
from openai import OpenAI
import logging

# Logging configuration for debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class ContextRecall:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("TEST_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OpenAI API key")
        
        #OpenAI client. You can use any client you wish as long as the response object is handled accordingly.
        self.client = OpenAI(api_key=self.api_key)
        
# Function that takes as input the CONTEXT retrieved from the vector db that is present in the data samples and returns 
# a list of strings that represent the claims that are stated in the context.
    def _claim_extractor(self, context) -> list[str]:
        prompt = f"""
You are extracting factual claims from retrieved context.

Rules:
- Extract ONLY factual, checkable claims.
- Split compound sentences into atomic claims.
- Output each claim on a new line.
- Ignore background, examples, or rhetorical text.
- Each claim must be self-contained.
- Output a single string with claims separated by '\n'.
- Do NOT include explanations or numbering.

Context:
{context}
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
            response = self.client.responses.create(model="gpt-4.1-nano-2025-04-14", input=prompt)
            check_result = response.output_text
            return check_result == "SUPPORTED"
        except Exception as error:
            logging.error(f"Error on Context Recall: |_claim_checker| : {error}")
            raise

# Main calculate context recall function. Calls both _claim_extractor and claim_checker and calculates the context recall 
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