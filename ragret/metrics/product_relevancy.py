# Do the products retrieved from the database match the user's request?

# Product Relevancy measures how many relevant products were retrieved
# OPTIONAL: Uses an LLM to decide product relevancy based on the search the user made
# To calculate this: 
# 1. Extract relevant products from the products retrieved
# 2. OPTIONAL: With LLM check whether each product is relevant to the user quer
# 3. Compute metric using the formula:
# Product Relevancy = (Number of believed relevant products) / (Total number of products)
import os
from openai import OpenAI
import logging

# Logging configuration for debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class ProductRelevancy:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("TEST_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OpenAI API key")
        
        #OpenAI client. You can use any client you wish as long as the response object is handled accordingly.
        self.client = OpenAI(api_key=self.api_key)
        
    def _relevant_product_extractor(self, product_list, user_query) -> int:
            prompt = f"""
You will be given a list of {len(product_list)} products in JSON format.
These products were returned from a database search.

Your task is to evaluate how many of these products are relevant to the user’s question.
using a single number between 0 and {len(product_list)}.

User question:
{user_query}

Products (JSON):
{product_list}

Return ONLY a single integer number:
"""
            try:
                logging.info(prompt)
                response = self.client.responses.create(model="gpt-4.1-nano-2025-04-14", input=prompt)
                relevant_products = response.output_text

                # Make the response a list that splits each sentence on '\n'
                return int(relevant_products)
            except Exception as error:
                logging.error(f"Error on Product Relevancy: |_relevant_product_extractor| : {error}")
                raise
    
    def score(self, user_query, products) -> dict:
        try:
            # Avoid dividing with zero
            if not products:
                product_relevancy_response = {
                "score": 1.0,
                "provided_products": 0,
                "relevant_products": 0
            }
                return product_relevancy_response

            relevant_products = self._relevant_product_extractor(products,user_query)
            logging.info(relevant_products)
            # Calculation formula
            product_relevancy = relevant_products / len(products)
            
            product_relevancy_response = {
                "score": product_relevancy,
                "provided_products": len(products),
                "relevant_products": relevant_products
            }
            logging.info(product_relevancy_response)
            return product_relevancy_response
        
        except Exception as error:
            logging.error(f"Error on Product Relevancy: |_calculate_product_relevancy|: {error}")
            raise
