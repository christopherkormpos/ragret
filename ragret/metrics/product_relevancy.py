# Do the products retrieved from the database match the user's request?

# Product Relevancy measures how many relevant products were retrieved
# OPTIONAL: Uses an LLM to decide product relevancy based on the search the user made
# To calculate this: 
# 1. Extract relevant products from the products retrieved
# 2. OPTIONAL: With LLM check whether each product is relevant to the user quer
# 3. Compute metric using the formula:
# Product Relevancy = (Number of believed relevant products) / (Total number of products)
import os
from llm_adapter import LLMAdapter
import logging

# Logging configuration for debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class ProductRelevancy:
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
        
# Function that takes as input the product_list retrieved by the system via database queries or similarity search
# Returns an integer that represent the correctly retrieved products based on an LLM response.        
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
                relevant_products = self.llm.generate(prompt)
                # Make the response a list that splits each sentence on '\n'
                return int(relevant_products)
            
            except Exception as error:
                logging.error(f"Error on Product Relevancy: |_relevant_product_extractor| : {error}")
                raise
            
# Main score product_relevancy function. Calls relevant_product_extractor and calculates the product_relevancy 
# of a response using the fomula: product_relevancy = relevant_products / total_products.
# Supported claims refers to supported RESPONSE claims    
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
            #logging.info(product_relevancy_response)
            return product_relevancy_response
        
        except Exception as error:
            logging.error(f"Error on Product Relevancy: |_calculate_product_relevancy|: {error}")
            raise
