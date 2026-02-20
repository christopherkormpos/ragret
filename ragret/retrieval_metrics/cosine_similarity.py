import os
from concurrent.futures import ThreadPoolExecutor
from ragret.utils.llm_adapter import LLMAdapter
import numpy as np
from numpy.typing import NDArray
import logging

# Logging configuration for debugging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class CosineSimilarity:
    def __init__(self, 
                 provider:str, 
                 api_key: str | None = None, 
                 ollama_url: str | None = None, 
                 embedding_model: str| None = None) -> None:
        
        supported_clients = ["openai","ollama"]
        self.provider = provider
        self.api_key = api_key or os.getenv("API_KEY")
        
        #Optional: model, ollama_url (for ollama use)
        self.ollama_url = ollama_url
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
                              model=None,
                              embedding_model=self.embedding_model)

# Function that given a text input, it computes the vector embedding of the input and returns it  
    def _compute_vector_embedding(self, input: str) -> NDArray[np.float32]:
        try:
            vector_embedding = self.llm.get_embedding(input=input)
            return np.array(vector_embedding)
        except Exception as error:
            logging.error(f"Error on Cosine Similarity |_compute_vector_embedding|: {error}")
            raise

# Main score for cosine similarity function. After making the user query and the retrieved documents into 
# vector embeddings, it calculates the cosine similarity of each of the retrieved contexts with the user query
    def score(self, user_query: str, retrieved_documents: list[str]) -> dict:
        cosine_similarity_matrix = []
        try:
            # Create a pool of worker threads and using the .map() 
            # send the user query and the list of the retrieved documents to a separate thread simultaneously
            with ThreadPoolExecutor() as executor:
                all_embeddings = list(executor.map(self._compute_vector_embedding, [user_query] + retrieved_documents))

            # The returned result is a list containing the vector embeddings of the user query (on index 0)
            # and the vector embeddings of the n questions generated and the user input (from index 1 till n-1)
            user_query_embedding = all_embeddings[0]
            document_embeddings = all_embeddings[1:]
            
            # For each document calculate the cosine similarity to the user query and append it to the final matrix
            for document_embedding in document_embeddings:
                cosine_similarity = np.dot(user_query_embedding, document_embedding) / (np.linalg.norm(user_query_embedding) * np.linalg.norm(document_embedding))
                cosine_similarity_matrix.append(float(cosine_similarity))
            return {
                "score": cosine_similarity_matrix
            }

        except Exception as error:
            logging.error(f"Error on Cosine Similarity: |score|: {error}")
            raise
