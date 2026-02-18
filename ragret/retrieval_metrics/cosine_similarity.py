# Cosine similarity measures the semantic similarity between the USER INPUT and each of the RETRIEVED DOCUMENTS
# It returns a list [] that contains the cosine similarity for all the documents in the retrieved context
import os
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
    def _compute_similarity(self, input: str) -> NDArray[np.float32]:
        try:
            vector_embedding = self.llm.get_embedding(input=input)
            return np.array(vector_embedding)
        except Exception as error:
            logging.error(f"Error on Cosine Similarity |_compute_similarity|: {error}")
            raise

# Main score for cosine similarity function. After making the user query and the retrieved documents into 
# vector embeddings, it calculates the cosine similarity of each of the retrieved contexts with the user query
    def score(self, user_query: str, retrieved_documents: list[str]) -> dict:
        cosine_similarity_matrix = []
        try:
            user_query_embedding = self._compute_similarity(user_query)
            # For every question generated, compare the vector embeddings of the question and the retrieved context
            for document in retrieved_documents:
                document_embedding = self._compute_similarity(document)
                cosine_similarity = np.dot(user_query_embedding, document_embedding) / (np.linalg.norm(user_query_embedding) * np.linalg.norm(document_embedding))
                cosine_similarity_matrix.append(float(cosine_similarity))
                #logging.info(cosine_similarity_matrix)
            return {
                "score": cosine_similarity_matrix
            }

        except Exception as error:
            logging.error(f"Error on Cosine Similarity: |score|: {error}")
            raise
