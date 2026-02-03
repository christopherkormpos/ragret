# “If we inferred questions from the answer, would they match the user’s question?”

# The Anser Relevancy metric measures how relevant a response is to the user input
# It ranges from 0 to 1 with higher scores indicating better alignment with the user input
# To calculate this
# 1. Generate a set of artificial questions (default is 3) based on the response
#   These questions are designed to reflect the content of the response
# 2. Compute the cosine similarity between the embedding of the user input and the embedding of each generated question
# 3. Take the average of these cosine similarity scores to get the Answer Relevancy
import os
from openai import OpenAI
import logging
import numpy as np
from numpy.typing import NDArray

# Logging configuration for debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class AnswerRelevancy:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("TEST_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OpenAI API key")
        
        #OpenAI client. You can use any client you wish as long as the response object is handled accordingly.
        self.client = OpenAI(api_key=self.api_key)

# Function that takes an llm_answer as input 
# Returns n number of questions that could be questioned to get the same llm_answer output
    def _generate_artificial_questions(self, llm_answer, n) -> list[str]:
        prompt = f"""
You are generating questions that could be answered by the given response.

Rules:
- Generate {n} distinct questions.
- Each question should be fully answerable by the response.
- Questions should reflect the main points of the response.
- Output each question on a new line.
- Do NOT include explanations or numbering.

Response:
{llm_answer}
"""
        try:
            response = self.client.responses.create(model="gpt-4.1-nano-2025-04-14", input=prompt)
            derived_questions = response.output_text

            # Make the response a list that splits each sentence on '\n'
            artificial_questions = derived_questions.split("\n")
            for i in range(len(artificial_questions)):
                artificial_questions[i] = artificial_questions[i].rstrip()
            return artificial_questions
        
        except Exception as error:
            logging.error(f"Error on Answer Relevancy |_generate_artificial_questions|: {error}")
            raise

# Function that given a text input, it computes the vector embedding of the input and returns it    
    def _compute_similarity(self, input) -> NDArray[np.float32]:
        try:
            vector_embedding = self.client.embeddings.create(input=input, model="text-embedding-3-small").data[0].embedding
            return np.array(vector_embedding)
        except Exception as error:
            logging.error(f"Error on Answer Relevancy |_compute_similarity|: {error}")
            raise

# Main _calculate answer relevancy function. Calls _generate_artificial_questions function and after making
# vector embeddings of the user query and the answers generated calculates the answer relevancy metric
# using the fomula: answer_relevancy = Σ(cosine_similarity) / n
    def score(self, user_input, llm_answer, n=3) -> dict:
        try:
            artificial_questions = self._generate_artificial_questions(llm_answer,n)
            query_embedding = self._compute_similarity(user_input)
            cosine_similarity_sum = 0
            # For every question generated, compare the vector embeddings of the question and the answer
            for answer in artificial_questions:
                answer_embedding = self._compute_similarity(answer)
                cosine_similarity = np.dot(query_embedding, answer_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(answer_embedding))
                #logging.info(cosine_similarity)
                cosine_similarity_sum += cosine_similarity
            answer_relevancy = cosine_similarity_sum / n
            #logging.info(answer_relevancy)
            return {
                "score": answer_relevancy,
                "generated_questions": artificial_questions
            }
        
        except Exception as error:
            logging.error(f"Error on Answer Relevancy: |_calculate_answer_relevancy|: {error}")
            raise
