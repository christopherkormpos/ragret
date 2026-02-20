# Import the metrics we want to use to evaluate our dataset, evaluator, and example dataset
from ragret import ContextRecall, ContextPrecision, AnswerRelevancy
from ragret.evaluators import Evaluator
from ragret.datasets import example_dataset
import pandas as pd

# Initialize metric classes with the desired provider.
# Other optional parameters:
# - api_key: provide your API key directly
# - ollama_url: for local LLM models
# - model: the LLM model name
# - embedding_model: the embedding model to use
context_recall = ContextRecall(provider="openai")
context_precision = ContextPrecision(provider="openai")
answer_relevancy = AnswerRelevancy(provider="openai")

# Create the evaluator with the dataset
# Use the calculate() method to evaluate the dataset with the selected metrics
results = Evaluator(example_dataset).calculate(
  context_recall,answer_relevancy,context_precision
  )

df = pd.DataFrame(results)
df.to_csv("evaluation_results.csv", index=False)
print("Results saved to evaluation_results.csv")








"""
# FOR DOCUMENTATION PURPOSES
results = []

for record in example_dataset:
    
    answer_relevancy_result = answer_relevancy.score(
        user_query=record["user_query"],
        llm_answer=record["llm_answer"])["score"]
    
    recall_result = context_recall.score(
        retrieved_documents=record["retrieved_context"],
        llm_answer=record["llm_answer"]
    )["score"]
    
    precision_result = context_precision.score(
        user_query=record["user_query"],
        retrieved_documents=record["retrieved_context"]
    )["score"]
    
    results.append({
        "context_precision": precision_result,
        "context_recall": recall_result,
        "answer_relevancy":answer_relevancy_result
    })

df = pd.DataFrame(results)
df.to_csv("evaluation_results.csv", index=False)
print("Results saved to evaluation_results.csv")
"""