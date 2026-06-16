# Import the metrics we want to use to evaluate our dataset, evaluator, and example dataset
from ragret import ContextRecall, ContextPrecision, AnswerRelevancy
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

results = []
# Use a simple for loop to evaluate the dataset with the selected metrics
for i,record in enumerate(example_dataset):
    print(f"Evaluating record {i+1} of {len(example_dataset)}")

    recall_result = context_recall.score(
        retrieved_documents=record["retrieved_documents"],
        ground_truth=record["ground_truth"]
    )["score"]
    
    precision_result = context_precision.score(
        user_query=record["user_query"],
        retrieved_documents=record["retrieved_documents"]
    )["score"]
    
    answer_relevancy_result = answer_relevancy.score(
        user_query=record["user_query"],
        llm_answer=record["llm_answer"])["score"]

    results.append({
        "context_precision": precision_result,
        "context_recall": recall_result,
        "answer_relevancy":answer_relevancy_result
    })

# Finally convert the results into a DataFrame and save the output in the current working directory.
df = pd.DataFrame(results)
df.to_csv("evaluation_results.csv", index=False)
print("Results saved to evaluation_results.csv")