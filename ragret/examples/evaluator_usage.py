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
    context_recall, 
    answer_relevancy, 
    context_precision
)

# Finally convert the results into a DataFrame and save the output in the current working directory.
df = pd.DataFrame(results)
df.to_csv("evaluation_results.csv", index=False)
print("Results saved to evaluation_results.csv")
