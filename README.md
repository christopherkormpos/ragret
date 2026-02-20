<p align="center">
  <picture>
    <source srcset="./images/ragret-dark.png" media="(prefers-color-scheme: dark)">
    <img src="./images/ragret-light.png" alt="ragret logo" style="height:200px;">
  </picture>
</p>
<h2 align="center">
  RAG evaluation with fewer regrets.
</h2>

<div align="center">
  <a href="https://img.shields.io/badge/release-v0.0.1-4c2c69"><img alt="Latest release" src="https://img.shields.io/badge/release-v0.0.1-4c2c69"></a>&ensp;
  <a href="https://img.shields.io/badge/Made_with-Python-09648a"><img alt="Made with Python" src="https://img.shields.io/badge/Made_with-Python-09648a"></a>&ensp;
  <a href="https://img.shields.io/badge/License-MIT_License-4cb636"><img alt="License" src="https://img.shields.io/badge/License-MIT_License-4cb636"></a>&ensp;
  <a href="https://img.shields.io/github/repo-size/christopherkormpos/ragret?color=a0c5f0"><img alt="repo size" src="https://img.shields.io/github/repo-size/christopherkormpos/ragret?color=a0c5f0"></a>
</div>

**ragret** is a lightweight, stable evaluation framework for Retrieval-Augmented Generation (RAG) systems that is designed for long-term consistency and only the necessary structural updates in mind.<br>
Its goal is simplicity: small, modular metrics that are easy to understand, extend, and integrate into existing pipelines. It was created out of the frustration with other frameworks constantly changing, making code from one version to the next obsolete and difficult to migrate. With **ragret**, the focus is clear: simple, implement-as-you-go metrics that you can rely on without having to rewrite your established code or digging through docs to figure out what changed overnight in your favorite framework.

## Metrics
**ragret** provides evaluation metrics for assessing different aspects of RAG system performance.<br>
It includes both LLM-based and non-LLM-based metrics, which are described with more detail in [METRICS](docs/METRICS.md)<br>

- AnswerRelevancy
- Faithfulness
- ContextPrecision
- ContextRecall
- CosineSimilarity
- F1Score
- ProductRelevancy (for systems related to product recommendation) *Custom*

## Installation
Use `pip` to install the package
```bash
pip install ragret
```
Or clone the repository locally:
```bash
git clone https://github.com/christopherkormpos/ragret.git
cd ragret
```

## Supported providers
<p align="center">
  <picture>
    <source srcset="./images/supported-models-dark.png" media="(prefers-color-scheme: dark)">
    <img src="./images/supported-models-light.png" alt="supported models" style="height:200px;">
  </picture> 

**ragret** currently supports only two LLM providers for generation and embeddings.<br>
The default models for text generation are `gpt-4.1-nano` for **OpenAI** and `gemma3:4b` for **Ollama**.<br>
For vector embeddings, the default models are `text-embeddings-small-3` for **OpenAI** and `nomic-embed-text` for **Ollama**.

## Basic Configuration
You will need to create a `.env` file and set your enviromental variable "API_KEY" to your providers API key (if you are using external LLMs)
```bash
API_KEY=your-api-key-here
```
Or you can pass it directly during class initialization.
```python
Faithfulness(provider="openai", api_key="your-api-key-here")
```

## Usage
All metrics are exposed on upper level. Therefore they can be imported as such:
```python
from ragret import (
  AnswerRelevancy
  Faithfulness
  ContextPrecision
  ContextRecall
  CosineSimilarity
  F1Score
  ProductRelevancy
)
```
Usage may vary depending on what you want to do. There are two ways to evaluate your dataset.

### Use case 1 (Common - Simpler - Faster): Use the Evaluator class
```python
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
```
With the help of pandas, we can convert our results into a DataFrame and save the output in the current working directory.<br>
> **Note:** It’s important that the dataset is structured like the example below so that the Evaluator class can work correctly and produce results.
```json
[
    {
        "user_query": "User Question 1",
        "retrieved_documents": ["Retrieved document text 1", 
                                "Retrieved document text 2", 
                                "Retrieved document text 3"],
        "llm_answer": "LLM Answer for Question 1"
    },
    {
        "user_query": "User Question 2",
        "retrieved_documents": ["Retrieved document text 1", 
                                "Retrieved document text 2", 
                                "Retrieved document text 3"],
        "llm_answer": "LLM Answer for Question 2"
    },...
]
```

### Use case 2 : Use the metrics classes directly
```python
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
        llm_answer=record["llm_answer"]
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
```

## Contact
If you encounter any issues or bugs with the application, feel free to reach out to me:

- **Email**: christopher.kormpos@gmail.com
- **GitHub**: https://github.com/christopherkormpos
- **LinkedIn**: [LinkedIn Profile](https://www.linkedin.com/in/christopher-kormpos-27808b194/)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.