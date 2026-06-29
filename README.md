<p align="center">
  <picture>
    <source srcset="https://raw.githubusercontent.com/christopherkormpos/ragret/main/images/ragret-dark.png" media="(prefers-color-scheme: dark)">
    <img src="https://raw.githubusercontent.com/christopherkormpos/ragret/main/images/ragret-light.png" alt="ragret logo" height="250">
  </picture>
</p>
<h2 align="center">
  RAG evaluation with fewer regrets.
</h2>

<div align="center">
  <a href="https://github.com/christopherkormpos/ragret/releases"><img alt="Latest release" src="https://img.shields.io/pypi/v/ragret?color=4c2c69&label=release"></a>&ensp;
  <a href="https://pypi.org/project/ragret/"><img alt="PyPi" src="https://img.shields.io/badge/Published_on-PyPI-09648a"></a>&ensp;
  <a href="https://github.com/christopherkormpos/ragret/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT_License-4cb636"></a>&ensp;
</div>
<br>

**ragret** is a lightweight, stable evaluation framework for Retrieval-Augmented Generation (RAG) systems that is designed for long-term consistency and only the necessary structural updates in mind.<br>
Its goal is simplicity: small, modular metrics that are easy to understand, extend, and integrate into existing pipelines. It was created out of the frustration with other frameworks constantly changing, making code from one version to the next obsolete and difficult to migrate. With **ragret**, the focus is clear: simple, implement-as-you-go metrics that you can rely on without having to rewrite your established code or digging through docs to figure out what changed overnight in your favorite framework.

## Metrics
**ragret** provides evaluation metrics for assessing different aspects of RAG system performance.<br>
It includes both LLM-based and non-LLM-based metrics, which are described with more detail in [METRICS](https://github.com/christopherkormpos/ragret/blob/main/docs/METRICS.md)<br>

- AnswerRelevancy
- Faithfulness
- ContextPrecision
- ContextRecall
- CosineSimilarity
- F1Score
- MRR
- TokenCounter *Custom (average tokens consumed on RAG system)*
- ProductRelevancy *Custom (for systems related to product recommendation)*

## Installation
Use `pip` to install the package
```bash
pip install ragret
```
Or
```bash
pip install git+https://github.com/christopherkormpos/ragret.git
```
Or you can clone the repository locally:
```bash
git clone https://github.com/christopherkormpos/ragret.git
cd ragret
```

## Supported providers
<p align="center">
  <picture>
    <source srcset="https://raw.githubusercontent.com/christopherkormpos/ragret/main/images/supported-models-dark.png" media="(prefers-color-scheme: dark)">
    <img src="https://raw.githubusercontent.com/christopherkormpos/ragret/main/images/supported-models-light.png" alt="supported models" style="width: 65%; max-height: 250px;">
  </picture> 

**ragret** currently supports three LLM providers for generation and embeddings.<br>
The default models for text generation are `gpt-4.1` for **OpenAI**, `gemini-3.1-flash-lite` for **Google** and `gemma3:4b` for **Ollama**.<br>
For vector embeddings, the default models are `text-embeddings-small-3` for **OpenAI**, `gemini-embedding-2` for **Google** and `nomic-embed-text` for **Ollama**.

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
  AnswerRelevancy,
  Faithfulness,
  ContextPrecision,
  ContextRecall,
  CosineSimilarity,
  F1Score,
  MRR,
  TokenCounter,
  ProductRelevancy
)
```

### Usage of the Evaluator class
The `Evaluator` is the standard way to run **ragret**. You pass in your dataset, along with the metrics you want, and it returns a result for each metric on every record. All metrics are used this way, with one exception: `TokenCounter`, which runs on its own and is covered in the next section.
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
  context_recall,
  answer_relevancy,
  context_precision
  )

# Finally convert the results into a DataFrame and save the output in the current working directory.
df = pd.DataFrame(results)
df.to_csv("evaluation_results.csv", index=False)
print("Results saved to evaluation_results.csv")
```
With the help of pandas, we convert our results into a DataFrame and save the output in the current working directory.<br>
> **Note:** It’s important that the dataset is structured like the example below so that the Evaluator class can work correctly and produce results.
```json
[
    {
        "user_query": "User Question 1",
        "retrieved_documents": ["Retrieved document text 1", 
                                "Retrieved document text 2", 
                                "Retrieved document text 3"],
        "llm_answer": "LLM Answer for Question 1",
        "ground_truth": "Ground-truth (reference) answer for Question 1"
    },
    {
        "user_query": "User Question 2",
        "retrieved_documents": ["Retrieved document text 1", 
                                "Retrieved document text 2", 
                                "Retrieved document text 3"],
        "llm_answer": "LLM Answer for Question 2",
        "ground_truth": "Ground-truth (reference) answer for Question 2"
    },...
]
```

### Usage of TokenCounter metric
`TokenCounter` is a standalone metric, and it does **not** go through the `Evaluator` class. It takes the whole dataset, rebuilds the input the way your RAG system would (using the `PROMPT` you provide) and returns the average input, output and total tokens across the dataset.
```python
from ragret import TokenCounter
from ragret.datasets import example_dataset

# Pass the SAME prompt your RAG system uses with {context} and {query} placeholders.
# Other optional parameters:
# - api_key: provide your API key directly
# - ollama_url: for local LLM models
# - model: the LLM model name
token_counter = TokenCounter(
    provider="openai",
    prompt_template="Answer the question using the context:\n{context}\n\nQuestion: {query}"
)

results = token_counter.score(example_dataset)
print(results)
```
> **Note:** TokenCounter reports raw token counts only and does not calculate cost. Multiply the averages by your model's per-token pricing to estimate spend, e.g. `avg_input_tokens * input_price + avg_output_tokens * output_price`.

## Contact
If you encounter any issues or bugs with the application, feel free to reach out to me:

- **Email**: christopher.kormpos@gmail.com
- **GitHub**: https://github.com/christopherkormpos
- **LinkedIn**: [LinkedIn Profile](https://www.linkedin.com/in/christopher-kormpos-27808b194/)

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/christopherkormpos/ragret/blob/main/LICENSE) file for details.