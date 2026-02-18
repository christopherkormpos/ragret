<p align="center">
  <picture>
    <source srcset="./images/ragret-dark.png" media="(prefers-color-scheme: dark)">
    <img src="./images/ragret-light.png" alt="ragret logo" style="height:200px;">
  </picture>
</p>
<h2 align="center">
  RAG evaluation with fewer regrets.
</h2>

**ragret** is a stable, lightweight evaluation framework for Retrieval-Augmented Generation (RAG) systems that doesn’t change often.<br>
Its goal is simplicity: small, modular metrics that are easy to understand, extend, and integrate into existing pipelines. It was created out of the frustration with other frameworks constantly changing, making code from one version to the next useless. With **ragret**, the focus is clear: simple, implement-as-you-go metrics that you can rely on without having to rewrite your established code or digging through docs to figure out what changed overnight in your favorite framework.

## Metrics
**ragret** provides evaluation metrics for assessing different aspects of RAG system performance, including:
- Faithfulness
- Context Recall
- Context Precision
- Response Relevancy
- Product Relevancy (for systems related to product recommendation)

## Installation
Use pip to install the package
```bash
pip install ragret
```
Or clone the repository:
```bash
git clone https://github.com/christopherkormpos/ragret.git
cd ragret
```
## Configuration
You will need to create a .env file and set you enviromental variable "API_KEY" to your providers API key 
```bash
API_KEY=your-api-key-here
```
Or you can pass it directly during class initialization.
```python
Faithfulness(provider="openai", api_key="your-api-key-here")
```

## Supported providers
<p align="center">
  <picture>
    <source srcset="./images/supported-models-dark.png" media="(prefers-color-scheme: dark)">
    <img src="./images/supported-models-light.png" alt="supported models" style="height:200px;">
  </picture> 

**ragret** currently supports only two llm providers for generation and embeddings<br>
The default models for text generation are gpt-4.1-nano-2025-04-14 for OpenAI and gemma3:4b for Ollama<br>
For vector embeddings the default models are text-embeddings-small-3 for OpenAI and nomic-embed-text for Ollama.

## Usage
All metrics are exposed on upper level. Therefore they can be imported as susch:
```python
from ragret import Faithfulness, ContextRecall, ContextPrecision, ResponseRelevancy, ProductRelevancy
```
------------------ this is where each metric will go and its explantaion along with its formula -------------
## Example
```python
from ragret import Faithfulness

metric = Faithfulness()

result = metric.score(
    user_query="How can I contact Marmero?",
    contexts=["FAQ: You can contact us via email..."],
    llm_answer="You can contact us via email at marmerostudio@gmail.com."
)
print(result)
```
Expected output
```python
{
    "score": 1.0,
    "claims": [...],
    "supported_claims": [...],
    "unsupported_claims": [...]
}
```
--

## Contact
If you encounter any issues or bugs with the application, feel free to reach out to me:

- **Email**: christopher.kormpos@gmail.com
- **GitHub**: https://github.com/christopherkormpos
- **LinkedIn**: [LinkedIn Profile](https://www.linkedin.com/in/christopher-kormpos-27808b194/)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.