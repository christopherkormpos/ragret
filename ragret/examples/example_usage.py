from ragret import ContextRecall
from ragret.datasets import example_dataset

metric = ContextRecall(provider="openai")

for record in example_dataset:
    result = metric.score(
        context=record["context"],
        llm_answer=record["llm_answer"]
    )
    print(result)
