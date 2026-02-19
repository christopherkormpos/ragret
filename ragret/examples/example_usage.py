import pandas as pd
from ragret.datasets import example_dataset
from ragret import ContextRecall, ContextPrecision,AnswerRelevancy

context_recall = ContextRecall(provider="openai")
context_precision = ContextPrecision(provider="openai")
answer_relevancy = AnswerRelevancy(provider="openai")

results = []

for i, record in enumerate(example_dataset):
    print(f"Evaluating record {i + 1}/{len(example_dataset)}...")
    
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