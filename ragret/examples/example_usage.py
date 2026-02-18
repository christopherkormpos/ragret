import tqdm
import pandas as pd
from ragret import ContextRecall, ContextPrecision, F1Score
from ragret.datasets import example_dataset

context_recall = ContextRecall(provider="openai")
context_precision = ContextPrecision(provider="openai")
f1_score = F1Score()

results = []

for record in tqdm.tqdm(example_dataset, desc="Evaluating records"):
    recall_result = context_recall.score(
        retrieved_documents=record["retrieved_context"],
        llm_answer=record["llm_answer"]
    )["score"]
    precision_result = context_precision.score(
        user_query=record["user_query"],
        retrieved_documents=record["retrieved_context"]
    )["score"]
    f1_score_result = f1_score.score(precision_result,recall_result)["score"]
    
    results.append({
        "context_precision": precision_result,
        "context_recall": recall_result,
        "f1_score": f1_score_result
    })

df = pd.DataFrame(results)
df.to_csv("evaluation_results.csv", index=False)
print("Results saved to evaluation_results.csv")