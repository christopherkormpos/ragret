<h2 align="center">
  METRICS DOCUMENTATION
</h2>

## Overview
Metrics are categorized into `Retrieval metrics` and `Generation metrics`, which constitute the original evaluation metrics. In addition, there are `Custom metrics` that allow users to define their own measures, making the evaluation process more flexible and tailored to specific needs. Furthermore, metrics are distinguished as `LLM-based` and `non-LLM-based`, depending on whether they leverage large language models. All metrics are presented below along with their corresponding tags for clarity and reference.

## Retrieval Metrics
### Context Precision 
`LLM-Based`<br>
*Does the retrieved context provide relevant information to support the answer?*<br>

Context Precision measures how much of the retrieved context is actually useful for answering the user query.<br>
It ranges from 0 to 1, where higher scores indicate that the retrieved context is highly relevant to the query.<br>

To calculate context precision:
1. Extract factual claims from the retrieved context using an LLM or other method.
2. Compare each claim against the user query to see if it is relevant for answering the query.
3. Compute Context Precision using the formula:
<p align="center">
<img src="../images/doc_images/context_precision.png" alt="metrics-formula" style="height:90px;">
</p>

### Context Recall
`LLM-Based`<br>
*Does the answer cover the important information present in the retrieved context?*

Context Recall measures how many of the relevant documents (or pieces of information) were successfully retrieved.<br>
It focuses on not missing important results.<br>
Higher recall means fewer relevant documents were left out. 
In short, recall is about not missing anything important.
To calculate this: 
1. Extract claims from the context
2. Check whether each context claim is covered by the answer
3. Compute recall using the formula:
<p align="center">
<img src="../images/doc_images/context_recall.png" alt="metrics-formula" style="height:90px;">
</p>

### Cosine Similarity
`Non LLM-Based`<br>
*Does the retrieved context semantically align with the user query?*

Cosine similarity measures the semantic similarity between the user input and each of the retrieved documents.<br>
It returns a list of numbers between 0 and 1 that represent the cosine similarity for all the documents in the retrieved context. <br>
It is computed using the formula:
<p align="center">
<img src="../images/doc_images/cosine_similarity.png" alt="metrics-formula" style="height:90px;">
</p>

### F1Score
`Non LLM-Based`<br>
*Does the answer cover the important information in the retrieved context accurately and completely?*

F1 score represents a harmonic mean of precision and recall, balancing both.<br>
It ranges from 0 to 1, where higher scores indicate better overall performance of the system.<br>
Compute f1-score using the formula:
<p align="center">
<img src="../images/doc_images/f1score.png" alt="metrics-formula" style="height:90px;">
</p>

## Generation Metrics
### Faithfulness
`LLM-Based`<br>
*Are the claims in the answer supported by the retrieved context?*

The Faithfulness metric measures how factually consistent a response is with the retrieved context. <br>
It ranges from 0 to 1, with higher scores indicating better consistency.<br>
A response is considered faithful if all its claims can be supported by the retrieved context.<br>
To calculate this: 
1. Identify all the claims in the response. 
2. Check each claim to see if it can be inferred from the retrieved context.
3. Compute the faithfulness score using the formula:
<p align="center">
<img src="../images/doc_images/faithfulness.png" alt="metrics-formula" style="height:90px;">
</p>

### Answer Relevancy
`LLM-Based`<br>
*If we inferred questions from the answer, would they match the user’s question?*

The Anser Relevancy metric measures how relevant a response is to the user input<br>
It ranges from 0 to 1 with higher scores indicating better alignment with the user input<br>
To calculate this
1. Generate a set of artificial questions (default is 3) based on the response
>These questions are designed to reflect the content of the response
2. Compute the cosine similarity between the embedding of the user input and the embedding of each generated question
3. Take the average of these cosine similarity scores to get the Answer Relevancy
You can calculate the Answer Relevancy using the formula:
<p align="center">
<img src="../images/doc_images/answer_relevancy.png" alt="metrics-formula" style="height:90px;">
</p>

## Custom Metrics
### Product Relevancy
`LLM-Based`<br>
*Do the products retrieved from the database match the user's request?*
Product Relevancy measures how many relevant products were retrieved<br>
>OPTIONAL: Use real human labels to decide product relevancy based on the search the user made
To calculate this: 
1. Extract relevant products from the products retrieved
2. OPTIONAL: With LLM check whether each product is relevant to the user quer
3. Compute metric using the formula:
<p align="center">
<img src="../images/doc_images/product_relevancy.png" alt="metrics-formula" style="height:90px;">
</p>
<br>

*For guidance on using all these metrics, please refer to the [README](../README.md) file.*