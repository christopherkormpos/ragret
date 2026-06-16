import logging


def extract_claims_from_answer(llm, llm_answer: str) -> list[str]:
    prompt = f"""
Extract explicit factual claims from an answer.

Rules:
- Only extract claims explicitly stated in the text. Do NOT infer or assume.
- Each claim must contain exactly one subject and one fact. Never combine two facts into one claim.
- Each claim must be self-contained and independently verifiable.
- Preserve the original language.
- Output a single string with claims separated by dollar sign ($).
- If no factual claims exist, output an empty string.

Example:
Answer: "We open at 9am on weekdays and 10am on weekends. We are closed on holidays."
Output: "We open at 9am on weekdays$We open at 10am on weekends$We are closed on holidays"

Answer:
{llm_answer}
"""
    try:
        result = llm.generate(prompt)
        return [c.strip() for c in result.split("$") if c.strip()]
    except Exception as error:
        logging.error(f"Error in extract_claims_from_answer: {error}")
        raise


def extract_claims_from_ground_truth(llm, ground_truth: str) -> list[str]:
    prompt = f"""
Extract explicit factual claims from a ground-truth (reference) answer.

Rules:
- Only extract claims explicitly stated in the text. Do NOT infer or assume.
- Each claim must contain exactly one subject and one fact. Never combine two facts into one claim.
- Each claim must be self-contained and independently verifiable.
- Preserve the original language.
- Output a single string with claims separated by dollar sign ($).
- If no factual claims exist, output an empty string.

Example:
Ground truth: "Orders over $50 ship free within the US. International shipping is charged at standard rates."
Output: "Orders over $50 ship free within the US$International shipping is charged at standard rates"

Ground truth:
{ground_truth}
"""
    try:
        result = llm.generate(prompt)
        return [c.strip() for c in result.split("$") if c.strip()]
    except Exception as error:
        logging.error(f"Error in extract_claims_from_ground_truth: {error}")
        raise


def extract_claims_from_context(llm, retrieved_documents: list[str]) -> list[str]:
    prompt = f"""
Extract explicit factual claims from retrieved context.

Rules:
- Only extract claims explicitly stated in the text. Do NOT infer or assume.
- Each claim must contain exactly one subject and one fact. Never combine two facts into one claim.
- Each claim must be self-contained and independently verifiable.
- Preserve the original language.
- Output a single string with claims separated by dollar sign ($).
- If no factual claims exist, output an empty string.

Example:
Context: "Orders over $50 get free shipping. International shipping is always charged."
Output: "Orders over $50 get free shipping$International shipping is always charged"

Context:
{retrieved_documents}
"""
    try:
        result = llm.generate(prompt)
        return [c.strip() for c in result.split("$") if c.strip()]
    except Exception as error:
        logging.error(f"Error in extract_claims_from_context: {error}")
        raise


def check_claim_against_context(llm, claim: str, retrieved_documents: list[str]) -> bool:
    prompt = f"""
You are checking whether a claim is supported by the given context.
Context:
{retrieved_documents}

Claim:
{claim}

Decide if the claim is:
- SUPPORTED: directly stated or clearly inferred
- NOT_SUPPORTED: not mentioned or insufficient information

Answer with exactly one label:
SUPPORTED
NOT_SUPPORTED
"""
    try:
        result = llm.generate(prompt)
        return result.strip().upper() == "SUPPORTED"
    except Exception as error:
        logging.error(f"Error in check_claim_against_context: {error}")
        raise


def check_claim_against_query(llm, claim: str, user_query: str) -> bool:
    prompt = f"""
You are checking whether a retrieved context claim is relevant to a user question.

Context Claim:
{claim}

User Question:
{user_query}

Decide if the claim is:
- SUPPORTED: clearly states or implies the claim
- NOT_SUPPORTED: does not mention or imply the claim

Answer with exactly one label:
SUPPORTED
NOT_SUPPORTED
"""
    try:
        result = llm.generate(prompt)
        return result.strip().upper() == "SUPPORTED"
    except Exception as error:
        logging.error(f"Error in check_claim_against_query: {error}")
        raise
