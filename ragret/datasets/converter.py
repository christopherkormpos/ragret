import logging

# Accepted key names for each ragret field
KEY_ALIASES = {
    "user_query": ["user_query", "user_queries", "question", "questions", "user_input"],
    "retrieved_documents": ["retrieved_documents", "contexts", "retrieved_contexts"],
    "llm_answer": ["llm_answer", "llm_answers", "answer", "answers", "response", "responses"],
    "ground_truth": ["ground_truth", "ground_truths", "reference", "references"],
}

# Function that finds which of the accepted aliases a dataset uses for a given 
# ragret field.
def _find_key(keys, field: str) -> str | None:
    matches = [alias for alias in KEY_ALIASES[field] if alias in keys]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous dataset keys for |{field}|: {matches}. Please keep only one of them")
    return matches[0] if matches else None

# Normalize value to what ragert expects (a single string)
def _normalize_value(field: str, value):
    if field == "ground_truth" and isinstance(value, list):
        return " ".join(value)
    return value

# Function that converts a dataset from ragas-like schemas into the ragret schema:
# Accepts either a columnar dict of lists ({"question": [...], "contexts": [[...], ...], ...})
# or a list of row dicts with any of the key names found in KEY_ALIASES.
# Fields missing from the dataset are left out
def dataset_converter(dataset: dict | list[dict]) -> list[dict]:
    try:
        # Columnar format -> split the lists into one dict per record
        if isinstance(dataset, dict):
            field_keys = {field: _find_key(dataset.keys(), field) for field in KEY_ALIASES}
            field_keys = {field: key for field, key in field_keys.items() if key is not None}
            if not field_keys:
                raise ValueError(f"No recognised keys found in dataset. Accepted keys: {KEY_ALIASES}")

            # Every column must be a list of the same length
            lengths = {}
            for key in field_keys.values():
                if not isinstance(dataset[key], list):
                    raise ValueError(f"Column |{key}| must be a list, got {type(dataset[key]).__name__}")
                lengths[key] = len(dataset[key])
            if len(set(lengths.values())) > 1:
                raise ValueError(f"All dataset columns must have the same length, got: {lengths}")

            return [
                {field: _normalize_value(field, dataset[key][i]) for field, key in field_keys.items()}
                for i in range(next(iter(lengths.values())))
            ]

        # Row format -> rename the keys of each record
        elif isinstance(dataset, list):
            converted = []
            for i, record in enumerate(dataset):
                if not isinstance(record, dict):
                    raise ValueError(f"Record {i} must be a dictionary, got {type(record).__name__}")
                converted_record = {}
                for field in KEY_ALIASES:
                    key = _find_key(record.keys(), field)
                    if key is not None:
                        converted_record[field] = _normalize_value(field, record[key])
                if not converted_record:
                    raise ValueError(f"No recognised keys found in record {i}. Accepted keys: {KEY_ALIASES}")
                converted.append(converted_record)
            return converted

        else:
            raise ValueError(f"Dataset must be a dict of lists or a list of dicts, got {type(dataset).__name__}")

    except Exception as error:
        logging.error(f"Error on Dataset Converter: |dataset_converter|: {error}")
        raise
