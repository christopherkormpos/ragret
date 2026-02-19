import inspect
import re
import logging
from tqdm import tqdm

# Logging configuration for debugging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

REQUIRED_KEYS = {"user_query", "retrieved_documents", "llm_answer"}

class Evaluator:
    def __init__(self, dataset: list[dict]) -> None:
        if not isinstance(dataset, list) or len(dataset) == 0:
            raise ValueError(f"Dataset must be a non-empty list of dictionaries with keys: |user_query|,|retrieved_documents|,|llm_answer|")
        self.dataset = dataset

    # Function that converts CamelCase class name to snake_case for the results dictionary
    def _get_metric_name(self, metric) -> str:
        name = type(metric).__name__
        # regex based convertion
        return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()

    # Function that inspects each metric's score() function and only passes the arguments that the coresponding tool needs
    def _call_metric(self, metric, record: dict) -> float:
        # Find the parameters the metric needs (no all metrics want the same inputs)
        try:
            params = inspect.signature(metric.score).parameters.keys()
            kwargs = {}

            if 'user_query' in params:
                kwargs['user_query'] = record['user_query']
            if 'retrieved_documents' in params:
                kwargs['retrieved_documents'] = record['retrieved_documents']
            if 'llm_answer' in params:
                kwargs['llm_answer'] = record['llm_answer']

        except Exception as error:
            logging.error(f"Error on Evaluator: |_call_metric|: {error}")
            raise

        # we only need the score result number from each succesfull run
        return metric.score(**kwargs)["score"]

    # Loops through every record in the dataset, runs all metrics on each, returns a list of result dicts
    def calculate(self, *metrics) -> list[dict]:
        results = []
        try:
            # For each record in the dataset provided
            for record in tqdm(self.dataset, desc="Evaluating"):
                record_result = {}
                # For each metric provided in the calculate function
                for metric in metrics:
                    # First get the name of the metric correct
                    name = self._get_metric_name(metric)
                    # return the score result of the metric
                    record_result[name] = self._call_metric(metric, record)
                # Finally append all of the metrics scores to the results table
                results.append(record_result)

        except Exception as error:
            logging.error(f"Error on Evaluator: |calculate|: {error}")
            raise

        return results
