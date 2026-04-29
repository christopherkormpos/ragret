import inspect
import re
import logging
from concurrent.futures import ThreadPoolExecutor
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

    # Loops through every record in the dataset, runs all metrics in parallel on each, returns a list of result dicts
    def calculate(self, *metrics) -> list[dict]:
        results = []
        try:
            for record in tqdm(self.dataset, desc="Evaluating"):
                with ThreadPoolExecutor() as executor:
                    futures = {
                        self._get_metric_name(metric): executor.submit(self._call_metric, metric, record)
                        for metric in metrics
                    }
                record_result = {name: future.result() for name, future in futures.items()}
                results.append(record_result)

        except Exception as error:
            logging.error(f"Error on Evaluator: |calculate|: {error}")
            raise

        return results
