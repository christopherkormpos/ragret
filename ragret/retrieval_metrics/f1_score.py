# F1 score represents a harmonic mean of precision and recall, balancing both.
# It ranges from 0 to 1, where higher scores indicate better overall performance of the system.
#
# Compute f1-score using the formula:
# Context f1-score = 2 x (precision x recall)/(precision + recall)
import logging

# Logging configuration for debugging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


class F1Score:
    # Main f1 score calculate function.
    # Needs a precision and recall to be provided so it can calculate the f1-score using the fomula:
    # f1-score = 2 x (precision x recall)/(precision + recall)
    @staticmethod
    def score(precision, recall) -> dict:
        try:
            if not (0 <= precision <= 1 and 0 <= recall <= 1):
                raise ValueError(
                    "Precision and recall must be between 0 and 1.")
            if precision+recall != 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                return {
                    "score": 0
                }
            return {
                "score": f1_score
            }
        except Exception as error:
            logging.error(f"Error on F1 Score: {error}")
            raise
