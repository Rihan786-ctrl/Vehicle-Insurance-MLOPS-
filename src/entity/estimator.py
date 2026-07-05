import sys
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from src.exception import MyException
from src.logger import logging

class TargetValueMapping:
    def __init__(self):
        self.yes: int = 0
        self.no: int = 1
    def _asdict(self):
        return self.__dict__
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))

class MyModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object, threshold: float = 0.5):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model 
        :param threshold: Classification threshold for the positive/minority class
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object
        self.threshold = threshold

    def predict(self, dataframe: pd.DataFrame) -> DataFrame:
        """
        Function accepts preprocessed inputs, applies scaling/encoding transformations,
        and performs prediction based on the saved optimal threshold.
        """
        try:
            logging.info("Starting prediction process.")

            # Step 1: Apply scaling and encoding transformations using the pre-trained preprocessing object
            transformed_feature = self.preprocessing_object.transform(dataframe)

            # Safe check: Fallback to 0.5 if evaluating an older model that lacks the 'threshold' attribute
            current_threshold = getattr(self, 'threshold', 0.5)

            # Step 2: Perform prediction using custom threshold via predict_proba if available
            logging.info(f"Using the trained model to get predictions with threshold: {current_threshold}")
            if hasattr(self.trained_model_object, "predict_proba"):
                probabilities = self.trained_model_object.predict_proba(transformed_feature)[:, 1]
                predictions = (probabilities >= current_threshold).astype(int)
            else:
                predictions = self.trained_model_object.predict(transformed_feature)

            return predictions

        except Exception as e:
            logging.error("Error occurred in predict method", exc_info=True)
            raise MyException(e, sys) from e

    def __repr__(self):
        current_threshold = getattr(self, 'threshold', 0.5)
        return f"{type(self.trained_model_object).__name__}(threshold={current_threshold})"

    def __str__(self):
        current_threshold = getattr(self, 'threshold', 0.5)
        return f"{type(self.trained_model_object).__name__}(threshold={current_threshold})"