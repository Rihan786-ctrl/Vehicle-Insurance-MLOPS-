import sys
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.entity.estimator import MyModel

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function trains a RandomForestClassifier with specified parameters
        
        Output      :   Returns metric artifact object and trained model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Training RandomForestClassifier with specified parameters")

            # Splitting the train and test data into features and target variables
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            logging.info("train-test split done.")

            # Initialize RandomForestClassifier with specified parameters
            model = RandomForestClassifier(
                n_estimators = self.model_trainer_config._n_estimators,
                min_samples_split = self.model_trainer_config._min_samples_split,
                min_samples_leaf = self.model_trainer_config._min_samples_leaf,
                max_depth = self.model_trainer_config._max_depth,
                criterion = self.model_trainer_config._criterion,
                random_state = self.model_trainer_config._random_state
            )

            # Fit the model
            logging.info("Model training going on...")
            model.fit(x_train, y_train)
            logging.info("Model training done.")

            # Get probability predictions for optimal threshold calculation
            logging.info("Calculating optimal threshold using F1 score...")
            y_pred_proba = model.predict_proba(x_test)[:, 1]
            precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_pred_proba)
            f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
            
            # Use optimal threshold for final predictions
            y_pred = (y_pred_proba >= optimal_threshold).astype(int)
            
            logging.info(f"Optimal threshold found: {optimal_threshold:.4f}")
            logging.info(f"F1 scores calculated with optimal threshold")
            
            # Predictions and evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            logging.info(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            
            # Store optimal threshold to be used in evaluation and prediction
            self.optimal_threshold = optimal_threshold

            # Creating metric artifact
            metric_artifact = ClassificationMetricArtifact(accuracy_score=accuracy, f1_score=f1, precision_score=precision, recall_score=recall)
            return model, metric_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model training steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            print("------------------------------------------------------------------------------------------------")
            print("Starting Model Trainer Component")
            # Load transformed train and test data
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("train-test data loaded")
            
            # Train model and get metrics
            trained_model, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            logging.info("Model object and artifact loaded.")
            
            # Load preprocessing object
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing obj loaded.")

            # Check if the model's F1 score meets the expected threshold (F1 is better for imbalanced data)
            x_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]
            y_train_proba = trained_model.predict_proba(x_train)[:, 1]
            y_train_pred = (y_train_proba >= self.optimal_threshold).astype(int)
            train_f1 = f1_score(y_train, y_train_pred)
            
            if train_f1 < self.model_trainer_config.expected_f1_score:
                logging.info(f"Model F1 score {train_f1:.4f} below minimum threshold {self.model_trainer_config.expected_f1_score}")
                raise Exception(f"No model found with F1 score above {self.model_trainer_config.expected_f1_score}")
            
            logging.info(f"Model accepted with training F1 score: {train_f1:.4f}")

            # Save the final model object that includes both preprocessing and the trained model
            logging.info("Saving new model as performace is better than previous one.")
            my_model = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=trained_model)
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            logging.info("Saved final model object that includes both preprocessing and the trained model")

            # Create and return the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e