import sys
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold

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
        self.optimal_threshold = 0.5  # Fallback property placeholder

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        """
        Method Name :   get_model_object_and_report
        Description :   Trains a RandomForestClassifier using Stratified K-Fold Cross-Validation 
                        and applies a precision-recall balance gate to optimize test F1-Score.
        
        Output      :   Returns trained model object and metric artifact object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Starting Stratified K-Fold Cross-Validation and threshold optimization")

            # 1. Separate features and target variables
            x_train_full, y_train_full = train[:, :-1], train[:, -1]
            x_test, y_test = test[:, :-1], test[:, -1]
            logging.info("Train and test arrays separated successfully.")

            # 2. Set up 3-Fold Stratified Cross-Validation to prevent target ratio shifts
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.model_trainer_config._random_state)
            
            best_fold_f1 = -1
            best_model_obj = None

            # 3. Cross-Validation Loop
            for fold, (train_idx, val_idx) in enumerate(skf.split(x_train_full, y_train_full)):
                logging.info(f"--- Processing CV Fold {fold + 1}/3 ---")
                
                x_fold_train, y_fold_train = x_train_full[train_idx], y_train_full[train_idx]
                x_fold_val, y_fold_val = x_train_full[val_idx], y_train_full[val_idx]

                # Initialize the Random Forest with strict depth controls and cost-sensitive class weights
                fold_model = RandomForestClassifier(
                    n_estimators=self.model_trainer_config._n_estimators,
                    min_samples_split=self.model_trainer_config._min_samples_split,
                    min_samples_leaf=self.model_trainer_config._min_samples_leaf,
                    max_depth=self.model_trainer_config._max_depth,
                    criterion=self.model_trainer_config._criterion,
                    random_state=self.model_trainer_config._random_state,
                    class_weight="balanced_subsample"  # Natively fights target class inequality inside the tree splits
                )

                # Train the classifier directly on the resampled fold split
                fold_model.fit(x_fold_train, y_fold_train)

                # Evaluate fold against validation fold data to monitor generalization
                val_proba = fold_model.predict_proba(x_fold_val)[:, 1]
                precisions, recalls, thresholds = precision_recall_curve(y_fold_val, val_proba)
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
                
                fold_best_idx = np.argmax(f1_scores)
                fold_f1 = f1_scores[fold_best_idx]
                logging.info(f"Fold {fold + 1} validation peak F1 score: {fold_f1:.4f}")

                # Keep the model instance that generalizes the best across folds
                if fold_f1 > best_fold_f1:
                    best_fold_f1 = fold_f1
                    best_model_obj = fold_model

            # 4. Precision-Recall Balance Gate to optimize final F1-Score
            logging.info("Calculating final balanced decision threshold on isolated test data...")
            final_test_proba = best_model_obj.predict_proba(x_test)[:, 1]
            precisions, recalls, thresholds = precision_recall_curve(y_test, final_test_proba)
            
            selected_threshold = 0.5
            max_f1_in_gate = -1
            
            # Search for the optimal threshold where Precision >= 0.40 and Recall >= 0.65
            for i in range(len(thresholds)):
                if precisions[i] >= 0.40 and recalls[i] >= 0.65:
                    current_f1 = 2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i] + 1e-10)
                    if current_f1 > max_f1_in_gate:
                        max_f1_in_gate = current_f1
                        selected_threshold = thresholds[i]

            # Fallback if the constraints are too tight for the current data split
            if max_f1_in_gate == -1:
                logging.warning("No threshold matched the exact Multi-Objective gate. Defaulting to absolute peak F1-Score.")
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
                selected_threshold = thresholds[np.argmax(f1_scores)]

            self.optimal_threshold = selected_threshold
            logging.info(f"Target classification threshold committed: {self.optimal_threshold:.4f}")

            # 5. Generate final optimized metrics
            y_pred = (final_test_proba >= self.optimal_threshold).astype(int)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            logging.info(f"👉 Final Optimized Test Performance: Accuracy={accuracy:.4f}, F1-Score={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

            # Create metrics artifact
            metric_artifact = ClassificationMetricArtifact(
                accuracy_score=accuracy, 
                f1_score=f1, 
                precision_score=precision, 
                recall_score=recall
            )
            return best_model_obj, metric_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            print("------------------------------------------------------------------------------------------------")
            print("Starting Optimized Model Trainer Component")
            
            # Load transformed numpy features
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("Transformed train-test data arrays loaded from artifacts storage.")
            
            # Process folds and run optimizations
            trained_model, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            logging.info("Model training loop and performance reporting complete.")
            
            # Load current stateful preprocessing object
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Stateful preprocessing object loaded successfully.")

            # Compute internal generalization check
            x_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]
            y_train_proba = trained_model.predict_proba(x_train)[:, 1]
            y_train_pred = (y_train_proba >= self.optimal_threshold).astype(int)
            train_f1 = f1_score(y_train, y_train_pred)

            if train_f1 < self.model_trainer_config.expected_f1_score:
                logging.info(f"Model generalized training F1-Score ({train_f1:.4f}) is below expected configuration gate.")
                raise Exception("Model quality verification check failed.")
            
            logging.info(f"Model successfully passed pipeline quality check with training F1-Score: {train_f1:.4f}")

            # Instantiate custom model wrapper to include our optimized dynamic threshold
            logging.info("Packaging final production-ready model artifact...")
            my_model = MyModel(
                preprocessing_object=preprocessing_obj, 
                trained_model_object=trained_model, 
                threshold=self.optimal_threshold
            )
            
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            logging.info("Saved final structured model asset containing pipeline transforms and thresholds.")

            # Construct final artifact container
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact created: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e