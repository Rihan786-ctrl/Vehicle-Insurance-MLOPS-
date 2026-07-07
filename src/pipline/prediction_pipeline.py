import sys
from pathlib import Path
import pandas as pd
from pandas import DataFrame

from src.entity.config_entity import VehiclePredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_object

class VehicleData:
    def __init__(self,
                Gender,
                Age,
                Driving_License,
                Region_Code,
                Previously_Insured,
                Annual_Premium,
                Policy_Sales_Channel,
                Vintage,
                Vehicle_Age_lt_1_Year,
                Vehicle_Age_gt_2_Years,
                Vehicle_Damage_Yes
                ):
        """
        Vehicle Data constructor matching the parameters passed from app.py.
        """
        try:
            self.Gender = Gender
            self.Age = Age
            self.Driving_License = Driving_License
            self.Region_Code = Region_Code
            self.Previously_Insured = Previously_Insured
            self.Annual_Premium = Annual_Premium
            self.Policy_Sales_Channel = Policy_Sales_Channel
            self.Vintage = Vintage
            self.Vehicle_Age_lt_1_Year = Vehicle_Age_lt_1_Year
            self.Vehicle_Age_gt_2_Years = Vehicle_Age_gt_2_Years
            self.Vehicle_Damage_Yes = Vehicle_Damage_Yes

        except Exception as e:
            raise MyException(e, sys) from e

    def get_vehicle_input_data_frame(self) -> DataFrame:
        """
        This function returns a DataFrame with the proper raw schema matching training data.
        """
        try:
            vehicle_input_dict = self.get_vehicle_data_as_dict()
            return DataFrame(vehicle_input_dict)
        except Exception as e:
            raise MyException(e, sys) from e

    def get_vehicle_data_as_dict(self):
        """
        Converts the raw numeric UI form fields back into categorical string representations
        expected by the model pipeline's stateful OneHotEncoder / Preprocessor.
        """
        logging.info("Entered get_vehicle_data_as_dict method of VehicleData class")

        try:
            # 1. Reverse engineer the numerical inputs to raw categorical text labels
            gender_raw = "Male" if int(float(self.Gender)) == 1 else "Female"

            if int(float(self.Vehicle_Age_lt_1_Year)) == 1:
                vehicle_age_raw = "< 1 Year"
            elif int(float(self.Vehicle_Age_gt_2_Years)) == 1:
                vehicle_age_raw = "> 2 Years"
            else:
                vehicle_age_raw = "1-2 Year"

            vehicle_damage_raw = "Yes" if int(float(self.Vehicle_Damage_Yes)) == 1 else "No"

            # 2. Build the exact payload schema structure matching schema.yaml requirements
            input_data = {
                "Gender": [gender_raw],
                "Age": [int(self.Age)],
                "Driving_License": [int(self.Driving_License)],
                "Region_Code": [float(self.Region_Code)],
                "Previously_Insured": [int(self.Previously_Insured)],
                "Vehicle_Age": [vehicle_age_raw],       # Raw original categorical column name
                "Vehicle_Damage": [vehicle_damage_raw],   # Raw original categorical column name
                "Annual_Premium": [float(self.Annual_Premium)],
                "Policy_Sales_Channel": [float(self.Policy_Sales_Channel)],
                "Vintage": [int(self.Vintage)]
            }

            logging.info(f"Created vehicle raw feature data dict: {input_data}")
            return input_data

        except Exception as e:
            raise MyException(e, sys) from e


class VehicleDataClassifier:
    def __init__(self, prediction_pipeline_config: VehiclePredictorConfig = VehiclePredictorConfig()) -> None:
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys)

    def _load_local_model(self):
        """Load the latest local model artifact when S3 is unavailable."""
        artifact_dir = Path("artifact")
        if not artifact_dir.exists():
            raise FileNotFoundError("No local artifact directory found.")

        model_files = sorted(
            artifact_dir.rglob("model.pkl"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        if not model_files:
            raise FileNotFoundError("No local model artifact found in artifact/.")

        return load_object(str(model_files[0]))

    def predict(self, dataframe: pd.DataFrame):
        """
        Predicts the output from the given dataframe.
        """
        try:
            logging.info("Entered predict method of VehicleDataClassifier class")
            try:
                model = Proj1Estimator(
                    bucket_name=self.prediction_pipeline_config.model_bucket_name,
                    model_path=self.prediction_pipeline_config.model_file_path,
                )
                return model.predict(dataframe)
            except Exception as s3_error:
                logging.warning("S3 model loading failed, falling back to local artifact. Error: %s", s3_error)
                local_model = self._load_local_model()
                return local_model.predict(dataframe=dataframe)
        
        except Exception as e:
            raise MyException(e, sys)