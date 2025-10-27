import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features: pd.DataFrame):
        """
        Predicts target variable using the trained model and preprocessor.
        """
        try:
            # Check if model and preprocessor exist
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            if not os.path.exists(self.preprocessor_path):
                raise FileNotFoundError(f"Preprocessor file not found at {self.preprocessor_path}")

            logging.info("Loading model and preprocessor objects")
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            logging.info("Transforming input data using preprocessor")
            data_scaled = preprocessor.transform(features)

            preds = model.predict(data_scaled)
            logging.info("Prediction completed successfully")

            return preds

        except Exception as e:
            raise CustomException(e)


class CustomData:
    """
    A helper class to structure input features for prediction.
    """
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: float,
        writing_score: float,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = float(reading_score)
        self.writing_score = float(writing_score)

    def get_data_as_data_frame(self) -> pd.DataFrame:
        """
        Converts the custom input into a pandas DataFrame for prediction.
        """
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info(f"Custom input data prepared: {df.to_dict(orient='records')}")
            return df

        except Exception as e:
            raise CustomException(e)
