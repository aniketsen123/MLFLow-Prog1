from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.logger import logging
from src.utils.main_utils import load_object
import sys
import pandas as pd
from typing import Optional
from dataclasses import dataclass
import pickle
import os


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class LocalProj1Estimator:
    """
    Handles local model loading, saving, and prediction (no AWS).
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.loaded_model = None

    def is_model_present(self) -> bool:
        """Check if the model file exists locally."""
        return os.path.exists(self.model_path)

    def load_model(self):
        """Load model from local path."""
        try:
            with open(self.model_path, "rb") as f:
                self.loaded_model = pickle.load(f)
            return self.loaded_model
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe: pd.DataFrame):
        """Make predictions using the loaded model."""
        if self.loaded_model is None:
            self.loaded_model = self.load_model()
        return self.loaded_model.predict(dataframe)


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[LocalProj1Estimator]:
        """
        Retrieve best (production) model from local storage if exists.
        """
        try:
            model_path = self.model_eval_config.local_model_path  # 👈 define in config_entity
            estimator = LocalProj1Estimator(model_path)
            if estimator.is_model_present():
                return estimator
            return None
        except Exception as e:
            raise MyException(e, sys)

    def _map_gender_column(self, df):
        """Map Gender column to 0 for Female and 1 for Male."""
        if "Gender" in df.columns:
            df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1}).astype(int)
        return df

    def _create_dummy_columns(self, df):
        """Create dummy variables for categorical features."""
        return pd.get_dummies(df, drop_first=True)

    def _rename_columns(self, df):
        """Rename specific columns and ensure integer types for dummy columns."""
        df = df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype("int")
        return df

    def _drop_id_column(self, df):
        """Drop ID column if exists."""
        for col in ["id", "_id"]:
            if col in df.columns:
                df = df.drop(col, axis=1)
        return df

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Evaluate trained model vs production model (local).
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            logging.info("Transforming test data...")
            x = self._map_gender_column(x)
            x = self._drop_id_column(x)
            x = self._create_dummy_columns(x)
            x = self._rename_columns(x)

            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            trained_model_f1 = self.model_trainer_artifact.metric_artifact.f1_score

            logging.info(f"Trained Model F1 Score: {trained_model_f1}")

            # Compare with local production model (if available)
            best_model_f1 = None
            best_model = self.get_best_model()

            if best_model is not None:
                logging.info("Evaluating existing production model...")
                y_hat_best = best_model.predict(x)
                best_model_f1 = f1_score(y, y_hat_best)
                logging.info(f"Production Model F1 Score: {best_model_f1}")

            tmp_best_f1 = 0 if best_model_f1 is None else best_model_f1

            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1,
                best_model_f1_score=best_model_f1,
                is_model_accepted=trained_model_f1 > tmp_best_f1,
                difference=trained_model_f1 - tmp_best_f1
            )

            return result

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Start the local model evaluation process.
        """
        try:
            logging.info("Starting local model evaluation...")

            eval_response = self.evaluate_model()

            model_evaluation_artifact = ModelEvaluationArtifact(
                  is_model_accepted=eval_response.is_model_accepted,
                  changed_accuracy=eval_response.difference,
                  local_model_path=self.model_eval_config.local_model_path,  # path where production/best model will be stored
                  trained_model_path=self.model_trainer_artifact.trained_model_file_path
                  )

            logging.info(f"Model evaluation artifact created: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise MyException(e, sys)
