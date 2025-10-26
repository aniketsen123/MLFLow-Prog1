import sys
import os
import shutil
from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig


class LocalModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        :param model_evaluation_artifact: Output reference of model evaluation stage
        :param model_pusher_config: Configuration for model pusher
        """
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Push the trained model to local production folder if accepted
        """
        logging.info("Entered initiate_model_pusher method")

        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Copying new model to production folder locally...")

            # ✅ Always use a fixed path: artifact/production_model/best_model.pkl
            production_model_path = self.model_pusher_config.local_model_path

            # ✅ Ensure the folder exists
            os.makedirs(os.path.dirname(production_model_path), exist_ok=True)

            # ✅ Copy the latest trained model (from timestamped folder)
            shutil.copy2(self.model_evaluation_artifact.trained_model_path,
                         production_model_path)

            # ✅ Log and return artifact
            model_pusher_artifact = ModelPusherArtifact(
                local_model_path=production_model_path
            )

            logging.info(f"✅ Model pushed successfully to: {production_model_path}")
            logging.info("Exited initiate_model_pusher method")

            return model_pusher_artifact

        except Exception as e:
            raise MyException(e, sys) from e
