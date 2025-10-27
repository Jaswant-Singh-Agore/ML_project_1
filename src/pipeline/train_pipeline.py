from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging


if __name__ == "__main__":
    logging.info("Starting full training pipeline...")

    # Step 1: Data Ingestion
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()

    # Step 2: Data Transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Step 3: Model Training
    model_trainer = ModelTrainer()
    r2, model_name = model_trainer.initiate_model_trainer(train_arr, test_arr)

    logging.info(f"Training completed successfully. Best Model: {model_name} | R² Score: {r2:.4f}")
    print(f"\n Training Completed Successfully!\nBest Model: {model_name}\nR² Score: {r2:.4f}\n")
