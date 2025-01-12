import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_cleaning import DataCleaning

class DataIngestion:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

    def load_data(self):
        """
        Load training and testing data from CSV files.
        """
        try:
            logging.info(f"Loading training data from {self.train_path}")
            train_data = pd.read_csv(self.train_path)

            logging.info(f"Loading testing data from {self.test_path}")
            test_data = pd.read_csv(self.test_path)

            logging.info(f"Training data shape: {train_data.shape}, Testing data shape: {test_data.shape}")
            return train_data, test_data

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        """
        Load data and initiate the data transformation process.
        """
        try:
            # Load data
            train_data, test_data = self.load_data()

            # Data transformation
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_obj_file_path = data_transformation.initiate_data_transformation(
                train_data, test_data
            )

            logging.info("Data ingestion and transformation completed.")
            return train_arr, test_arr, preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    # Correct file paths assuming 'train.csv' and 'test.csv' are in the root directory
    train_path = os.path.join(os.getcwd(), 'train.csv')
    test_path = os.path.join(os.getcwd(), 'test.csv')

    data_ingestion = DataIngestion(train_path, test_path)
    train_data, test_data = data_ingestion.load_data()
