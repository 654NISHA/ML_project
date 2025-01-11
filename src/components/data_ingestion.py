import pandas as pd
import logging
from src.components.data_transformation import DataTransformation
from src.exception import CustomException
import sys

def load_data(train_path, test_path):
    """
    Load the data from the given paths.
    """
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logging.info(f"Training data loaded with shape: {train_df.shape}")
        logging.info(f"Test data loaded with shape: {test_df.shape}")
        return train_df, test_df
    except Exception as e:
        raise CustomException(e, sys)

def initiate_data_ingestion(train_path, test_path):
    """
    Load data and initiate the data transformation process.
    """
    try:
        # Load data
        train_data, test_data = load_data(train_path, test_path)

        # Data transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_obj_file_path = data_transformation.initiate_data_transformation(
            train_data, test_data)
        
        return train_arr, test_arr, preprocessor_obj_file_path

    except Exception as e:
        raise CustomException(e, sys)
