# src/components/data_transformation.py

import os
import sys
import logging
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.utils import save_object
from src.components.data_cleaning import preprocess_datetime, remove_outliers_iqr, filter_data_by_age

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class LabelEncodingTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for Label Encoding.
    """
    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        for column in X.columns:
            le = LabelEncoder()
            le.fit(X[column])
            self.encoders[column] = le
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for column in X.columns:
            le = self.encoders[column]
            X_transformed[column] = le.transform(X[column])
        return X_transformed

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Returns the data transformer object (preprocessor) that can be used on the train and test data.
        """
        try:
            # Define columns
            numerical_columns = ['Income', 'Year_Birth', 'Kidhome', 'Teenhome', 'MntWines', 'MntFruits', 'MntMeatProducts',
                                 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
                                 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
                                 'Z_CostContact', 'Z_Revenue']
            categorical_columns = ['Education', 'Marital_Status', 'Dt_Customer', 'Recency', 'AcceptedCmp3', 
                                   'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Response']

            # Pipelines for numerical and categorical data
            num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
            cat_pipeline = Pipeline(steps=[ 
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("label_encoder", LabelEncodingTransformer())
            ])

            # Preprocessor (ColumnTransformer)
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e)

    def initiate_data_transformation(self, train_df, test_df):
        """
        Orchestrates the data transformation process.
        """
        try:
            # Strip any extra spaces in the column names
            train_df.columns = train_df.columns.str.strip()
            test_df.columns = test_df.columns.str.strip()

            # Check if 'Dt_Customer' column is missing
            if 'Dt_Customer' not in train_df.columns:
                logging.error("Error: 'Dt_Customer' column is missing in training data.")
                raise ValueError("'Dt_Customer' column is missing from the training dataset")

            if 'Dt_Customer' not in test_df.columns:
                logging.error("Error: 'Dt_Customer' column is missing in testing data.")
                raise ValueError("'Dt_Customer' column is missing from the testing dataset")

            # Apply datetime transformation
            train_df = preprocess_datetime(train_df)
            test_df = preprocess_datetime(test_df)

            # Only remove outliers from training data
            train_df = remove_outliers_iqr(train_df, 'Income')

            # Apply age filtering to both train and test data
            train_df = filter_data_by_age(train_df)
            test_df = filter_data_by_age(test_df)

            # Get the preprocessing pipeline
            preprocessing_obj = self.get_data_transformer_object()

            # Apply transformations to training and testing data
            train_arr = preprocessing_obj.fit_transform(train_df)
            test_arr = preprocessing_obj.transform(test_df)

            # Save the transformed data and preprocessing object
            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessing_obj)

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e)
