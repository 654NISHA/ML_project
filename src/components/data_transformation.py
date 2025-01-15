import os
import sys

import pandas as pd
import numpy as np
from datetime import datetime

from src.logger import logging
from src.exception import CustomException

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from src.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")
    logging.info(f"Save preprocessor obj")

    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def perform_feature_engineering(self, data):
        """
        Perform feature engineering, handle missing values, and drop unnecessary columns.
        """
        try:
            logging.info("Starting feature engineering.")

            # Create new features
            current_year = datetime.today().year
            data['Age'] = current_year - data['Year_Birth']
            data['Kids'] = data['Teenhome'] + data['Kidhome']
            data['New_Education'] = data['Education'].replace(['Basic', '2n Cycle'], 'School')

            # Handle date-related features
            data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y', errors='coerce')
            d1 = data['Dt_Customer'].max()
            data['Days_since_joining'] = (d1 - data['Dt_Customer']).dt.days
            data = data.drop(columns=['Dt_Customer'], errors='ignore')

            # Total amounts and purchases
            data['Total_amount_spent'] = data[['MntFishProducts', 'MntFruits', 'MntGoldProds',
                                               'MntMeatProducts', 'MntWines', 'MntSweetProducts']].sum(axis=1)
            data['Accepted_campaigns'] = data[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
                                               'AcceptedCmp4', 'AcceptedCmp5']].sum(axis=1)
            data['Number_of_purchases'] = data[['NumDealsPurchases', 'NumWebPurchases',
                                                'NumCatalogPurchases', 'NumStorePurchases']].sum(axis=1)

            # Drop unnecessary columns
            columns_to_drop = ['Year_Birth', 'Teenhome', 'Kidhome', 'ID', 'Z_CostContact', 'Z_Revenue',
                               'Education', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
                               'NumStorePurchases', 'NumWebVisitsMonth', 'MntFishProducts', 'MntFruits',
                               'MntGoldProds', 'MntMeatProducts', 'MntWines', 'MntSweetProducts',
                               'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
            data = data.drop(columns=columns_to_drop, errors='ignore')

            # Remove outliers
            Q1, Q3 = data['Income'].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            data = data[(data['Income'] >= Q1 - 1.5 * IQR) & (data['Income'] <= Q3 + 1.5 * IQR)]
            data = data[data['Age'] <= 90]

            logging.info("Feature engineering completed.")
            return data

        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        """
        Returns a preprocessing pipeline for numerical and categorical features.
        """
        try:
            logging.info("Creating preprocessor object.")

            # Define columns
            numerical_columns = ['Income', 'Age', 'Total_amount_spent', 'Days_since_joining',
                                'Number_of_purchases','Recency']
            categorical_columns = ['New_Education', 'Marital_Status']

            
            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('label_encoder', 'passthrough')  # LabelEncoder applied manually later
                ]
            )

            # Combined preprocessing
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_columns),
                    ('cat', cat_pipeline, categorical_columns)
                ]
            )

            logging.info("Preprocessor object created.")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        """
        Orchestrates the data transformation process.
        """
        try:
            # Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            # Feature engineering
            train_df = self.perform_feature_engineering(train_df)
            test_df = self.perform_feature_engineering(test_df)
            logging.info("Feature engineering completed in train and test data")

            # Create preprocessor object
            preprocessor = self.get_data_transformer_object()

             # Apply transformations
            train_df_transformed = preprocessor.fit_transform(train_df)
            test_df_transformed = preprocessor.transform(test_df)

            # Extract columns
            transformed_columns = ['Income', 'Age', 'Total_amount_spent', 'Days_since_joining', 'Number_of_purchases', 'Recency', 
                                   'New_Education', 'Marital_Status']

            # Convert to DataFrame
            train_df = pd.DataFrame(train_df_transformed, columns=transformed_columns) 
            test_df = pd.DataFrame(test_df_transformed, columns=transformed_columns)
            
            # Manually apply LabelEncoder to categorical columns
            for col in ['New_Education', 'Marital_Status']:
                le = LabelEncoder()
                train_df[col] = le.fit_transform(train_df[col])
                test_df[col] = le.transform(test_df[col])

            # Convert to numpy arrays for training
            train_arr = train_df.values
            test_arr = test_df.values

            # Save preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            logging.info("Data transformation process completed.")
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
