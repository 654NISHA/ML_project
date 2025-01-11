# src/components/data_cleaning.py

import pandas as pd
from datetime import datetime
from src.exception import CustomException
import sys
import logging

def preprocess_datetime(df):
    """
    Convert Dt_Customer to numerical feature (days since customer signed up).
    """
    try:
        current_date = datetime.now()
        if 'Dt_Customer' in df.columns:
            df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y', errors='coerce')  # Handle errors in date conversion
            df['Days_since_signup'] = (current_date - df['Dt_Customer']).dt.days
            df.drop(columns=['Dt_Customer'], inplace=True)  # Optionally drop the original date column
            logging.info("Dt_Customer transformed into Days_since_signup")
        else:
            logging.warning("'Dt_Customer' column is missing.")
        return df
    except Exception as e:
        raise CustomException(e, sys)

def filter_data_by_age(df):
    """
    Filter data by age, removing individuals over 90.
    """
    try:
        if 'Year_Birth' in df.columns:
            df['Age'] = datetime.now().year - df['Year_Birth']  # Make sure 'Age' is added
            df = df[df['Age'] <= 90].reset_index(drop=True)
            logging.info(f"Data filtered by age. Shape after filtering: {df.shape}")
        else:
            logging.warning("'Year_Birth' column is missing.")
        return df
    except Exception as e:
        raise CustomException(e, sys)

def remove_outliers_iqr(df, column):
    """
    Remove outliers using IQR method for a specific column.
    """
    try:
        if column in df.columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR

            # Filtering out the outliers
            df = df[(df[column] >= lower_limit) & (df[column] <= upper_limit)].reset_index(drop=True)
            logging.info(f"Outliers removed from {column}. Shape after removal: {df.shape}")
        else:
            logging.warning(f"'{column}' column is missing.")
        return df
    except Exception as e:
        raise CustomException(e, sys)
