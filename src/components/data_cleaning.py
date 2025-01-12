import pandas as pd
import logging
from datetime import datetime
from src.exception import CustomException
import sys

class DataCleaning:
    def __init__(self):
        pass

    def clean_data(self, data):
        """
        Perform data cleaning such as handling missing values and creating new features.
        """
        try:
            logging.info("Starting data cleaning process.")

            # Dropping unnecessary columns
            data.drop(['ID', 'Z_CostContact', 'Z_Revenue'], axis=1, inplace=True)

            # Creating new features
            data['Kids'] = data['Teenhome'] + data['Kidhome']

            # Age from Year_Birth
            data['Age'] = datetime.now().year - data['Year_Birth']

            # Replacing Basic and 2n Cycle with School
            data['New_Education'] = data['Education'].replace(['Basic', '2n Cycle'], 'School')

            # Converting Dt_Customer to datetime and creating new date-related features
            data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y')
            d1 = max(data['Dt_Customer'])
            data['Days_for'] = (d1 - data['Dt_Customer']).dt.days
            data['Year_of_joining'] = data['Dt_Customer'].dt.year
            data['Month_of_joining'] = data['Dt_Customer'].dt.month

            # Calculating total amount spent and accepted campaigns
            data['Total_amount_spent'] = (
                data['MntFishProducts'] + data['MntFruits'] + data['MntGoldProds'] +
                data['MntMeatProducts'] + data['MntWines'] + data['MntSweetProducts']
            )

            data['Accepted_campaign'] = (
                data['AcceptedCmp1'] + data['AcceptedCmp2'] + data['AcceptedCmp3'] +
                data['AcceptedCmp4'] + data['AcceptedCmp5']
            )

            # Total number of purchases
            data['Number_of_purchases'] = (
                data['NumDealsPurchases'] + data['NumWebPurchases'] +
                data['NumCatalogPurchases'] + data['NumStorePurchases']
            )

            logging.info("Data cleaning process completed.")
            return data
        except Exception as e:
            raise CustomException(e, sys)
