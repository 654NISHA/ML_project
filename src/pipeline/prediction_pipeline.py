import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object 
from sklearn.preprocessing import LabelEncoder

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            pca_path = 'artifacts/pca.pkl'

            # Load the trained model, preprocessor, and PCA transformer
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            pca = load_object(file_path=pca_path)

            # Encode categorical variables
            le = LabelEncoder()
            features['Education'] = le.fit_transform(features['Education'])
            features['Marital_Status'] = le.fit_transform(features['Marital_Status'])

             # Apply preprocessing
            transformed_features = preprocessor.transform(features)

            # Apply PCA to match model's expected input
            transformed_features = pca.transform(transformed_features)
            
            preds = model.predict(transformed_features)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
    

class CustomData:
    def __init__(self,
                 Income: int,
                 Age: int,
                 Total_amount_spent: int,
                 Kids: int,
                 Complain: int,
                 Number_of_purchases: int,
                 Accepted_campaigns: int,
                 Education: str,
                 Marital_Status: str):
        
        self.Income = Income
        self.Age = Age
        self.Total_amount_spent = Total_amount_spent
        self.Kids = Kids
        self.Complain = Complain
        self.Number_of_purchases = Number_of_purchases
        self.Accepted_campaigns = Accepted_campaigns
        self.Education = Education
        self.Marital_Status = Marital_Status

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'Income':[self.Income],
                'Age': [self.Age],
                'Total_amount_spent':[self.Total_amount_spent],
                'Kids':[self.Kids],
                'Complain': [self.Complain],
                'Number_of_purchases':[self.Number_of_purchases],
                'Accepted_campaigns': [self.Accepted_campaigns],
                'Education':[self.Education],
                'Marital_Status': [self.Marital_Status]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e,sys)
