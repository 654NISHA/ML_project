from flask import Flask, request,  render_template
import numpy as np
import pandas as pd

from src.pipeline.prediction_pipeline import CustomData, PredictionPipeline

application = Flask(__name__)
app = application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
        Income = int(request.form.get('Income')),
        Age = int(request.form.get('Age')),
        Total_amount_spent = float(request.form.get('Total_amount_spent')),
        Kids = int(request.form.get('Kids')),
        Complain = int(request.form.get('Complain')),
        Number_of_purchases = int(request.form.get('Number_of_purchases')),
        Accepted_campaigns = int(request.form.get('Accepted_campaigns')),
        Education = request.form.get('Education'),
        Marital_Status = request.form.get('Marital_Status')
        )

        pred_df=data.get_data_as_data_frame()
        print("Input DataFrame:\n", pred_df)

        predict_pipeline=PredictionPipeline()
        result=predict_pipeline.predict(pred_df)

        label_map = {0: "Low Value Customer", 1: "High Value Customer"}
        final_result = label_map.get(result[0], "Unknown")  

        print(f"Prediction Result: {final_result}")  
        
        return render_template('home.html', results=final_result)
    
if __name__=='__main__':
    app.run(host='0.0.0.0')