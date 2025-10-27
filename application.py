from flask import Flask, request, render_template
import os
import numpy as np
import pandas as pd
from src.logger import logging
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )

            pred_df = data.get_data_as_data_frame()
            logging.info(f"Input data for prediction: {pred_df.to_dict(orient='records')}")

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            prediction_value = float(results[0])
            logging.info(f"Prediction successful. Result: {prediction_value}")

            return render_template('home.html', results=prediction_value)

        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return render_template('home.html', results="Error occurred during prediction.")

@application.route('/health') 
def health():
    return "OK"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    application.run(host="0.0.0.0", port=port, debug=False)
