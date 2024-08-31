from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
 
app=Flask(__name__)

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['Get', 'POST'])
def predict_data():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            test_preparation_course=request.form.get('test_preparation_course'),
            lunch=request.form.get('lunch'),
            writing_score=int(request.form.get('writing_score')),
            reading_score=int(request.form.get('reading_score'))
        )
        
        pred_df=data.get_data_as_dataframe()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        
        results = predict_pipeline.predict(pred_df)
        if results > 100:
            results = ["The score is greater than 100. Please check the input values"]
        print(results)
        return render_template('home.html', results=results[0])
    



if __name__ == '__main__':
    app.run(port=5001, debug=True)