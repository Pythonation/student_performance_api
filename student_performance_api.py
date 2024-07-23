# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("student_performance_api")

# Create input/output pydantic models
input_model = create_model("student_performance_api_input", **{'Age': 15.0, 'Gender': 1.0, 'Ethnicity': 3.0, 'ParentalEducation': 2.0, 'StudyTimeWeekly': 8.093074798583984, 'Absences': 29.0, 'Tutoring': 1.0, 'ParentalSupport': 0.0, 'Extracurricular': 0.0, 'Sports': 0.0, 'Music': 0.0, 'Volunteering': 0.0})
output_model = create_model("student_performance_api_output", prediction=4.0)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
