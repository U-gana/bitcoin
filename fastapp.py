# importing packages
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI
import joblib

#loading the model
model = joblib.load("bitcoin_predictor.pkl")
app = FastAPI()

#index message
@app.get('/{name}')
def get_name(name:str):
    return(f'Hi there {name} I am U~gana\'s Bitcoin prophet, GHOST I Am Still a work in progress')

# deplotyment 
class InputData(BaseModel):
    High: float
    Low: float
    Open: float
    Close:float
    Volume: float
    Marketcap: float
    
# prediction 
@app.post("/predict")
def predict(data: InputData):
    input_array = np.array([[data.High, data.Low, data.Open,data.Close, data.Volume, data.Marketcap]])
    prediction = model.predict(input_array)
    return {"Predicted_Close": prediction[0]}
