# importing packages
from fastapi import FastAPI
import joblib

#loading the model
model = joblib.load("bitcoin_predictor.pkl")
app = FastAPI()
