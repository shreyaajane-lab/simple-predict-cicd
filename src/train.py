# tests/test_age_model.py
import os
import joblib
from src.predict import predict_age

def test_model_file_exists():
    assert os.path.exists("age_model.pkl"), "Model file does not exist!"

def test_predict_method():
    model = joblib.load("age_model.pkl")
    assert hasattr(model, "predict"), "Model has no .predict method"

def test_adult_prediction():
    assert predict_age(20) == "Adult"

def test_minor_prediction():
    assert predict_age(10) == "Minor"