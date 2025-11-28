# src/predict.py
import joblib

def predict_age(features):
    model = joblib.load("age_model.pkl")   # Load model only when function is called
    return model.predict([features])[0]


if __name__ == "_main_":
    age = int(input("Enter age: "))
    print(predict_age(age))