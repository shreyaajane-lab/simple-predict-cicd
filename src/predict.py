# src/predict.py
import joblib

model = joblib.load("age_model.pkl")

def predict_age(age):
    pred = model.predict([[age]])[0]
    return "Adult" if pred == 1 else "Minor"

if __name__ == "_main_":
    age = int(input("Enter age: "))
    print(predict_age(age))