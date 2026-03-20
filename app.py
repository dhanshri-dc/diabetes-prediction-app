
import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("diabetes_model.pkl", "rb"))

st.title("Diabetes Prediction App")

pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 200, 120)
bp = st.number_input("Blood Pressure", 0, 150, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 32.0)
dpf = st.number_input("DPF", 0.0, 3.0, 0.5)
age = st.number_input("Age", 0, 100, 30)

input_data = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]],
columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])

if st.button("Predict"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.error("Diabetes: YES")
    else:
        st.success("Diabetes: NO")

    st.write("Probability:", prob)
