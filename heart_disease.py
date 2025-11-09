import streamlit as st
import pandas as pd
import joblib

def load_artifacts():
    try:
        model = joblib.load("heart_disease_model.pkl")
        preprocessor = joblib.load("preprocessor.pkl")
        return model, preprocessor, None
    except Exception as e:
        return None, None, e

def user_input():
    
    BMI = st.number_input("BMI (Weight(kg) / Height(m)²)", 10.0, 50.0, 24.0)
    Smoking = st.selectbox("Do you smoke?", ["Yes", "No"])
    AlcoholDrinking = st.selectbox("Do you drink alcohol?", ["Yes", "No"])
    Stroke = st.selectbox("Have you had a stroke?", ["Yes", "No"])
    PhysicalHealth = st.slider("Physical Health (days)", 0, 30, 5)
    MentalHealth = st.slider("Mental Health (days)", 0, 30, 5)
    DiffWalking = st.selectbox("Difficulty in walking?", ["Yes", "No"])
    Sex = st.selectbox("Sex", ["Male", "Female"])
    AgeCategory = st.selectbox("Age Category", [
        "18-24","25-29","30-34","35-39","40-44",
        "45-49","50-54","55-59","60-64","65-69",
        "70-74","75-79","80 or older"
    ])
    Race = st.selectbox("Race", ["White","Black","Asian","Other"])
    Diabetic = st.selectbox("Are you diabetic?", ["Yes", "No"])
    PhysicalActivity = st.selectbox("Do you exercise regularly?", ["Yes", "No"])
    GenHealth = st.selectbox("General Health", ["Excellent","Very good","Good","Fair","Poor"])
    SleepTime = st.slider("Sleep Time (hours/day)", 1, 24, 7)
    Asthma = st.selectbox("Asthma", ["Yes", "No"])
    KidneyDisease = st.selectbox("Kidney Disease", ["Yes", "No"])
    SkinCancer = st.selectbox("Skin Cancer", ["Yes", "No"])

    data = {
        "BMI": BMI,
        "Smoking": Smoking,
        "AlcoholDrinking": AlcoholDrinking,
        "Stroke": Stroke,
        "PhysicalHealth": PhysicalHealth,
        "MentalHealth": MentalHealth,
        "DiffWalking": DiffWalking,
        "Sex": Sex,
        "AgeCategory": AgeCategory,
        "Race": Race,
        "Diabetic": Diabetic,
        "PhysicalActivity": PhysicalActivity,
        "GenHealth": GenHealth,
        "SleepTime": SleepTime,
        "Asthma": Asthma,
        "KidneyDisease": KidneyDisease,
        "SkinCancer": SkinCancer
    }
    return pd.DataFrame([data])

def main():
    st.title("Heart Disease Prediction App")

    model, preprocessor, err = load_artifacts()
    if err:
        st.error(f"Error loading model/preprocessor: {err}")
        return

    input_df = user_input()

    if st.button("Predict"):
        processed = preprocessor.transform(input_df)
        pred = model.predict(processed)[0]
        proba = model.predict_proba(processed)[0][1]

        if pred == 1:
            st.error(f"High Risk — Probability: {proba*100:.2f}%")
        else:
            st.success(f"Low Risk — Probability: {proba*100:.2f}%")

    st.text("Built with LightGBM & Streamlit")

if __name__ == "__main__":
    main()
