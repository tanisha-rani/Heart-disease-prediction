# 💓 Heart Disease Prediction App  
A machine learning application built using **LightGBM**, **scikit-learn**, and **Streamlit** to predict the risk of heart disease based on lifestyle and health-related factors.

---

## 🚀 Project Overview
Heart disease is a major global health concern. Early prediction helps in taking preventive measures and saving lives.

This project uses:
- ✅ A large heart disease dataset  
- ✅ Data preprocessing using scikit-learn  
- ✅ Model training (Random Forest + LightGBM)  
- ✅ Best model saved as `.pkl`  
- ✅ A Streamlit web app for real-time prediction  

The final deployed app accepts user inputs (like BMI, age, smoking, sleep time, etc.) and predicts the probability of heart disease.

---

## 🧠 Machine Learning Workflow

### **1️⃣ Data Preprocessing**
- Handling missing values  
- Encoding categorical variables  
- Scaling numeric features  
- OneHotEncoding & StandardScaler wrapped inside a ColumnTransformer  

### **2️⃣ Model Training**
Trained multiple models:
- ✅ Random Forest  
- ✅ LightGBM (best performing)  

Achieved **~91% accuracy** on test data.

Saved models:
