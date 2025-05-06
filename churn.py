import pandas as pd
import numpy as np

# import model final
from xgboost import XGBClassifier

# import load model
import pickle
import joblib

import streamlit as st

# =============================================================

# judul
st.write("""
         <div style="text-align: center;">
         <h2>Churn Customer Prediction</h2>
         </div>
         """, unsafe_allow_html=True)

# sidebar menu for input
st.sidebar.header("Please Input Your Customer Feature")

# untuk input numerik
def satrio():
    cred_score = st.sidebar.slider(label = "Credit Score", 
                    min_value = 350, 
                    max_value = 850, 
                    value = 580)

    balance = st.sidebar.slider(label = "Balance", 
                    min_value = 0, 
                    max_value = 251000, 
                    value = 125000)

    salary = st.sidebar.slider(label = "EstimatedSalary", 
                    min_value = 11, 
                    max_value = 200000, 
                    value = 100000)

    age = st.sidebar.number_input(label = "Age",
                            min_value = 18,
                            max_value = 92,
                            value = 18)

    tenure = st.sidebar.number_input(label = "Tenure",
                            min_value = 0,
                            max_value = 10,
                            value = 0)

    num_product = st.sidebar.number_input(label = "Number Of Product",
                            min_value = 1,
                            max_value = 5,
                            value = 1)

    has_cc = st.sidebar.selectbox(label = "Has Credit Card",
                            options=[0,1])

    is_acm = st.sidebar.selectbox(label = "Is Active Member",
                            options=[0,1])

    geo = st.sidebar.selectbox(label = "Geography",
                            options=["France", "Spain", "Germany"])

    gender = st.sidebar.selectbox(label = "Gender",
                            options=["Female", "Male"])
    df = pd.DataFrame()
    df['CreditScore'] = [cred_score]
    df['Geography'] = [geo]
    df['Gender'] = [gender]
    df['Age'] = [age]
    df['Tenure'] = [tenure]
    df['Balance'] = [balance]
    df['NumOfProducts'] = [num_product]
    df['HasCrCard'] = [has_cc]
    df['IsActiveMember'] = [is_acm]
    df['EstimatedSalary'] = [salary]
    return df

df_feature = satrio()

# memanggil model
model = joblib.load("model_xgboost_joblib")

# predict
pred = model.predict(df_feature)

st.write("Tujuan dari dashboard ini adalah menentukan apakah seorang customer akan melakukan churn dari bank ini.")

# untuk membuat layout menjadi 2 bagian
shafira, ardelia = st.columns(2)

with shafira:
    st.subheader("Customer Characteristics")
    st.write(df_feature.transpose())
    
with ardelia:
    st.subheader("Prediction Result")
    if pred == [1]:
        st.write("Your Customer is likely to CHURN")
    else:
        st.write("Your Customer is predicted to STAY")