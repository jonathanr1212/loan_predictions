import streamlit as st
import pandas as pd
import pickle
import numpy as np
from PIL import Image

col1, col2, col3 = st.columns([1,1,1])
header = st.container()
information = st.container()

with col1:
    st.write("")

with col2:
    image = Image.open('images/loan_app.png')
    st.image(image)

with col3:
    st.write("")

with header:
    st.markdown("<h1 style='text-align: center; color: white;'>Loan Prediction App</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: white;'>Will you get approved?</h2>", unsafe_allow_html=True)
    st.write('In this app, you can enter your information to check whether or not you would be approved for a loan. Fill out the information below to find out.')

with information:
    with open('finalized_loan_model.pkl', 'rb') as f:
        model = pickle.load(f)

gender=st.selectbox("Gender: Choose options between Male or Female", options = ['Male', 'Female'])
married = st.selectbox("Married: Choose options between No for unmarried, Yes for married", options = ['No', 'Yes'])
dependents = st.selectbox("Dependents: Choose number of dependents in the household", options = ['0', '1', '2', '3+'])
education = st.selectbox("Education: Choose options between Graduate for college degree holder, Not Graduate for non college degree holder", options = ['Graduate', 'Not Graduate'])
self_employed = st.selectbox("Self Employed: Choose options between No for employed by company or unemployed, Yes for self employed", options = ['No', 'Yes'])
applicant_income = st.number_input("Applicant Income: Enter amount of income for applicant", 0)
coapplicant_income = st.number_input("Coapplicant Income: Enter amount of income for coapplicant (0 if none) ", 0)
loan_amount = st.number_input("Loan Amount: Enter how much you would like to borrow ", 0)
loan_amount_term = st.selectbox("Loan Amount Term: Choose options of Loan Terms based in months", options = [12.0, 36.0, 60.0, 84.0, 120.0, 180.0, 240.0, 300.0, 360.0, 480.0])
credit_history = st.selectbox("Credit History: Choose options between 0 for No Credit History, 1 for Has Credit History", options = [0.0, 1.0])
property_area = st.selectbox("Property Area: Choose options between Urban for city, Semiurban for Suburbs, Rural for Countryside", options = ['Urban', 'Semiurban', "Rural"]) 



col_names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term','Credit_History', 'Property_Area']
def predict():
    row =  np.array([gender, married, dependents, education, self_employed, applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history, property_area])
    X = pd.DataFrame([row], columns = col_names)
    prediction = model.predict(X)

    if prediction == 1:
        st.success("You are approved! :thumbsup:")
    else:
        st.error('You were not approved! :thumbsdown: sorry!')

st.button('Check', on_click = predict)










