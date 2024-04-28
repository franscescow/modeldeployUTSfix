#Franscesco William Gazali
#2602138536

import streamlit as st
import joblib
import numpy as np

model = joblib.load('XGboost_churn.pkl')

def main():
    st.title('Model Deployment UTS')
    age = st.number_input('Age :',value = 1)
    gender = st.radio('Gender',["Male", "Female"])
    gender_enc = None
    if gender == 'Male':
        gender_enc = 1
    else:
        gender_enc = 0

    
    region = st.radio('Geography',["France", "Spain", "Germany", "Other"])
    region_enc = None
    if region == 'France':
        region_enc = 0
    elif region == 'Spain':
        region_enc = 1
    elif region == 'Germany':
        region_enc = 2
    else:
        region_enc = 3

    tenure = st.number_input('Tenure :',value = 1)
    cscore = st.number_input('Credit Score :',value = 1)
    balance = st.number_input('Balance :',value = 1)
    numprod = st.number_input('number of products :',value = 1)
    creditcard = st.radio('Has credit card?',["Yes", "No"])
    creditcard_enc = None
    if creditcard == 'Yes':
        creditcard_enc = 1
    else:
        creditcard_enc = 0

    isactivemember = st.radio('Is active member?',["Yes", "No"])
    activemember_enc = None
    if isactivemember == 'Yes':
        activemember_enc = 1
    else:
        activemember_enc = 0

    salary = st.number_input('Salary :',value = 1)
    
    
    if st.button('Make Prediction'):
        feat_list = [cscore, region_enc, gender_enc, age, tenure, balance,numprod, creditcard_enc, activemember_enc, salary]
        features = np.array(feat_list, dtype = object)

        result = make_prediction(features)
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    # Use the loaded model to make predictions
    # Replace this with the actual code for your model
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    
    if prediction == 0:
        output = 'not churn'
    else:
        output = 'churn'
    
    return output

if __name__ == '__main__':
    main()


