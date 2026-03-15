import streamlit as st
import pandas as pd
import joblib

model = joblib.load("extra_trees.pkl")
encoders = {col : joblib.load(f"{col}_encoder.pkl") for col in ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']}

st.title("Credit Risk Pred")
st.write("Enter applicant information to predict credit risk")

age=st.number_input("Age", min_value=18, max_value=80, value=30)
sex = st.selectbox("Sex", ["male","female"])
job = st.number_input("Job (0-3)", min_value = 0, max_value =3 , value=1)
housing = st.selectbox("Housing", ['own','rent','free'])
savings_acc = st.selectbox("Savings Accounts", ['little', 'moderate', 'rich', 'quite rich'])
checking_acc = st.selectbox("Checking Accounts", ['little', 'moderate', 'rich', 'quite rich'])
credit_amount = st.number_input("Credit Amount", min_value = 0, value=100)
duration = st.number_input("Duration (Months)", min_value = 1, value=12)
purpose = st.selectbox("Purpose", ['radio/TV', 'furniture/equipment', 'car', 'business',
       'domestic appliances', 'repairs', 'vacation/others', 'education'])


input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [job],
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([savings_acc])[0]],
    "Checking account": [encoders["Checking account"].transform([checking_acc])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration],
    "Purpose": [encoders["Purpose"].transform([purpose])[0]],
    "Job": [job],
})


if st.button("Predict Risk"):
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.success("The risk is **GOOD**")
    else:
        st.error("The risk is **BAD**")