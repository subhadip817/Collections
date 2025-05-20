import streamlit as st
import pandas as pd
import joblib

# Load trained models
clf7 = joblib.load("clf7.pkl")
clf14 = joblib.load("clf14.pkl")
clf30 = joblib.load("clf30.pkl")
clf_channel = joblib.load("clf_channel.pkl")
clf_time = joblib.load("clf_time.pkl")
clf_risk = joblib.load("clf_risk.pkl")

# UI
st.set_page_config(page_title="Collections Probability", layout="centered")
st.title("🤖 AI-Driven Collections Probability")
st.markdown("Fill out the form below:")

# Input Fields
dpd = st.slider("📌 Days Past Due (DPD)", 0, 90, 5)
loan = st.number_input("💰 Loan Amount", 5000, 500000, 50000)
vintage = st.slider("📆 Customer Vintage (in years)", 0, 24, 1)
turnover = st.number_input("📈 Annual Turnover (in ₹)", 100000, 10000000, 500000)
emi = st.number_input("💸 EMI Amount", 1000, 50000, 3000)
overdue = st.number_input("🧾 Overdue Amount", 0, 50000, 0)
comm_score = st.slider("📞 Communication Score", 0.0, 1.0, 0.5)
util_score = st.slider("💡 Utility Payment Score", 0.0, 1.0, 0.6)
sms_resp = st.slider("📲 SMS Response Rate", 0.0, 1.0, 0.3)
bureau = st.slider("🏦 Credit Bureau Score", 300, 900, 600)
last_pmt = st.slider("⏱️ Days Since Last Payment", 0, 90, 30)

# Prediction Button
if st.button("🔮 Predict Collection Insights"):
    X = pd.DataFrame([[
        dpd, loan, comm_score, util_score, sms_resp, bureau, vintage, turnover, emi, overdue,last_pmt
    ]], columns=[
        "dpd", "loan_amount", "communication_score", "utility_payment_score", "sms_response_rate", "credit_bureau_score", "business_vintage_years", "annual_turnover", "emi_amount", "overdue_amount", "days_since_last_repayment"
    ])

    # Predictions
    st.subheader("📊 Repayment Probability")
    st.metric("In 7 Days", f"{clf7.predict_proba(X)[0][1]*100:.0f}%")
    st.metric("In 14 Days", f"{clf14.predict_proba(X)[0][1]*100:.0f}%")
    st.metric("In 30 Days", f"{clf30.predict_proba(X)[0][1]*100:.0f}%")

    st.subheader("⚠️ Risk & Recommendations")
    st.metric("Risk of Default", "High" if clf_risk.predict(X)[0] else "Low")
