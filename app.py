# fraud_app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Optional SHAP
try:
    import shap
    shap_available = True
except ModuleNotFoundError:
    shap_available = False

# =============================
# LOAD DATA AND MODEL
# =============================
@st.cache_data
def load_data(max_rows=500000):
    df = pd.read_csv("onlinefraud.csv")
    
    # Reduce memory usage
    df["type"] = df["type"].astype("category")
    df["nameOrig"] = df["nameOrig"].astype("category")
    df["nameDest"] = df["nameDest"].astype("category")
    
    # If dataset is huge, sample it
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42).reset_index(drop=True)
    
    ads_df = pd.DataFrame()
    ads_df["AccountID"] = df["nameOrig"]
    ads_df["CampaignID"] = df["nameDest"]
    ads_df["ClickType"] = df["type"]
    ads_df["BidValue"] = df["amount"]
    ads_df["ClickTime"] = df["step"]
    ads_df["AccountBalanceBefore"] = df["oldbalanceOrg"]
    ads_df["AccountBalanceAfter"] = df["newbalanceOrig"]
    ads_df["CampaignBalanceBefore"] = df["oldbalanceDest"]
    ads_df["CampaignBalanceAfter"] = df["newbalanceDest"]
    ads_df["FraudLabel"] = df["isFraud"]
    
    # Feature engineering
    ads_df["BalanceChange"] = ads_df["AccountBalanceBefore"] - ads_df["AccountBalanceAfter"]
    ads_df["CampaignBalanceChange"] = ads_df["CampaignBalanceAfter"] - ads_df["CampaignBalanceBefore"]
    ads_df["ClicksPerAccount"] = ads_df.groupby("AccountID")["AccountID"].transform("count")
    ads_df["HighBidFlag"] = (ads_df["BidValue"] > ads_df["BidValue"].quantile(0.95)).astype(int)
    
    # Encode ClickType
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    ads_df["ClickType"] = le.fit_transform(ads_df["ClickType"])
    
    # Drop IDs for modeling
    model_df = ads_df.drop(columns=["AccountID", "CampaignID"])
    
    return ads_df, model_df

ads_df, model_df = load_data()

st.title("🚀 Adversarial Click-Bot Fraud Detection")
st.write(f"Dataset Loaded: {ads_df.shape[0]} rows")

# Load trained model
model = joblib.load("fraud_bot_detector.pkl")

# Save feature order from training
feature_order = [
    "ClickType",
    "BidValue",
    "ClickTime",
    "AccountBalanceBefore",
    "AccountBalanceAfter",
    "CampaignBalanceBefore",
    "CampaignBalanceAfter",
    "BalanceChange",
    "CampaignBalanceChange",
    "ClicksPerAccount",
    "HighBidFlag"
]

# =============================
# RISK BUCKET FUNCTION
# =============================
def risk_bucket(score):
    if score < 0.3:
        return "Low Risk"
    elif score < 0.75:
        return "Review"
    else:
        return "High Risk"

# =============================
# USER INPUT (MAIN PAGE)
# =============================
st.subheader("Enter Click Features:")

click_time = st.number_input("Click Time (step)", value=1)
bid_value = st.number_input("Bid Value", value=1000.0)
click_type = st.selectbox("Click Type", ["PAYMENT", "TRANSFER", "CASH_OUT"])
account_balance_before = st.number_input("Account Balance Before", value=10000.0)
account_balance_after = st.number_input("Account Balance After", value=9500.0)
campaign_balance_before = st.number_input("Campaign Balance Before", value=5000.0)
campaign_balance_after = st.number_input("Campaign Balance After", value=4800.0)
clicks_per_account = st.number_input("Clicks Per Account", value=3)
high_bid_flag = st.checkbox("High Bid Flag?")

# Encode click_type
click_type_encoded = 0 if click_type == "CASH_OUT" else (1 if click_type == "PAYMENT" else 2)

# Prepare input dataframe
input_df = pd.DataFrame([{
    "ClickType": click_type_encoded,
    "BidValue": bid_value,
    "ClickTime": click_time,
    "AccountBalanceBefore": account_balance_before,
    "AccountBalanceAfter": account_balance_after,
    "CampaignBalanceBefore": campaign_balance_before,
    "CampaignBalanceAfter": campaign_balance_after,
    "BalanceChange": account_balance_before - account_balance_after,
    "CampaignBalanceChange": campaign_balance_after - campaign_balance_before,
    "ClicksPerAccount": clicks_per_account,
    "HighBidFlag": int(high_bid_flag)
}])

# Ensure correct feature order
input_df = input_df[feature_order]

# =============================
# PREDICT RISK
# =============================
risk_score = model.predict_proba(input_df)[:,1][0]
risk_label = risk_bucket(risk_score)

st.subheader("⚡ Risk Assessment")
st.write(f"**Risk Score:** {risk_score:.3f}")
st.write(f"**Risk Tier:** {risk_label}")

# =============================
# FULL DATASET ANALYSIS
# =============================
st.subheader("📊 Full Dataset Risk Analysis")
if st.button("Run Full Dataset Risk Analysis"):
    st.write("Running predictions on dataset...")
    model_df_full = model_df.drop("FraudLabel", axis=1)
    model_df_full = model_df_full[feature_order]  # ensure correct order
    risk_scores = model.predict_proba(model_df_full)[:,1]
    risk_labels = [risk_bucket(s) for s in risk_scores]
    
    risk_df = pd.DataFrame({
        "Score": risk_scores,
        "RiskTier": risk_labels,
        "ActualLabel": model_df["FraudLabel"]
    })
    
    st.write("### Risk Tier Distribution")
    st.bar_chart(risk_df["RiskTier"].value_counts())
    
    st.write("### Sample Risk Data")
    st.dataframe(risk_df.head(20))
    
    st.write("### Risk Analysis by Actual Fraud")
    risk_analysis = risk_df.groupby("RiskTier")["ActualLabel"].value_counts()
    st.write(risk_analysis)

# =============================
# SHAP FEATURE IMPORTANCE (SAFE)
# =============================
if shap_available:
    st.subheader("📈 Feature Importance (SHAP)")
    try:
        explainer = shap.TreeExplainer(model)
        sample = model_df.sample(500, random_state=42)  # small safe sample
        sample = sample[feature_order]  # ensure correct order
        
        shap_values = explainer.shap_values(sample.values)  # convert to numpy
        
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, sample, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP plotting failed: {e}")
else:
    st.warning("SHAP module not found. Install with `pip install shap` to see feature importance.")