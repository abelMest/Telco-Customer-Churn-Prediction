import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

st.title("Telco Customer Churn Prediction")
st.markdown("""
**Assessment Requirement:** Model Integration and Input/Output Handling  
This application performs real-time churn inference using trained ML artifacts.
""")

# ============================================================
# LOAD ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("best_model.pkl")
        scaler = joblib.load("scaler.pkl")
        encoders = joblib.load("encoders.pkl")
        features = joblib.load("features.pkl")
        return model, scaler, encoders, features
    except FileNotFoundError:
        return None, None, None, None

model, scaler, encoders, features = load_artifacts()

if model is None:
    st.error("Model artifacts not found. Run ml_pipeline.py first.")
    st.stop()

# ============================================================
# SIDEBAR INPUTS
# ============================================================
st.sidebar.header("Customer Profile")

def user_input_features():
    input_data = {}

    # -------------------------
    # NUMERICAL
    # -------------------------
    input_data["tenure"] = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    input_data["MonthlyCharges"] = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 50.0)
    input_data["TotalCharges"] = st.sidebar.number_input("Total Charges ($)", 0.0, 9000.0, 500.0)

    # -------------------------
    # CATEGORICAL (LabelEncoded)
    # -------------------------
    categorical_cols = [
        "Contract",
        "InternetService",
        "PaymentMethod",
        "OnlineSecurity",
        "TechSupport",
    ]

    for col in categorical_cols:
        if col in encoders:
            classes = encoders[col].classes_
            selected = st.sidebar.selectbox(col, classes)
            input_data[col] = encoders[col].transform([selected])[0]

    # -------------------------
    # BINARY
    # -------------------------
    input_data["PaperlessBilling"] = 1 if st.sidebar.checkbox("Paperless Billing", True) else 0
    input_data["SeniorCitizen"] = 1 if st.sidebar.checkbox("Senior Citizen", False) else 0
    input_data["Partner"] = 1 if st.sidebar.checkbox("Has Partner", False) else 0
    input_data["Dependents"] = 1 if st.sidebar.checkbox("Has Dependents", False) else 0

    return input_data


user_inputs = user_input_features()

# ============================================================
# PREDICTION
# ============================================================
if st.button("Predict Churn"):
    try:
        # --------------------------------------------------------
        # BUILD INPUT DATAFRAME WITH EXACT TRAINING FEATURES
        # --------------------------------------------------------
        input_df = pd.DataFrame(columns=features)
        input_df.loc[0] = 0

        # Fill user-provided values
        for col, val in user_inputs.items():
            if col in input_df.columns:
                input_df.at[0, col] = val

        # --------------------------------------------------------
        # ENGINEERED FEATURES (DEFAULTS)
        # --------------------------------------------------------
        if "Is_Outlier" in input_df.columns:
            input_df.at[0, "Is_Outlier"] = 0   # assume not an outlier

        if "Cluster_Segment" in input_df.columns:
            input_df.at[0, "Cluster_Segment"] = 0  # default cluster

        # --------------------------------------------------------
        # SAFETY CHECK — FEATURE ALIGNMENT
        # --------------------------------------------------------
        if hasattr(scaler, "feature_names_in_"):
            missing = set(scaler.feature_names_in_) - set(input_df.columns)
            if missing:
                st.error(f"Missing features: {missing}")
                st.stop()

        # --------------------------------------------------------
        # SCALE
        # --------------------------------------------------------
        input_scaled = scaler.transform(input_df)

        # --------------------------------------------------------
        # PREDICT CLASS
        # --------------------------------------------------------
        prediction = model.predict(input_scaled)

        # --------------------------------------------------------
        # ROBUST PROBABILITY HANDLING
        # --------------------------------------------------------
        churn_prob = None

        # Case 1 — models with predict_proba
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_scaled)
            churn_prob = prob[0][1]

        # Case 2 — models with decision_function
        elif hasattr(model, "decision_function"):
            score = model.decision_function(input_scaled)
            churn_prob = 1 / (1 + np.exp(-score[0]))  # sigmoid

        # --------------------------------------------------------
        # DISPLAY RESULTS
        # --------------------------------------------------------
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.error("CHURN DETECTED")
            else:
                st.success("RETAINED")

        with col2:
            st.subheader("Probability")
            if churn_prob is not None:
                st.metric("Churn Probability", f"{churn_prob*100:.1f}%")
                st.progress(float(churn_prob))
            else:
                st.warning("Probability output not available for this model.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("Developed for CSIS505 Cloud Computing Final Assessment — Abel Mekonnen, abemekonnen@osiriuniversity.org")
