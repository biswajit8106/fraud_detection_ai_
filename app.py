<<<<<<< HEAD
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load XGBoost model
def load_model():
    return joblib.load("xgb_fraud_model.pkl")

model = load_model()

# Streamlit page config
st.set_page_config(page_title="Fraud Detection System", layout="centered")
st.title("Fraud Detection System")
st.markdown("Predict fraudulent transactions using a trained XGBoost model.")

# Label Encoding Map
type_mapping = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4
}

# Prediction Mode
mode = st.radio("Choose Prediction Mode:", ["Real Time Prediction", "Batch via CSV"])

# Real Time Prediction
if mode == "Real Time Prediction":
    st.markdown("Enter Transaction Details")

    step = st.number_input("Step (Hour)", min_value=1, max_value=744, value=1)
    type_ = st.selectbox("Transaction Type", list(type_mapping.keys()))
    amount = st.number_input("Amount", min_value=0.0, value=1000.0)
    oldbalanceOrg = st.number_input("Sender Old Balance", min_value=0.0, value=5000.0)
    newbalanceOrig = st.number_input("Sender New Balance", min_value=0.0, value=4000.0)
    oldbalanceDest = st.number_input("Receiver Old Balance", min_value=0.0, value=1000.0)
    newbalanceDest = st.number_input("Receiver New Balance", min_value=0.0, value=2000.0)

    if st.button(" Predict Fraud"):
        encoded_type = type_mapping[type_]
        input_df = pd.DataFrame({
            'step': [step],
            'type': [encoded_type],
            'amount': [amount],
            'oldbalanceOrg': [oldbalanceOrg],
            'newbalanceOrig': [newbalanceOrig],
            'oldbalanceDest': [oldbalanceDest],
            'newbalanceDest': [newbalanceDest]
        })

        pred = model.predict(input_df)[0]

        if pred == 1:
            st.error(f" Fraud Detected! ")
        else:
            st.success(f"Genuine Transaction.")

# Batch prediction
elif mode == "Batch via CSV":
    st.markdown("Upload CSV File for Batch Prediction")
    st.info("Required columns: `step`, `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Drop unnecessary columns if present
            drop_cols = ['nameOrig', 'nameDest', 'isFraud', 'isFlaggedFraud']
            dropped = [col for col in drop_cols if col in df.columns]
            if dropped:
                df.drop(columns=dropped, inplace=True)
                st.warning(f"Dropped columns: {', '.join(dropped)}")

            # Required columns
            required_cols = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(f"CSV is missing required columns: {', '.join(missing_cols)}")
            else:
                # Encode 'type' if needed
                if df['type'].dtype == 'object':
                    df['type'] = df['type'].map(type_mapping)

                # Predict
                preds = model.predict(df)
            

                df['Prediction'] = ['Fraud' if p == 1 else 'Genuine' for p in preds]
            

                st.success("Prediction Complete")
                st.dataframe(df)

                # Download results
                csv_out = df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇Download Predictions CSV", data=csv_out, file_name="fraud_predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")
=======
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load XGBoost model
def load_model():
    return joblib.load("xgb_fraud_model.pkl")

model = load_model()

# Streamlit page config
st.set_page_config(page_title="Fraud Detection System", layout="centered")
st.title("Fraud Detection System")
st.markdown("Predict fraudulent transactions using a trained XGBoost model.")

# Label Encoding Map
type_mapping = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4
}

# Prediction Mode
mode = st.radio("Choose Prediction Mode:", ["Single Transaction", "Batch via CSV"])

# Real Time Prediction
if mode == "Real Time Prediction":
    st.markdown("Enter Transaction Details")

    step = st.number_input("Step (Hour)", min_value=1, max_value=744, value=1)
    type_ = st.selectbox("Transaction Type", list(type_mapping.keys()))
    amount = st.number_input("Amount", min_value=0.0, value=1000.0)
    oldbalanceOrg = st.number_input("Sender Old Balance", min_value=0.0, value=5000.0)
    newbalanceOrig = st.number_input("Sender New Balance", min_value=0.0, value=4000.0)
    oldbalanceDest = st.number_input("Receiver Old Balance", min_value=0.0, value=1000.0)
    newbalanceDest = st.number_input("Receiver New Balance", min_value=0.0, value=2000.0)

    if st.button(" Predict Fraud"):
        encoded_type = type_mapping[type_]
        input_df = pd.DataFrame({
            'step': [step],
            'type': [encoded_type],
            'amount': [amount],
            'oldbalanceOrg': [oldbalanceOrg],
            'newbalanceOrig': [newbalanceOrig],
            'oldbalanceDest': [oldbalanceDest],
            'newbalanceDest': [newbalanceDest]
        })

        pred = model.predict(input_df)[0]

        if pred == 1:
            st.error(f" Fraud Detected! ")
        else:
            st.success(f"Genuine Transaction.")

# Batch prediction
elif mode == "Batch via CSV":
    st.markdown("Upload CSV File for Batch Prediction")
    st.info("Required columns: `step`, `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Drop unnecessary columns if present
            drop_cols = ['nameOrig', 'nameDest', 'isFraud', 'isFlaggedFraud']
            dropped = [col for col in drop_cols if col in df.columns]
            if dropped:
                df.drop(columns=dropped, inplace=True)
                st.warning(f"Dropped columns: {', '.join(dropped)}")

            # Required columns
            required_cols = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(f"CSV is missing required columns: {', '.join(missing_cols)}")
            else:
                # Encode 'type' if needed
                if df['type'].dtype == 'object':
                    df['type'] = df['type'].map(type_mapping)

                # Predict
                preds = model.predict(df)
            

                df['Prediction'] = ['Fraud' if p == 1 else 'Genuine' for p in preds]
            

                st.success("Prediction Complete")
                st.dataframe(df)

                # Download results
                csv_out = df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇Download Predictions CSV", data=csv_out, file_name="fraud_predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")
>>>>>>> 5facfa6dcc417773c7110e8ce8860838e2c76698
