import streamlit as st
import pandas as pd
import joblib
from web3 import Web3
import requests
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import shap
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv(".env")

# Load your trained model
model = joblib.load("models/fraud_model.pkl")

#load the data scaler for features
scaler = joblib.load("models/scaler.pkl")

def form_query_string(address, page, offset, api_key):
    base_url = "https://api.etherscan.io/v2/api"

    params = {
        "module": "account",
        "action": "txlist",
        "chainid":1,
        "address": address,
        "startblock": 0,
        "endblock": 99999999,
        "page": page,
        "offset": offset,
        "sort": "asc",
        "apikey": api_key
    }

    return base_url, params

# Aggregate transactional data for a given address
def get_address_stats(address):
    try:
        # api_key = os.environ.get("ETHERSCAN_API_KEY")
        api_key = os.environ["ETHERSCAN_API_KEY"]

        if not api_key:
            st.error("Missing ETHERSCAN_API_KEY environment variable")
            return None

        url, params = form_query_string(address, 1, 10000, api_key)

        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()

        payload = response.json()

 #       st.write("RAW ETHERSCAN RESPONSE")
 #       st.json(payload)

        # Handle new + old schemas
        if "result" in payload:
            txs = payload["result"]
        elif "data" in payload and "result" in payload["data"]:
            txs = payload["data"]["result"]
        else:
            st.error("Unexpected Etherscan response format")
            return None

        # Etherscan error guard
        if isinstance(txs, str):
            st.error(f"Etherscan API error: {txs}")
            return None

        if not txs or len(txs) == 0:
            st.warning("No transactions found for this address")
            return None

        df = pd.DataFrame(txs)

        # Normalize timestamp column
        if "timeStamp" not in df.columns and "timestamp" in df.columns:
            df["timeStamp"] = df["timestamp"]

        # Enforce numeric conversion safely
        df["timeStamp"] = pd.to_numeric(df["timeStamp"], errors="coerce").fillna(0).astype(int)

        # Normalize ETH value (string or int compatible)
        df["eth value"] = df["value"].apply(
            lambda x: float(Web3.from_wei(int(float(x)), "ether")) if pd.notnull(x) else 0
        )

        stats = calculate_stats(df, address)

        return stats

    except requests.exceptions.RequestException as e:
        st.error(f"Etherscan network error: {e}")

    except Exception as e:
        st.error(f"Processing error: {e}")

    return None


# Separate function to calculate stats
def calculate_stats(sample_df, address):

    # Enforce time ordering
    sample_df = sample_df.sort_values("timeStamp")

    sent_df = sample_df[sample_df["from"].str.lower() == address.lower()]
    received_df = sample_df[sample_df["to"].str.lower() == address.lower()]

    stats = {
        "Avg min between sent tnx":
            sent_df["timeStamp"].diff().mean() / 60 if len(sent_df) > 1 else 0,

        "Avg min between received tnx":
            received_df["timeStamp"].diff().mean() / 60 if len(received_df) > 1 else 0,

        "Time Diff between first and last (Mins)":
            (sample_df["timeStamp"].max() - sample_df["timeStamp"].min()) / 60,

        "Unique Received From Addresses":
            received_df["from"].nunique(),

        "Min value received":
            received_df["eth value"].min() if not received_df.empty else 0,

        "Max value received":
            received_df["eth value"].max() if not received_df.empty else 0,

        "Avg val received":
            received_df["eth value"].mean() if not received_df.empty else 0,

        "Min val sent":
            sent_df["eth value"].min() if not sent_df.empty else 0,

        "Avg val sent":
            sent_df["eth value"].mean() if not sent_df.empty else 0,

        "Total transactions (including tnx to create contract)":
            len(sample_df),

        "Total ether received":
            received_df["eth value"].sum(),

        "Total ether balance":
            received_df["eth value"].sum() - sent_df["eth value"].sum()
    }

    return stats
    
# Load the scaler used during training
joblib.dump(scaler, "scaler.pkl")

    # Convert to array and scale (use transform NOT fit_transform)
from joblib import load

def prepare_model_features(stats):

    log_stats = {
        key: np.log(value) if value > 0 else 0
        for key, value in stats.items()
    }

    scaler = load("scaler.pkl")

    stats_array = np.array(list(log_stats.values())).reshape(1, -1)

    normalized_values = scaler.transform(stats_array)

    return normalized_values

# Streamlit App
st.title("Fraudulent Ethereum Wallet Detection App")

# Input field for Ethereum address
address = st.text_input("Enter Ethereum Wallet Address:")

if st.button("Predict"):
    if address:
        st.info("Fetching transactional data...")
        raw_stats = get_address_stats(address)

        model_features = np.asarray(prepare_model_features(raw_stats)).reshape(1, -1)

        if raw_stats:
            st.write("Transactional Data:")
            col1, col2, col3 = st.columns(3)

            col1.metric("Total Transactions", raw_stats["Total transactions (including tnx to create contract)"])
            col2.metric("Total Ether Received", round(raw_stats["Total ether received"], 4))
            col3.metric("Wallet Balance", round(raw_stats["Total ether balance"], 4))

            with st.expander("View Full Transaction Statistics"):
             st.json(raw_stats)

            # Define a dictionary to map class labels to human-readable strings
            class_mapping = {0: 'non-fraudulent', 1: 'fraudulent'}
            
            prediction = model.predict(model_features)[0]

            probs = model.predict_proba(model_features)

            prediction_prob = probs[0].tolist()

            # Map the predicted class label to the corresponding string for prediction
            prediction_label = class_mapping.get(prediction, 'Unknown')

             # Ensure that the keys are strings and the values are native Python types (float)
            # Map class labels to human-readable labels in the probability dictionary
            prediction_prob_dict = {class_mapping.get(label, str(label)): float(prob) for label, prob in zip(model.classes_, prediction_prob)}

            # Display the prediction
            st.subheader("Prediction Results")
            st.write(f"Predicted Class: **{prediction_label}**")
            #st.write("Prediction Probabilities:")
            #st.json(prediction_prob_dict)

            fraud_prob = prediction_prob_dict["fraudulent"]

            st.subheader("Fraud Risk Level")

            st.progress(fraud_prob)

            st.write(f"Fraud Risk Score: {fraud_prob:.2%}")


    if prediction_label:
        st.divider()
        st.subheader("🔍 Prediction Analysis (Explainability)")

        # Note: Using TreeExplainer is best for XGBoost performance
        actual_model = model.best_estimator_ 
        explainer = shap.TreeExplainer(actual_model)
        
        # 2. Calculate SHAP values for the specific input
        feature_names = list(raw_stats.keys())
        df_features = pd.DataFrame(model_features, columns=feature_names)

        # Explain the DataFrame instead of the numpy array
        shap_results = explainer(df_features)

        # 3. Create the Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # We use .values[0] to get the explanation for the single wallet entered
        # Mapping back to raw_stats.keys() ensures the labels are readable
        shap.plots.waterfall(shap_results[0], max_display=10, show=False)
        
        st.pyplot(plt.gcf())
        plt.clf() 

        st.info("""
        **How to read this chart:**
        * **f(x)**: The final model output (log-odds).
        * **E[f(X)]**: The base value (average risk across all wallets).
        * **Red arrows**: These features *increased* the suspicion of fraud.
        * **Blue arrows**: These features *decreased* the suspicion of fraud.
        """)


    else:
        st.error("Please try a valid address")
