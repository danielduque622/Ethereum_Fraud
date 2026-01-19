Ethereum Wallet Fraud Detection App
Overview

This application is a Streamlit-based web interface that uses a trained XGBoost classification model to assess whether an Ethereum wallet address is likely to be fraudulent or non-fraudulent based on on-chain transactional behavior.

The app automatically pulls wallet transaction data, engineers features consistent with the training pipeline, applies scaling transformations, and produces real-time fraud risk predictions with interpretable outputs.

Key Features

Real-time Ethereum wallet analysis

Automated transaction data retrieval via Etherscan API

Feature engineering and scaling pipeline consistent with model training

XGBoost-based fraud classification

Probability-based fraud risk scoring

Clean, dashboard-style UI with metric summaries

Transparent display of raw wallet statistics

Technology Stack

Python 3.10+

Streamlit — Web application framework

XGBoost — Fraud classification model

Pandas / NumPy — Data processing

Scikit-learn — Feature scaling pipeline

Etherscan API — Blockchain transaction data
