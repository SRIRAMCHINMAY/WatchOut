 WatchOut

AWID Wireless Intrusion Detection using Machine Learning
📘 Project Overview

This project implements anomaly detection and multi-class attack classification on the AWID (Aegean Wi-Fi Intrusion Dataset).
The goal is to automatically detect and classify Wi-Fi network intrusions using machine learning models, including Random Forest and Multi-class Neural Networks (Deep Learning).

🧩 Dataset

AWID (Aegean Wi-Fi Intrusion Dataset) contains real-world Wi-Fi traffic captured under normal and attack conditions.
Each packet is labeled as either normal or one of several attack types such as:

Flooding

Impersonation

Injection

The dataset includes multiple numeric and categorical features derived from the wireless frame headers.

⚙️ Methodology
1. Data Preprocessing

Loaded AWID dataset and selected relevant numeric features.

Handled missing values and normalized continuous features.

Encoded categorical labels into numeric class IDs:

0 → Flooding
1 → Impersonation
2 → Injection
3 → Normal

2. Feature Selection

Initial experiments used the top 76 most relevant features.

Additional feature engineering included:

Time-based features (e.g., frame.time_delta, frame.time_relative)

Signal features (radiotap.dbm_antsignal, radiotap.channel.freq)

3. Model Training

Two models were trained and compared:

🏗️ Random Forest Classifier

Used as a strong baseline for tabular classification.

Tuned number of trees (n_estimators) and depth (max_depth).

Produced high accuracy and fast inference times.

🧠 Multi-class Neural Network

Implemented a simple feed-forward deep learning model.

Architecture:

Input layer: based on selected features

Hidden layers: ReLU activations + Dropout

Output layer: Softmax for multi-class prediction

Optimized using categorical cross-entropy and Adam optimizer.

📊 Evaluation Metrics

Accuracy

Precision / Recall / F1-Score

Confusion Matrix to evaluate class-wise performance.

Attack vs Normal Traffic Ratios for better anomaly detection analysis.
