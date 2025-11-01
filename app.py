# ==========================================
# 🌍 Earthquake Alert Prediction - ML App
# ==========================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ------------------------------------------
# Streamlit App Title and Description
# ------------------------------------------
st.set_page_config(page_title="Earthquake Alert Prediction", layout="wide")

st.title("🌋 Earthquake Alert Prediction - Machine Learning App")

st.markdown("""
### 🧠 Predict Earthquake Alert Levels using Multiple ML Algorithms
Upload your dataset, choose your target and feature columns, and this app will:
- Train **KNN**, **Naive Bayes**, **Logistic Regression**, and **SVM**
- Compare their accuracies visually using bar charts
- Display which model performs the best 🏆
""")

# ------------------------------------------
# File Upload Section
# ------------------------------------------
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Preview of Uploaded Data")
    st.write(df.head())

    # Column selection
    target_col = st.selectbox("🎯 Select Target Column (what you want to predict):", df.columns)
    feature_cols = st.multiselect("🧩 Select Feature Columns:", [col for col in df.columns if col != target_col])

    if len(feature_cols) > 0 and st.button("🚀 Train Models"):
        X = df[feature_cols]
        y = df[target_col]

        # Encode categorical target if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Encode categorical features
        X = pd.get_dummies(X, drop_first=True)

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Standardize numerical data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Define ML models
        models = {
            "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Support Vector Machine (SVM)": SVC()
        }

        # Train and evaluate each model
        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name] = acc

        # ------------------------------------------
        # Display Results
        # ------------------------------------------
        st.subheader("📈 Model Accuracy Comparison")
        results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
        st.dataframe(results_df.style.format({"Accuracy": "{:.2%}"}))

        # Main Bar Chart for Accuracy Comparison
        st.subheader("📊 Accuracy Comparison Chart")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(results_df["Model"], results_df["Accuracy"], color=['skyblue', 'orange', 'green', 'red'])
        ax.set_xlabel("Algorithms")
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Performance Comparison")
        plt.xticks(rotation=15)
        st.pyplot(fig)

        # Individual algorithm performance charts
        st.subheader("📉 Individual Algorithm Performance")
        for name, acc in results.items():
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar([name], [acc], color='teal')
            ax.set_ylim(0, 1)
            ax.set_ylabel("Accuracy")
            ax.set_title(f"{name} Accuracy")
            st.pyplot(fig)

        # Best Model
        best_model = max(results, key=results.get)
        best_acc = results[best_model]
        st.success(f"🏆 Best Model: **{best_model}** with Accuracy: **{best_acc*100:.2f}%**")

else:
    st.info("👆 Please upload a CSV file to start. Example: `earthquake_alert_balanced_dataset.csv`")
