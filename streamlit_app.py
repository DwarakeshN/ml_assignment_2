import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix
)

# -------------------------------
# App Title
# -------------------------------
st.set_page_config(page_title="Stellar Classification", layout="centered")

st.title("Stellar Object Classification")
st.write(
    "This app classifies celestial objects into **Star**, **Galaxy**, or **Quasar** "
    "using multiple machine learning models."
)

# -------------------------------
# Load Models and Scaler
# -------------------------------
models = {
    "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
    "Decision Tree": joblib.load("models/decision_tree.pkl"),
    "KNN": joblib.load("models/knn.pkl"),
    "Naive Bayes": joblib.load("models/naive_bayes.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "XGBoost": joblib.load("models/xgboost.pkl"),
}

scaler = joblib.load("models/scaler.pkl")

# -------------------------------
# Upload CSV
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV format only)",
    type=["csv"]
)

if uploaded_file is not None:

    # -------------------------------
    # Read and preprocess data
    # -------------------------------
    df = pd.read_csv(uploaded_file)

    cols_to_drop = [
        "obj_ID",
        "run_ID",
        "rerun_ID",
        "cam_col",
        "field_ID",
        "spec_obj_ID"
    ]

    df.drop(columns=cols_to_drop, inplace=True)

    # Encode target column exactly like in training
    label_encoder = LabelEncoder()
    df["class"] = label_encoder.fit_transform(df["class"])

    X = df.drop("class", axis=1)
    y = df["class"]

    # Scale input features
    X_scaled = scaler.transform(X)

    # -------------------------------
    # Model selection
    # -------------------------------
    model_name = st.selectbox(
        "Select Machine Learning Model",
        list(models.keys())
    )

    model = models[model_name]

    # -------------------------------
    # Prediction
    # -------------------------------
    y_pred = model.predict(X_scaled)

    # -------------------------------
    # Evaluation Metrics
    # -------------------------------
    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", round(accuracy_score(y, y_pred), 4))
    col2.metric("Precision", round(precision_score(y, y_pred, average="macro"), 4))
    col3.metric("Recall", round(recall_score(y, y_pred, average="macro"), 4))

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", round(f1_score(y, y_pred, average="macro"), 4))
    col5.metric("MCC", round(matthews_corrcoef(y, y_pred), 4))

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_scaled)
        auc = roc_auc_score(
            y,
            y_proba,
            multi_class="ovr",
            average="macro"
        )
        col6.metric("AUC", round(auc, 4))

    # -------------------------------
    # Confusion Matrix
    # -------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    st.pyplot(fig)

