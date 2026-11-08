from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

st.set_page_config(
    page_title="Bank Marketing Classifier",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "model"
if not MODEL_DIR.exists():
    MODEL_DIR = BASE_DIR / "models"

TEST_DATA_PATH = BASE_DIR / "test_data.csv"
if not TEST_DATA_PATH.exists():
    TEST_DATA_PATH = BASE_DIR / "data" / "test_data.csv"


MODEL_FILES = {
    "Logistic Regression": ["logistic_regression.pkl"],
    "Decision Tree Classifier": ["decision_tree.pkl"],
    "K-Nearest Neighbor": ["knn.pkl"],
    "Gaussian Naive Bayes": [
        "gaussian_nb.pkl",
        "gaussian_naive_bayes.pkl",
        "naive_bayes.pkl",
    ],
    "Support Vector Machine": ["svc.pkl", "svm.pkl"],
}

st.markdown(
    """
    <style>
        .stApp {
            background-color: #111827;
        }

        .block-container {
            max-width: 1400px;
            padding-top: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        /* Centered navigation bar */
        div[role="radiogroup"] {
            display: flex;
            justify-content: center;
            background-color: #1F2937;
            border: 1px solid #374151;
            border-radius: 999px;
            padding: 6px;
            width: fit-content;
            margin: 0 auto 1rem auto;
        }

        /* Hide the radio circles */
        div[role="radiogroup"] > label > div:first-child {
            display: none;
        }

        /* Each navigation option */
        div[role="radiogroup"] > label {
            padding: 8px 22px;
            border-radius: 999px;
            cursor: pointer;
            margin: 0;
            color: #9CA3AF;
            transition: all 0.2s ease;
        }

        div[role="radiogroup"] > label p {
            font-size: 15px;
            font-weight: 500;
            margin: 0;
        }

        /* Selected option */
        div[role="radiogroup"] > label:has(input:checked) {
            background-color: #4F46E5;
            color: #FFFFFF;
        }

        div[role="radiogroup"] > label:has(input:checked) p {
            color: #FFFFFF;
        }

        div[role="radiogroup"] > label:hover {
            color: #F9FAFB;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_test_data():
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test data not found: {TEST_DATA_PATH}"
        )

    return pd.read_csv(TEST_DATA_PATH)

@st.cache_resource
def load_models():
    loaded_models = {}

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}"
        )

    for model_name, filenames in MODEL_FILES.items():
        model_path = next(
            (
                MODEL_DIR / filename
                for filename in filenames
                if (MODEL_DIR / filename).exists()
            ),
            None,
        )

        if model_path is None:
            raise FileNotFoundError(
                f"Model not found for {model_name}. "
                f"Expected one of: {filenames}"
            )

        loaded_models[model_name] = joblib.load(model_path)

    return loaded_models

def calculate_metrics(model, X_test, y_test):
    positive_label = "yes"
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        pos_label=positive_label,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label=positive_label,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label=positive_label,
        zero_division=0,
    )

    mcc = matthews_corrcoef(y_test, y_pred)

    # Use prediction probabilities when available.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        classes = list(model.classes_)
        positive_index = classes.index(positive_label)
        y_scores = probabilities[:, positive_index]

    # Otherwise, use the model's decision scores (e.g. SVM).
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
        classes = list(model.classes_)

        # decision_function scores correspond to classes_[1].
        if classes[1] != positive_label:
            y_scores = -y_scores

    else:
        y_scores = None

    auc = (
        roc_auc_score(
            (y_test == positive_label).astype(int),
            y_scores,
        )
        if y_scores is not None
        else np.nan
    )

    return {
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc,
    }

def create_comparison_chart(all_metrics, height=460):
    metric_names = [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC",
    ]

    fig = go.Figure()

    for model_name, metrics in all_metrics.items():
        values = [metrics[metric] for metric in metric_names]

        fig.add_trace(
            go.Scatter(
                x=metric_names,
                y=values,
                mode="lines+markers",
                name=model_name,
                line=dict(width=2),
                marker=dict(size=6),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "%{x}: %{y:.4f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB", size=11),
        xaxis=dict(
            title="Evaluation Metrics",
            showgrid=False,
        ),
        yaxis=dict(
            title="Score",
            range=[-1, 1],
            gridcolor="rgba(255,255,255,0.12)",
            zeroline=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ),
        hovermode="x unified",
    )

    return fig

try:
    test_data = load_test_data()
    trained_models = load_models()

except Exception as error:
    st.error(f"Application loading error: {error}")
    st.stop()


if "deposit" not in test_data.columns:
    st.error("The test_data.csv file must contain the 'deposit' target column.")
    st.stop()

X_test = test_data.drop(columns=["deposit"])
y_test = test_data["deposit"]


# Calculate the metrics for all five models.
all_metrics = {}

for model_name, model in trained_models.items():
    all_metrics[model_name] = calculate_metrics(
        model,
        X_test,
        y_test,
    )

st.markdown(
    "<h1 style='text-align:center;'>Bank Marketing Analytics</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p style='text-align:center; color:#A5B4FC;'>
        Machine Learning Classification Dashboard
    </p>
    """,
    unsafe_allow_html=True,
)

page = st.radio(
    "Navigation",
    ["Home", "Classification"],
    horizontal=True,
    label_visibility="collapsed",
)

st.divider()

if page == "Home":

    st.header("Welcome to the Classification Dashboard")

    st.write(
        """
        This project explores machine learning classification algorithms
        using the Bank Marketing dataset.

        The objective is to predict whether a customer will subscribe
        to a term deposit based on their personal and campaign-related
        information.
        """
    )

    st.subheader("Machine Learning Models")

    models_description = {
        "Logistic Regression": (
            "A statistical classification algorithm that predicts "
            "the probability of a binary outcome."
        ),
        "Decision Tree Classifier": (
            "A tree-based algorithm that makes predictions using "
            "a sequence of feature-based decisions."
        ),
        "K-Nearest Neighbor": (
            "Classifies a customer based on the majority class "
            "among its nearest data points."
        ),
        "Gaussian Naive Bayes": (
            "A probabilistic classifier based on Bayes' theorem "
            "and the assumption of Gaussian feature distributions."
        ),
        "Support Vector Machine": (
            "Finds an optimal decision boundary that separates "
            "different classes."
        ),
    }

    for name, description in models_description.items():
        with st.container(border=True):
            st.markdown(f"**{name}**")
            st.write(description)

    st.subheader("Evaluation Metrics")

    metrics_description = {
        "Accuracy": (
            "Measures the percentage of total predictions that are correct."
        ),
        "AUC Score": (
            "Measures how well the model distinguishes between positive "
            "and negative classes across different thresholds."
        ),
        "Precision": (
            "Measures how many predicted positive cases are actually positive."
        ),
        "Recall": (
            "Measures how many actual positive cases are correctly identified."
        ),
        "F1 Score": (
            "The harmonic mean of precision and recall."
        ),
        "Matthews Correlation Coefficient (MCC)": (
            "Measures binary classification quality using all four "
            "confusion-matrix outcomes."
        ),
    }

    for name, description in metrics_description.items():
        with st.container(border=True):
            st.markdown(f"**{name}**")
            st.write(description)

    st.info(
        "Navigate to the Classification page to select a trained model, "
        "enter customer details and view its performance."
    )

elif page == "Classification":

    metrics_col, input_col = st.columns([4, 6], gap="medium")

    with metrics_col:
        st.markdown("##### Comparison of All Five Models")

        st.caption(
            "MCC ranges from -1 to 1; the other metrics range from 0 to 1."
        )

        comparison_fig = create_comparison_chart(all_metrics)

        st.plotly_chart(
            comparison_fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with input_col:

        st.subheader("Customer Classification")

        prediction_model_name = st.selectbox(
            "Choose Classification Model",
            list(trained_models.keys()),
            key="prediction_model",
        )

        st.markdown("#### Enter Customer Details")

        categorical_features = X_test.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        numerical_features = X_test.select_dtypes(
            include=np.number
        ).columns.tolist()

        input_values = {}

        with st.form("customer_form"):

            st.markdown("##### Numerical Information")

            input_left, input_right = st.columns(2)

            for index, feature in enumerate(numerical_features):
                target_col = (
                    input_left if index % 2 == 0 else input_right
                )

                with target_col:
                    column_data = X_test[feature]

                    default_value = float(column_data.median())
                    min_value = float(column_data.min())
                    max_value = float(column_data.max())

                    # Avoid invalid number_input ranges.
                    if min_value == max_value:
                        min_value -= 1.0
                        max_value += 1.0

                    default_value = min(
                        max(default_value, min_value),
                        max_value,
                    )

                    input_values[feature] = st.number_input(
                        feature.replace("_", " ").title(),
                        min_value=min_value,
                        max_value=max_value,
                        value=default_value,
                        step=1.0,
                        key=f"input_{feature}",
                    )

            st.divider()
            st.markdown("##### Categorical Information")

            categorical_left, categorical_right = st.columns(2)

            for index, feature in enumerate(categorical_features):
                target_col = (
                    categorical_left if index % 2 == 0 else categorical_right
                )

                with target_col:
                    options = sorted(
                        X_test[feature].dropna().unique().tolist()
                    )

                    input_values[feature] = st.selectbox(
                        feature.replace("_", " ").title(),
                        options,
                        key=f"input_{feature}",
                    )

            submitted = st.form_submit_button(
                "Predict Customer Subscription",
                use_container_width=True,
            )

        # ------------------------------------------
        # PREDICTION RESULT
        # ------------------------------------------

        if submitted:

            input_df = pd.DataFrame([input_values])
            input_df = input_df[X_test.columns]

            model = trained_models[prediction_model_name]
            prediction = model.predict(input_df)[0]

            st.divider()
            st.subheader("Prediction Result")

            if prediction == "yes":
                st.success(
                    "The customer is likely to subscribe to the term deposit."
                )
            else:
                st.warning(
                    "The customer is unlikely to subscribe to the term deposit."
                )

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                classes = list(model.classes_)
                positive_index = classes.index("yes")
                probability = probabilities[positive_index]

                st.metric(
                    "Subscription Probability",
                    f"{probability * 100:.2f}%",
                )

                st.progress(float(probability))

            st.caption(
                f"Prediction generated using {prediction_model_name}"
            )