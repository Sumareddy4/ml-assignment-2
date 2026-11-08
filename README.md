# Bank Marketing Analytics – Machine Learning Classification Dashboard

## 1. Problem Statement

The objective of this project is to develop a machine learning-based classification system using the Bank Marketing dataset.

The project predicts whether a customer will subscribe to a term deposit based on personal, financial, and campaign-related information.

Five machine learning classification algorithms are implemented and evaluated. Their performance is presented through evaluation metrics and an interactive Streamlit dashboard.

The main objectives are:

- Perform data preprocessing and feature transformation.
- Implement five machine learning classification algorithms.
- Evaluate and compare the models.
- Develop an interactive Streamlit application.
- Deploy the application using Streamlit Community Cloud.

## 2. Dataset Description

- **Dataset name:** Bank Marketing Dataset
- **Dataset source:** Kaggle
- **Dataset link:** https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset
- **Dataset file:** `bank.csv`
- **Target variable:** `deposit`
- **Problem type:** Binary classification

The dataset contains customer information and details about marketing campaigns conducted by a banking institution. The target variable indicates whether a customer subscribed to a term deposit:

- `yes` – Customer subscribed to a term deposit.
- `no` – Customer did not subscribe to a term deposit.

## 3. GitHub Repository Link

**GitHub repository:**

### 4.1 Model Performance Comparison

The following accuracy scores were obtained from the final model evaluation:

| ML Model                     |   Accuracy |        AUC |  Precision |     Recall |   F1 Score |        MCC |
| ---------------------------- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| Support Vector Machine (SVM) | **0.8554** | **0.9177** |     0.8244 | **0.8828** | **0.8526** | **0.7125** |
| K-Nearest Neighbor (KNN)     |     0.8298 |     0.8848 | **0.8291** |     0.8072 |     0.8180 |     0.6585 |
| Logistic Regression          |     0.8253 |     0.9072 |     0.8275 |     0.7977 |     0.8123 |     0.6495 |
| Gaussian Naive Bayes         |     0.7179 |     0.8100 |     0.8048 |     0.5340 |     0.6420 |     0.4489 |
| Decision Tree                |     0.5262 |     0.5000 |     0.0000 |     0.0000 |     0.0000 |     0.0000 |

### 4.2 Observations on Model Performance

| ML Model               | Observation about model performance                                                                                                             |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression    | Achieved 82.53% accuracy. It provided good classification performance and served as a baseline model.                                           |
| Decision Tree          | Achieved 52.62% accuracy, the lowest among the five models in this experiment. Its performance may be affected by the selected tree parameters. |
| KNN                    | Achieved 82.98% accuracy. It performed well by using similarities between customer records.                                                     |
| Gaussian Naive Bayes   | Achieved 71.79% accuracy. It provided moderate performance using a probabilistic classification approach.                                       |
| Support Vector Machine | Achieved 85.54% accuracy, the highest among the five models. The SVM with an RBF kernel performed best by accuracy on this test dataset.        |

### Overall Observation

**Best model by accuracy: Support Vector Machine (SVM)**

SVM achieved the highest test accuracy of **85.54%**, followed by KNN at **82.98%** and Logistic Regression at **82.53%**. Decision Tree achieved the lowest accuracy at **52.62%**.

Based on the available accuracy results, SVM is the best-performing model in this experiment. The overall winner across all six required metrics should be confirmed after the AUC, Precision, Recall, F1 Score, and MCC values have also been compared.
