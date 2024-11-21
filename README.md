# Fine-Tuning-Multiple-Models
To continue your project with the goal of improving model accuracy (80-95%) and evaluating the performance of various models (Naive Bayes, Logistic Regression, Deep Neural Networks, and Random Forests), you can follow these steps:

    Update the models: Tune hyperparameters, optimize the models, and apply them to the dataset.
    Evaluate model performance: For each model, calculate the accuracy, loss, confusion matrix, and AUC (Area Under the Curve).

Here’s a Python code snippet that walks through the steps of training, tuning, and evaluating these models using the scikit-learn library for Naive Bayes, Logistic Regression, Random Forest, and a simple Deep Neural Network (DNN) using TensorFlow/Keras.
1. Install Required Libraries

If you don't have these libraries installed, install them with pip:

pip install numpy pandas scikit-learn tensorflow matplotlib seaborn

2. Example Code for Model Training and Evaluation

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 1. Load your dataset (replace with your own dataset)
# For example, using the UCI Iris dataset:
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target

# Only use a binary classification task (e.g., setosa vs non-setosa)
X = X[y != 2]  # Remove class 2 (versicolor)
y = y[y != 2]  # Remove class 2 (versicolor)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Scale the features (important for models like Logistic Regression, DNN, etc.)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Model Definitions and Hyperparameter Tuning
# Naive Bayes Model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# Logistic Regression Model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Deep Neural Network Model (DNN) using MLPClassifier from sklearn
dnn_model_sklearn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
dnn_model_sklearn.fit(X_train, y_train)
y_pred_dnn_sklearn = dnn_model_sklearn.predict(X_test)

# Deep Neural Network Model (DNN) using Keras/TensorFlow
def create_dnn():
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification, so use sigmoid

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

dnn_model_keras = create_dnn()
dnn_model_keras.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

# 4. Model Evaluation
def evaluate_model(model_name, model, X_test, y_test, y_pred=None, plot_roc=False):
    print(f"\nModel: {model_name}")
    
    # Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred) if y_pred is not None else model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"Accuracy: {accuracy:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title(f"Confusion Matrix ({model_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    # AUC-ROC Curve
    if model_name != 'DNN (Sklearn)':  # Sklearn doesn't return probabilities by default
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    else:
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, 'predict_proba') else 0.5
    print(f"AUC: {auc:.4f}")
    
    if plot_roc:
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.title(f"ROC Curve ({model_name})")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.show()

# Evaluate all models
evaluate_model('Naive Bayes', nb_model, X_test, y_test, y_pred_nb, plot_roc=True)
evaluate_model('Logistic Regression', lr_model, X_test, y_test, y_pred_lr, plot_roc=True)
evaluate_model('Random Forest', rf_model, X_test, y_test, y_pred_rf, plot_roc=True)
evaluate_model('DNN (Sklearn)', dnn_model_sklearn, X_test, y_test, y_pred_dnn_sklearn, plot_roc=True)

# Evaluate the Keras-based DNN (custom model)
y_pred_dnn_keras = (dnn_model_keras.predict(X_test) > 0.5).astype("int32")
evaluate_model('DNN (Keras)', dnn_model_keras, X_test, y_test, y_pred_dnn_keras, plot_roc=True)

Explanation of the Code

    Dataset Loading:
        The code uses the Iris dataset from scikit-learn, but you can replace it with your own dataset by loading it with pandas or any other method.
        The dataset is split into training and test sets using train_test_split.

    Model Definition:
        Naive Bayes: Using GaussianNB for the Naive Bayes classifier.
        Logistic Regression: A basic logistic regression model with max_iter=1000 for convergence.
        Random Forest: A Random Forest Classifier with 100 trees.
        Deep Neural Network: Two variants:
            Sklearn MLPClassifier for DNN using Scikit-learn.
            Keras-based DNN using TensorFlow for a custom model with dense layers.

    Model Evaluation:
        Accuracy: The basic classification accuracy.
        Confusion Matrix: Visualizes the number of correct and incorrect predictions for each class.
        AUC: The Area Under the ROC Curve, which is a performance metric for classification problems.
        ROC Curve: Visualizes the true positive rate vs. the false positive rate for different thresholds.

3. Adjust Hyperparameters for Each Model

To achieve 80-95% accuracy, you will need to perform hyperparameter tuning (e.g., using GridSearchCV for Logistic Regression, Random Forest, or MLP) to find the optimal parameters that best fit your data. This can significantly improve the model performance.
4. Next Steps

    Hyperparameter Tuning: Use GridSearchCV or RandomizedSearchCV for hyperparameter optimization.
    Cross-Validation: Apply cross-validation to ensure that your model generalizes well to unseen data.
    Data Preprocessing: Depending on your dataset, you may need to perform additional preprocessing like feature engineering, handling class imbalance, etc.

This code sets you up for fine-tuning and evaluating multiple models, with visualizations for AUC and confusion matrix to better understand model performance
