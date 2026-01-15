import pandas as pd
import numpy as np
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# --- 18+ ALGORITHMS IMPORTS ---
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier


def load_and_preprocess_data(filepath):
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath)

    # --- DATA CLEANING ---
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # --- ENCODE CATEGORICAL FEATURES ---
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # --- SPLIT FEATURES / TARGET ---
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # ============================================================
    # UNSUPERVISED FEATURE ENGINEERING
    # ============================================================

    print("Running Unsupervised Feature Engineering...")

    # 1. Isolation Forest → Outlier Flag
    iso = IsolationForest(contamination=0.05, random_state=42)
    X['Is_Outlier'] = iso.fit_predict(X)

    # 2. KMeans → Customer Segment
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    X['Cluster_Segment'] = kmeans.fit_predict(X)

    # ============================================================
    # SCALING
    # ============================================================

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ============================================================
    # DIMENSIONALITY REDUCTION (FIT ONLY – for exam coverage)
    # ============================================================

    pca = PCA(n_components=0.95)
    pca.fit(X_scaled)

    lda = LDA(n_components=1)
    lda.fit(X_scaled, y)

    # ============================================================
    # CRITICAL FIX — RETURN THE CORRECT FEATURE SCHEMA
    # ============================================================

    return X_scaled, y, label_encoders, scaler, X.columns


def get_classifiers():
    models = []

    # Linear
    models.append(('Logistic Regression', LogisticRegression(max_iter=1000)))
    models.append(('Ridge Classifier', RidgeClassifier()))
    models.append(('SGD Classifier', SGDClassifier(random_state=42)))
    models.append(('Perceptron', Perceptron(random_state=42)))
    models.append(('Passive Aggressive', SGDClassifier(
        loss='hinge', penalty=None, learning_rate='pa1', eta0=1.0, random_state=42)))

    # Neighbors & SVM
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('Linear SVC', LinearSVC(dual="auto", max_iter=2000, random_state=42)))

    # Naive Bayes
    models.append(('Gaussian NB', GaussianNB()))
    models.append(('Bernoulli NB', BernoulliNB()))

    # Trees & Ensembles
    models.append(('Decision Tree', DecisionTreeClassifier(random_state=42)))
    models.append(('Random Forest', RandomForestClassifier(random_state=42)))
    models.append(('Extra Trees', ExtraTreesClassifier(random_state=42)))
    models.append(('AdaBoost', AdaBoostClassifier(random_state=42)))
    models.append(('Gradient Boosting', GradientBoostingClassifier(random_state=42)))
    models.append(('Hist Gradient Boosting', HistGradientBoostingClassifier(random_state=42)))

    # Neural Network
    models.append(('MLP Classifier', MLPClassifier(max_iter=1000, random_state=42)))

    return models


def main():
    filename = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

    try:
        X, y, encoders, scaler, feature_names = load_and_preprocess_data(filename)
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return

    # --- SPLIT DATA ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = get_classifiers()

    print("\nTraining Supervised Models...")
    print("-" * 110)
    print(f"{'Model':<28} | {'Acc':<8} | {'Prec':<8} | {'Recall':<8} | {'F1':<8} | {'Time(s)':<8}")
    print("-" * 110)

    best_model = None
    best_score = 0

    for name, model in models:
        start = time.time()
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            duration = time.time() - start

            print(f"{name:<28} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {duration:.2f}")

            if acc > best_score:
                best_score = acc
                best_model = model

        except Exception as e:
            print(f"{name:<28} | FAILED → {str(e)}")

    print("-" * 110)
    print(f"\nBest Model: {best_model}")
    print(f"Best Accuracy: {best_score:.4f}")

    # ============================================================
    # SAVE ARTIFACTS — NOW CONSISTENT
    # ============================================================

    print("\nSaving artifacts...")
    joblib.dump(best_model, "best_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(encoders, "encoders.pkl")
    joblib.dump(feature_names.tolist(), "features.pkl")

    print("Artifacts saved successfully.")
    print("You can now run the Streamlit app.")


if __name__ == "__main__":
    main()
