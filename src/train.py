import pandas as pd
import numpy as np
import argparse
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import os

import matplotlib.pyplot as plt
import seaborn as sns

def load_and_process(data_path):
    # Load dataset
    df = pd.read_csv(data_path)

    # Select relevant features + target
    features = ['cpu_request', 'mem_request', 'cpu_limit',
                'mem_limit', 'runtime_minutes', 'controller_kind']
    target = 'cpu_usage'

    X = df[features]
    y = df[target]

    return X, y


def build_pipeline():
    # Preprocess categorical + numeric
    categorical_features = ['controller_kind']
    numeric_features = ['cpu_request', 'mem_request',
                        'cpu_limit', 'mem_limit', 'runtime_minutes']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ('num', 'passthrough', numeric_features)
        ]
    )

    # ML model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    # Full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline


def train(data_path, model_out):
    # Load & process
    X, y = load_and_process(data_path)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build pipeline
    pipeline = build_pipeline()

    # Enable MLflow tracking
    mlflow.set_experiment("cpu_usage_prediction")
    with mlflow.start_run():
        pipeline.fit(X_train, y_train)

        # Predictions
        y_pred = pipeline.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # Log model (MLflow style)
        mlflow.sklearn.log_model(pipeline, name="cpu_usage_model")

        # 🔑 Ensure directory exists before saving
        os.makedirs(os.path.dirname(model_out), exist_ok=True)

        # Save pipeline locally (for DVC tracking)
        joblib.dump(pipeline, model_out)
        print(f"Model saved to {model_out}")
        print(f"Metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

        os.makedirs("plots", exist_ok=True)

        # 1. Predictions vs Actual
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
        plt.xlabel("Actual CPU Usage")
        plt.ylabel("Predicted CPU Usage")
        plt.title("Predicted vs Actual CPU Usage")
        plt.savefig("plots/pred_vs_actual.png")
        mlflow.log_artifact("plots/pred_vs_actual.png")
        plt.close()

        # 2. Residuals
        residuals = y_test - y_pred
        plt.figure(figsize=(6, 4))
        sns.histplot(residuals, bins=30, kde=True)
        plt.xlabel("Residuals")
        plt.title("Residual Distribution")
        plt.savefig("plots/residuals.png")
        mlflow.log_artifact("plots/residuals.png")
        plt.close()

        # 3. Feature Importance (only works for RandomForest)
        model = pipeline.named_steps['model']
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        importances = model.feature_importances_
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(8, 5))
        sns.barplot(x="Importance", y="Feature", data=fi_df)
        plt.title("Feature Importances")
        plt.tight_layout()
        plt.savefig("plots/feature_importance.png")
        mlflow.log_artifact("plots/feature_importance.png")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/dataset.csv")
    parser.add_argument("--out", type=str, default="models/cpu_model.pkl")
    args = parser.parse_args()

    train(args.data, args.out)
