#!/usr/bin/env python3
"""
Runs sequentially: load → validate → preprocess → feature engineering
"""

import os
import sys
import time
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from posthog import project_root
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from xgboost import XGBClassifier

# === Fix import path for local modules ===
# ESSENTIAL: Allows imports from src/ directory structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

#Local modules - core pipeline components
from src.data.load_data import load_data                    # Data loading with error handling
from src.data.preprocess import preprocess_data            # Basic data cleaning
from src.features.build_features import build_features     # Feature engineering (CRITICAL for model performance)
from src.utils.validate_data import validate_data    # Data quality validation


def main(args):
    """
    Main training pipeline function that orchestrates the complete ML workflow.
    
    """
    # === MLflow Setup - ESSENTIAL for experiment tracking ===
    # Configure MLflow to use local file-based tracking (not a tracking server)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
    mlruns_path = args.mlflow_uri or f"file://{project_root}/mlruns" #check if user provided the MLFlow tracking URI through command-line arguments
    #This means that the mlflow experiment data will be stored in your local mlruns folder unless given another URI
    mlflow.set_tracking_uri(mlruns_path)
    # Tells mlflow where to log your experiments
    mlflow.set_experiment(args.experiment)
    # Set(or creates) an MLFlow experiment - a logical container for all your runs 
    # args.experiment is typically a string passed from CLI or script

    with mlflow.start_run():
        # === Log hyperparameters and configuration ===
        # REQUIRED: These parameters are essential for model reproducibility
        mlflow.log_param("model", "xgboost")
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("test_size", args.test_size)

        # === Step 1: Data loading and validation ===
        print(" Loading Data ...")
        df = load_data(args.input) #load raw CSV data with error handling
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # === Critical: Data Quality Validation ===
        # This step is essential for production ML - validates data quality before training
        print("🔍 Validating data quality with Great Expectations...")
        is_valid, failed = validate_data(df)
        mlflow.log_metric("data_quality_pass",int(is_valid))

        if not is_valid:
            #Log validation failures for debugging
            import json
            mlflow.log_text(json.dumps(failed,indent=2), artifact_file="failed_expectations.json")
            raise ValueError(f"❌ Data quality check failed. Issues: {failed}")
        else:
            print("✅ Data validation passed. Logged to MLflow.")
        

        # === Stage 2: Data preprocessing ===
        print("🔧 Preprocessing data...")
        df = preprocess_data(df) #basic cleaning (handle missing value and fix data type)

        # Saved processed dataset for reproducibilty and debugging
        processed_path = os.path.join(project_root, "data", "processed", "telco_churn_processed.csv")
        os.makedirs(os.path.dirname(processed_path),exist_ok=True)
        df.to_csv(processed_path,index=False)
        print(f"✅ Processed dataset saved to {processed_path} | Shape: {df.shape}")


        # === Stage 3: Feature Engineering ===
        print("🛠️  Building features...")
        target = args.target
        if target not in df.columns:
            raise ValueError(f"Target columns '{target}' not found in data")
        
        # Apply feature engineerng transformation 
        df_enc = build_features(df, target_col=target) #Binary encoding + one-hot encoding

        # Important convert boolean columns to integer for xgboost compatibility
        for c in df_enc.select_dtypes(include =["bool"]).columns:
            df_enc[c]= df_enc[c].astype(int)
        print(f"✅ Feature engineering completed: {df_enc.shape[1]} feature")

        # === CRITICAL: Save Feature Metadata for Serving Consistency ===
        # This ensures serving pipeline uses exact same features in exact same order
        import json, joblib  #joblib is often used to save models or large python object efficiently
        artifacts_dir = os.path.join(project_root,"artifacts") 
        os.makedirs(artifacts_dir, exist_ok=True) #create a directory for your artifact inside project to store output like model file, encoders and metadata

        # Extract feature columns(exclude target)
        feature_cols = list(df_enc.drop(columns=[target]).columns)

        # Save locally for development serving
        with open(os.path.join(artifacts_dir,"feature_columns.json"),"w") as f:
            json.dump(feature_cols, f)

        # Log to MLFlow for production serving
        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt") 

        # ESSENTIAL: Save preprocessing artifacts for serving pipeline
        # These artifacts ensure training and serving use identical transformations
        preprocessing_artifacts = {
            "feature_column": feature_cols,
            "target": target
        }
        joblib.dump(preprocessing_artifacts, os.path.join(artifacts_dir,"preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir,"preprocessing.pkl"))
        print(f"✅ Saved {len(feature_cols)} feature columns for serving consistency")

        # === Stage 4: Train - Test split === 
        print("📊 Splitting data...")
        X = df_enc.drop(columns = [target])
        y = df_enc[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=args.test_size,    # Default: 20% for testing
            stratify=y,                  # Maintain class balance
            random_state=42              # Reproducible splits
        )
        print(f"✅ Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

        # === Critical : Handle class imbalance ===
        # Calculate scale_pos_weight to handle imbalanced dataset
        # This tells XGBoost to give more weight to the minority class (churners)
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() 
        print(f"📈 Class imbalance ratio: {scale_pos_weight:.2f} (applied to positive class)")
        # If your dataset is imbalanced — for example, only 10% churners (1s) — your model 
        # might learn to predict “no churn” all the time.

        # === Stage 5 : Model training with optimized parameters ===
        print("🤖 Training XGBoost model...")
        # IMPORTANT: These hyperparameters were optimized through hyperparameter tuning
        # In production, consider using hyperparameter optimization tools like Optuna
        model = XGBClassifier(
            # Tree structure parameters
            n_estimators=301,        # Number of trees (OPTIMIZED)
            learning_rate=0.034,     # Step size shrinkage (OPTIMIZED)  
            max_depth=7,            # Maximum tree depth (OPTIMIZED)
            
            # Regularization parameters
            subsample=0.95,         # Sample ratio of training instances
            colsample_bytree=0.98,  # Sample ratio of features for each tree
            
            # Performance parameters
            n_jobs=-1,              # Use all CPU cores
            random_state=42,        # Reproducible results
            eval_metric="logloss",  # Evaluation metric
            
            # ESSENTIAL: Handle class imbalance
            scale_pos_weight=scale_pos_weight  # Weight for positive class (churners)
        )

        # === Train model and track training time ===
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train time", train_time)
        print(f"✅ Model trained in {train_time:.2f} seconds")

        # Stage 6: Model Evaluation
        print("📊 Evaluating model performance...")
        
        # Generate predictions and track inference time
        t1 = time.time()
        proba = model.predict_proba(X_test)[:,1] # Get probability of churn (class 1)
        # Apply classification threshold (default: 0.35, optimized for churn detection)
        # Lower threshold = more sensitive to churn (higher recall, lower precision)
        y_pred = (proba >= args.threshold).astype(int)
        pred_time = time.time() - t1
        mlflow.log_metric("pred_time", pred_time)

        # === CRITICAL: Log Evaluation Metrics to MLflow ===
        # These metrics are essential for model comparison and monitoring
        precision = precision_score(y_test, y_pred)    # Of predicted churners, how many actually churned?
        recall = recall_score(y_test, y_pred)          # Of actual churners, how many did we catch?
        f1 = f1_score(y_test, y_pred)                  # Harmonic mean of precision and recall
        roc_auc = roc_auc_score(y_test, proba)         # Area under ROC curve (threshold-independent)

        # Log all metrics for experiment tracking
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall) 
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        print(f"🎯 Model Performance:")
        print(f"   Precision: {precision:.3f} | Recall: {recall:.3f}")
        print(f"   F1 Score: {f1:.3f} | ROC AUC: {roc_auc:.3f}")

        # Stage 7 :  === Model Serilization and Logging ===
        print("💾 Saving model to MLflow...")
        # ESSENTIAL: Log model in MLflow's standard format for serving
        mlflow.sklearn.log_model(
            model,
            artifact_path = "model" # This creates a 'model/' folder in MLflow run artifacts
        )
        print("✅ Model saved to MLflow for serving pipeline")

        # === Final performance Summary ===
        print(f"\n⏱️  Performance Summary:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Inference time: {pred_time:.4f}s")
        print(f"   Samples per second: {len(X_test)/pred_time:.0f}")

        print(f"\n📈 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, digits=3))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=" Run churn pipeline with XGBoost + MLFlow ")
    p.add_argument("--input", type = str, required=True,
                   help = "path to CSV (e.g., data/raw/Telco-Customer-Churn.csv)")
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Telco Churn")
    p.add_argument("--mlflow_uri", type=str, default=None,
                    help="override MLflow tracking URI, else uses project_root/mlruns")
    
    args = p.parse_args()
    main(args)

"""
# Use this below to run the pipeline:

python scripts/run_pipeline.py \                                            
    --input data/raw/Telco-Customer-Churn.csv \
    --target Churn

"""


