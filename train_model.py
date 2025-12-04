#!/usr/bin/env python3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def main():
    print("Loading training data...")
    try:
        df = pd.read_csv("training_data.csv")
    except FileNotFoundError:
        print("Error: training_data.csv not found. Run collect_training_data.py first.")
        return

    # Features and Target
    feature_cols = [
        "rsi", "rsi_5", "adx", "macd_hist", 
        "atr_pct", "dist_lower", "sma50_slope", 
        "vol_rel", "hour"
    ]
    
    X = df[feature_cols]
    y = df["label"]
    
    # Split Data (80% Train, 20% Test)
    # Shuffle=False to respect time series nature (train on past, test on future)?
    # Actually, for this proof of concept, random split is okay, but time-split is better.
    # Let's do a simple time-based split manually to be rigorous.
    
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")
    
    # Initialize XGBoost Classifier
    # scale_pos_weight: useful if classes are imbalanced. 
    # Our baseline is ~64% win, so it's not super imbalanced.
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n=== Model Evaluation (Test Set) ===")
    print(classification_report(y_test, y_pred))
    
    # Win Rate Analysis
    baseline_wr = y_test.mean()
    
    # Simulate "Model Gate"
    # Only take trades where model confidence > threshold
    threshold = 0.65
    selected_trades = y_test[y_prob > threshold]
    
    if len(selected_trades) > 0:
        model_wr = selected_trades.mean()
        print(f"\n=== Strategy Improvement ===")
        print(f"Baseline Win Rate: {baseline_wr:.2%}")
        print(f"Model Win Rate:    {model_wr:.2%} (Threshold > {threshold})")
        print(f"Trades Taken:      {len(selected_trades)} / {len(y_test)} ({len(selected_trades)/len(y_test):.1%})")
        
        improvement = model_wr - baseline_wr
        print(f"Improvement:       +{improvement:.2%}")
    else:
        print("\nNo trades met the confidence threshold.")

    # Feature Importance
    print("\n=== Feature Importance ===")
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance)
    
    # Save Model
    model.save_model("rubberband_xgb.json")
    print("\nModel saved to rubberband_xgb.json")

if __name__ == "__main__":
    main()
