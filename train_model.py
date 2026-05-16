import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def build_ml_model():
    print("Loading historical data...")
    try:
        df = pd.read_csv("telecom_dataset.csv")
    except FileNotFoundError:
        print("Dataset not found. Run generate_dataset.py first.")
        return

    # 1. Create a Target Variable (Historical Churn)
    # In a real telecom, this column exists (1 = Left, 0 = Stayed). 
    # We will derive a realistic history based on the behavior patterns.
    np.random.seed(42)
    def simulate_historical_churn(row):
        # Base probability of having left in the past based on their scenario
        if row['actual_scenario'] == 0: prob = 0.05
        elif row['actual_scenario'] == 1: prob = 0.60
        else: prob = 0.85
        return 1 if np.random.rand() < prob else 0
        
    df['historical_churn'] = df.apply(simulate_historical_churn, axis=1)
    
    # 2. Define Features (What the ML model is allowed to learn from)
    features = [
        'plan_price', 'used_data_gb', 'unused_data_gb', 
        'exhaustion_days', 'booster_count', 'booster_spend', 
        'total_monthly_spend', 'recharge_delay_days'
    ]
    
    X = df[features]
    y = df['historical_churn']
    
    # 3. Split into Train (80%) and Test (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest AI Model...")
    # Initialize a Random Forest with 100 decision trees
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluate the Model (Testing it on data it hasn't seen yet)
    y_pred = model.predict(X_test)
    print("\n======================================")
    print(" ML MODEL EVALUATION RESULTS ")
    print("======================================")
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))
    
    # 5. Feature Importance (Why do customers churn?)
    print("\n--- Feature Importance (What drives churn the most?) ---")
    importances = model.feature_importances_
    for name, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
        print(f"{name.ljust(25)}: {imp:.4f} ({round(imp*100)}%)")
        
    # 6. Generate the REAL AI Risk Scores for all customers
    # This replaces our old "simulated" scores with actual model predictions
    probabilities = model.predict_proba(X)[:, 1] # Probability of Class 1 (Churn)
    
    df['churn_risk_score'] = np.round(probabilities, 2)
    
    # Map the probability to a label
    def assign_risk_label(score):
        if score < 0.35: return "Low"
        elif score < 0.65: return "Medium"
        else: return "High"
        
    df['churn_risk_label'] = df['churn_risk_score'].apply(assign_risk_label)
    
    # Clean up the dataset (remove the helper columns)
    df.drop(columns=['historical_churn'], inplace=True)
    
    # 7. Save the trained model and the newly scored dataset
    joblib.dump(model, 'churn_prediction_model.pkl')
    print("\nModel saved to 'churn_prediction_model.pkl'")
    
    df.to_csv("telecom_dataset.csv", index=False)
    print("Dataset updated with REAL Machine Learning predictions!")
    print("======================================")

if __name__ == "__main__":
    build_ml_model()
