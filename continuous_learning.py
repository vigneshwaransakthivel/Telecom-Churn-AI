import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def retrain_model():
    print("======================================")
    print(" CONTINUOUS LEARNING PIPELINE STARTED")
    print("======================================")
    
    # 1. Load the original base dataset
    df_base = pd.read_csv("telecom_dataset.csv")
    print(f"1. Loaded base dataset: {len(df_base)} historical records.")
    
    # We need to recreate the target variable for the base dataset to train on
    np.random.seed(42)
    def simulate_historical_churn(row):
        if row['actual_scenario'] == 0: prob = 0.05
        elif row['actual_scenario'] == 1: prob = 0.60
        else: prob = 0.85
        return 1 if np.random.rand() < prob else 0
        
    df_base['historical_churn'] = df_base.apply(simulate_historical_churn, axis=1)
    
    # 2. Check for new Live API Logs
    log_file = "api_history_logs.csv"
    if os.path.exists(log_file):
        df_logs = pd.read_csv(log_file)
        print(f"2. Found {len(df_logs)} new records in Live API Logs.")
        
        # Enrich the live logs so they match the ML training schema
        df_logs['plan_price'] = 389
        df_logs['total_monthly_gb'] = 60.0
        
        # Calculate unused data based on what they used
        df_logs['unused_data_gb'] = np.maximum(0, df_logs['total_monthly_gb'] - df_logs['Used_Data_GB'])
        df_logs.loc[df_logs['Booster_Count'] > 0, 'unused_data_gb'] = 0
        
        df_logs['booster_spend'] = df_logs['Booster_Count'] * 19
        df_logs['total_monthly_spend'] = df_logs['plan_price'] + df_logs['booster_spend']
        df_logs['recharge_delay_days'] = 0 
        
        # Rename columns to match
        df_logs = df_logs.rename(columns={
            'Customer_ID': 'customer_id',
            'Used_Data_GB': 'used_data_gb',
            'Exhaustion_Days': 'exhaustion_days',
            'Booster_Count': 'booster_count'
        })
        
        # In a real system, we look at the 'offer_performance_report' to see if they churned.
        # Here we simulate the ground truth for the new data.
        df_logs['historical_churn'] = np.where(df_logs['AI_Risk_Score'] > 0.65, 1, 0)
        
        # 3. Merge the datasets together
        features_to_keep = [
            'plan_price', 'used_data_gb', 'unused_data_gb', 
            'exhaustion_days', 'booster_count', 'booster_spend', 
            'total_monthly_spend', 'recharge_delay_days', 'historical_churn'
        ]
        
        df_combined = pd.concat([df_base[features_to_keep], df_logs[features_to_keep]], ignore_index=True)
        print(f"3. Merged Data! New training size: {len(df_combined)} records.")
    else:
        print("2. No API logs found. Using base dataset only.")
        features_to_keep = [
            'plan_price', 'used_data_gb', 'unused_data_gb', 
            'exhaustion_days', 'booster_count', 'booster_spend', 
            'total_monthly_spend', 'recharge_delay_days', 'historical_churn'
        ]
        df_combined = df_base[features_to_keep]
        
    # 4. Retrain the AI Model
    features = features_to_keep[:-1] # Everything except the target
    
    X = df_combined[features]
    y = df_combined['historical_churn']
    
    # We change the random state to shuffle the data differently and make it robust
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    
    print("4. Retraining Random Forest AI Model on new data...")
    # Making the model slightly deeper (smarter) because it has more data
    model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=101) 
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    
    print("\n======================================")
    print(" ML CONTINUOUS LEARNING RESULTS ")
    print("======================================")
    print(f"New Model Accuracy: {accuracy:.2f}%")
    
    # 5. Overwrite the old model with the new, smarter one
    joblib.dump(model, 'churn_prediction_model.pkl')
    print("New model successfully compiled and saved to 'churn_prediction_model.pkl'")
    print("NOTE: Restart the API Server for the new model to take effect.")
    print("======================================")

if __name__ == "__main__":
    retrain_model()
