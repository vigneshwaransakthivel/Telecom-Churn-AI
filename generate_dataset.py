import pandas as pd
import numpy as np
import random
import os

def generate_telecom_data(n=1000):
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    data = []
    
    # Generate 'n' random customers
    for i in range(n):
        customer_id = f"CUST_{i:04d}"
        
        # Determine which scenario group the customer falls into:
        # 0: Normal User (Happy) - 40%
        # 1: Underuser (Data Waster) - 15%
        # 2: Price Sensitive (Needs data, won't buy boosters) - 15%
        # 3: Booster Buyer - 15%
        # 4: Heavy User - 15%
        scenario = np.random.choice([0, 1, 2, 3, 4], p=[0.4, 0.15, 0.15, 0.15, 0.15])
        
        # Standard Plan assumed for simplicity
        plan_price = 389
        daily_limit_gb = 2.0
        total_monthly_gb = 60.0
        
        # Inject behavior metrics based on scenario
        if scenario == 0: # Normal User
            used_data_gb = np.random.uniform(45, 55)
            exhaustion_days = np.random.randint(0, 3)
            booster_count = 0
            recharge_delay_days = np.random.randint(0, 2)
            
        elif scenario == 1: # Underuser
            used_data_gb = np.random.uniform(20, 45)
            exhaustion_days = 0
            booster_count = 0
            recharge_delay_days = np.random.randint(0, 3)
            
        elif scenario == 2: # Price Sensitive
            used_data_gb = total_monthly_gb # Used exactly what they had
            exhaustion_days = np.random.randint(12, 22)
            booster_count = 0
            recharge_delay_days = np.random.randint(2, 6)
            
        elif scenario == 3: # Booster Buyer
            used_data_gb = total_monthly_gb + np.random.uniform(5, 12)
            exhaustion_days = np.random.randint(8, 15)
            booster_count = np.random.randint(3, 7)
            recharge_delay_days = 0
            
        elif scenario == 4: # Heavy User
            used_data_gb = total_monthly_gb + np.random.uniform(15, 30)
            exhaustion_days = np.random.randint(22, 30)
            booster_count = np.random.randint(7, 15)
            recharge_delay_days = 0
            
        # Computed fields
        unused_data_gb = max(0, total_monthly_gb - used_data_gb) if booster_count == 0 else 0
        booster_spend = booster_count * 19 # Assuming ₹19 per booster
        total_monthly_spend = plan_price + booster_spend
        
        # ML Churn Risk Simulation (Ground truth for our model)
        if scenario == 0:
            churn_risk_score = np.random.uniform(0.01, 0.20)
            churn_label = "Low"
        elif scenario == 1:
            churn_risk_score = np.random.uniform(0.40, 0.70)
            churn_label = "Medium"
        elif scenario in [2, 3, 4]:
            churn_risk_score = np.random.uniform(0.75, 0.99)
            churn_label = "High"
            
        data.append({
            "customer_id": customer_id,
            "plan_price": plan_price,
            "daily_limit_gb": daily_limit_gb,
            "total_monthly_gb": total_monthly_gb,
            "used_data_gb": round(used_data_gb, 1),
            "unused_data_gb": round(unused_data_gb, 1),
            "exhaustion_days": exhaustion_days,
            "booster_count": booster_count,
            "booster_spend": booster_spend,
            "total_monthly_spend": total_monthly_spend,
            "recharge_delay_days": recharge_delay_days,
            "actual_scenario": scenario, # Hidden label to test our logic engine later
            "churn_risk_score": round(churn_risk_score, 2),
            "churn_risk_label": churn_label
        })
        
    df = pd.DataFrame(data)
    
    # Ensure directory exists if needed, or save in current
    filepath = "telecom_dataset.csv"
    df.to_csv(filepath, index=False)
    print(f"Generated synthetic dataset with {len(df)} records.")
    print(f"Saved to: {os.path.abspath(filepath)}")
    
    # Print a quick preview
    print("\nSample Data (First 5 Rows):")
    print(df.head())

if __name__ == "__main__":
    generate_telecom_data(2000) # Generate 2000 customers
