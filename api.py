from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
import csv
from datetime import datetime
import os
import json

app = FastAPI(
    title="📡 Telecom Churn AI - Real-Time Production API",
    description="Now with Competitor Intelligence. This API analyzes usage AND market competition to prevent churn.",
    version="2.0"
)

# Load the trained Random Forest model
try:
    model = joblib.load("churn_prediction_model.pkl")
    print("ML Model Loaded Successfully")
except FileNotFoundError:
    model = None
    print("Error: ML Model not found. Train the model first.")

# Load Competitor Plans
def load_competitor_data():
    try:
        with open("competitor_plans.json", "r") as f:
            return json.load(f)
    except:
        return []

# Define payload schema
class CustomerData(BaseModel):
    customer_id: str
    plan_price: int
    total_monthly_gb: float
    used_data_gb: float
    exhaustion_days: int
    booster_count: int
    recharge_delay_days: int

@app.get("/")
def health_check():
    return {"status": "API is Active", "model_loaded": model is not None}

@app.post("/predict_and_recommend")
def get_recommendation(data: CustomerData):
    if model is None:
        return {"error": "ML Model not found."}
        
    # 1. Feature Engineering
    unused_gb = max(0, data.total_monthly_gb - data.used_data_gb) if data.booster_count == 0 else 0
    booster_spend = data.booster_count * 19
    total_spend = data.plan_price + booster_spend
    
    features = pd.DataFrame([{
        'plan_price': data.plan_price, 'used_data_gb': data.used_data_gb, 'unused_data_gb': unused_gb,
        'exhaustion_days': data.exhaustion_days, 'booster_count': data.booster_count,
        'booster_spend': booster_spend, 'total_monthly_spend': total_spend, 'recharge_delay_days': data.recharge_delay_days
    }])
    
    # 2. ML Churn Prediction
    churn_prob = model.predict_proba(features)[0][1]
    risk_label = "Low" if churn_prob < 0.35 else "Medium" if churn_prob < 0.65 else "High"
    
    # 3. Competitor Analysis (The new "Eyes")
    competitor_plans = load_competitor_data()
    best_competitor_offer = None
    market_threat = False
    
    # Simple logic: Is there a plan with similar data for less money or more perks?
    # For demo: assume our user is on a "Standard" or "Premium" tier based on price
    user_tier = "Premium" if data.plan_price > 500 else "Standard"
    
    for plan in competitor_plans:
        if plan['tier'] == user_tier:
            # If competitor is cheaper OR has significantly more perks (like the Jio Gemini offer)
            if plan['price'] < data.plan_price or "Google Gemini Pro (18 months)" in plan['perks']:
                best_competitor_offer = plan
                market_threat = True
                break

    # 4. Recommendation Logic (Dynamic & Competitive)
    offer_category = "No Action Needed"
    offer_details = "Customer usage is optimal."
    new_spend = data.plan_price
    
    # Trigger competitive match if there is a threat, even if churn risk is medium
    if market_threat and risk_label in ["Medium", "High"]:
        offer_category = "Competitive Retention Match"
        offer_details = f"Detected superior offer from {best_competitor_offer['provider']} (₹{best_competitor_offer['price']}). Match with Loyalty Discount + {best_competitor_offer['perks'][0]} equivalent."
        new_spend = min(data.plan_price, best_competitor_offer['price'])
    
    elif risk_label in ["Medium", "High"]:
        # Fallback to standard behavioral rules
        if unused_gb >= 10 and data.exhaustion_days <= 3 and data.booster_count == 0:
            offer_category = "Unused Data Credit"
            credit = round(unused_gb * 1.5, 2)
            offer_details = f"Crediting ₹{credit} for unused data."
            new_spend = max(0, data.plan_price - credit)
        elif data.exhaustion_days >= 20:
            offer_category = "Plan Upgrade + Value Benefit"
            offer_details = "Recommend higher tier plan + Premium OTT Bundle."
            new_spend = data.plan_price + 100 # Simulated upgrade price
        else:
            offer_category = "Retention Booster"
            offer_details = "Providing 5GB emergency data to ensure service continuity."
            
    # 5. Logging
    log_file = "api_history_logs.csv"
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'Customer_ID', 'Used_Data_GB', 'Exhaustion_Days', 'Booster_Count', 'AI_Risk_Score', 'AI_Offer', 'Market_Threat'])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), data.customer_id, data.used_data_gb, data.exhaustion_days, data.booster_count, round(churn_prob, 2), offer_category, market_threat])
            
    return {
        "customer_id": data.customer_id,
        "ai_analysis": {
            "churn_probability": round(churn_prob, 2),
            "risk_level": risk_label,
            "market_competition_detected": market_threat
        },
        "retention_action": {
            "recommended_offer": offer_category,
            "offer_details": offer_details,
            "competitor_benchmark": best_competitor_offer['provider'] if market_threat else "None",
            "projected_cost": new_spend
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
