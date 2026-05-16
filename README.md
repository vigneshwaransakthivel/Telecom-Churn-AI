# 📡 AI Telecom Churn Prediction & Personalized Retention System

## 🚀 Overview
Telecom companies provide fixed prepaid recharge plans, but customers have dynamic usage patterns. This project solves the "plan-customer mismatch" and "competitor undercutting" by building a 4-layer AI system that predicts churn, diagnoses the reason, and recommends market-aware personalized offers.

---

## 🧠 The 4-Layer Architecture

### 1. The Machine Learning Engine (`train_model.py`)
- **Algorithm:** Random Forest Classifier
- **Core Features:** Learns from 8 behavioral signals (e.g., `unused_data_gb`, `exhaustion_days`, `booster_count`).
- **Function:** Scores every customer with a real probability of churn (0.0 to 1.0).

### 2. The Competitor Intelligence Feed (`competitor_plans.json`)
- **Function:** Acts as a real-time market tracker.
- **Data:** Contains live plans from Jio, Airtel, and Vi (e.g., Jio's ₹949 Gemini Pro offer).
- **Intelligence:** Allows the AI to "see" when a competitor is undercutting our prices or offering better perks.

### 3. The Production API (`api.py`)
- **Framework:** FastAPI
- **Real-Time Inference:** Receives JSON data and returns an instant decision.
- **Audit Logging:** Every single request is permanently logged for future training.
- **Competitive Matching:** Automatically triggers a "Competitive Retention Match" if a market threat is detected.

### 4. The Agent Dashboard (`dashboard.py`)
- **Framework:** Streamlit
- **Visual Analytics:** Shows portfolio health and intervention distribution.
- **Live Sandbox:** Allows agents to manually simulate scenarios and test the API's response.
- **Market Comparison:** Displays a side-by-side comparison of how our AI's offer beats rival plans from Jio and Airtel.

### 5. Continuous Learning Pipeline (`continuous_learning.py`)
- **MLOps:** Automatically merges live API logs with historical data.
- **Function:** Retrains the model on the fly to ensure the AI gets smarter every month.

---

## 🛠️ Project Execution Order

1. **`generate_dataset.py`**: Create the initial synthetic dataset.
2. **`train_model.py`**: Train the base Random Forest model.
3. **`api.py`**: Start the production server (Backend).
4. **`dashboard.py`**: Launch the agent UI (Frontend).
5. **`continuous_learning.py`**: Run this periodically to update the model with new live data.

---

## 💡 Key Design Decisions
- **Anti-Churn Strategy:** Targets "Competitive Churn" by matching high-value perks (OTT/AI Cloud) when a rival threat is detected.
- **Zero-Waste Policy:** Implemented a "Unused Data Credit" rule to appease customers who feel they are overpaying.
- **Dynamic Response:** The system is not static; it responds to market price drops in real-time.

---
*Built as a state-of-the-art Proof of Concept for AI-Driven Telecom Retention.*
