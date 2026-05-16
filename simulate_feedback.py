import pandas as pd
import numpy as np

def run_feedback_simulation():
    try:
        df = pd.read_csv("telecom_recommendations.csv")
    except FileNotFoundError:
        print("Error: telecom_recommendations.csv not found.")
        return
        
    # We only care about tracking customers who actually received an offer
    offers_df = df[df['Offer_Category'] != 'No Action Needed'].copy()
    
    np.random.seed(101)
    
    # ---------------------------------------------------------
    # Simulate Conversion Rates based on Offer Psychology
    # ---------------------------------------------------------
    # A well-targeted offer should have a high conversion rate.
    # We assign realistic baseline probabilities for each offer type.
    conversion_rates = {
        "Unused Data Credit": 0.85,             # Extremely High: Getting money back is a no-brainer
        "Limited Free Booster": 0.75,           # Very High: Free data for those who need it
        "Plan Upgrade + Value Benefit": 0.60,   # Medium-High: The Spotify bundle makes it tempting, but it's still an upgrade
        "Paid Booster Bundle": 0.55             # Medium: Customer still has to spend money, so some will decline
    }
    
    outcomes = []
    
    for index, row in offers_df.iterrows():
        offer = row['Offer_Category']
        base_rate = conversion_rates.get(offer, 0.50)
        
        # Add a tiny bit of random variance to make the data realistic
        prob_of_accepting = np.random.uniform(base_rate - 0.05, base_rate + 0.05)
        
        # Did the customer accept the offer?
        if np.random.rand() < prob_of_accepting:
            status = "Offer Accepted"
            churned = "No (Retained)"  # If they accepted the offer, we saved them!
        else:
            status = "Offer Ignored"
            # If they ignored the offer, they were already High Risk, so chance of churning is very high
            churned = "Yes (Churned)" if np.random.rand() < 0.75 else "No (Retained)"
            
        outcomes.append({
            "Customer_ID": row['Customer_ID'],
            "Offer_Category": offer,
            "Customer_Response": status,
            "Final_Account_Status": churned
        })
        
    feedback_df = pd.DataFrame(outcomes)
    feedback_df.to_csv("offer_performance_report.csv", index=False)
    
    # Generate Output Summary
    print("\n=======================================================")
    print(" TELECOM AI - FEEDBACK LOOP & PERFORMANCE REPORT ")
    print("=======================================================\n")
    
    total_offers = len(feedback_df)
    accepted = len(feedback_df[feedback_df['Customer_Response'] == 'Offer Accepted'])
    retained = len(feedback_df[feedback_df['Final_Account_Status'] == 'No (Retained)'])
    
    print(f"Total Offers Sent:   {total_offers}")
    print(f"Offers Accepted:     {accepted} ({round(accepted/total_offers*100, 1)}%)")
    print(f"Overall Retention:   {retained} ({round(retained/total_offers*100, 1)}%)\n")
    
    print("--- Conversion Success Rate by Offer Type ---")
    grouped = feedback_df.groupby('Offer_Category')['Customer_Response'].apply(lambda x: (x == 'Offer Accepted').mean() * 100).round(1)
    
    # Sort for better presentation
    grouped = grouped.sort_values(ascending=False)
    
    for offer, rate in grouped.items():
        print(f"{offer.ljust(30)} : {rate}% Acceptance")
        
    print(f"\nDetailed report saved to: offer_performance_report.csv")
    print("=======================================================")

if __name__ == "__main__":
    run_feedback_simulation()
