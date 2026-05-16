import pandas as pd
import numpy as np

def run_recommendation_engine(input_csv="telecom_dataset.csv", output_csv="telecom_recommendations.csv"):
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: {input_csv} not found. Please generate the dataset first.")
        return
        
    recommendations = []
    
    for index, row in df.iterrows():
        cust_id = row['customer_id']
        risk_label = row['churn_risk_label']
        
        # Customer Data Points
        unused_gb = row['unused_data_gb']
        exhaust_days = row['exhaustion_days']
        boosters = row['booster_count']
        spend = row['total_monthly_spend']
        current_plan = row['plan_price']
        
        # Defaults
        offer_category = "No Action Needed"
        offer_details = "Customer usage is optimal. Low churn risk."
        carry_forward_rule = "N/A"
        new_projected_spend = current_plan
        
        # We only intervene if the ML model flagged them as Medium or High risk
        if risk_label in ["Medium", "High", "Medium-High"]:
            
            # -------------------------------------------------------------
            # RULE ENGINE (Applying our 7 Scenarios)
            # -------------------------------------------------------------
            
            # Scenario 1: The Data Waster (Underuser)
            if unused_gb >= 10 and exhaust_days <= 3 and boosters == 0:
                credit = round(unused_gb * 1.5, 2)
                new_projected_spend = max(0, current_plan - credit)
                offer_category = "Unused Data Credit"
                offer_details = f"Crediting ₹{credit} for {unused_gb}GB unused data. Next recharge drops to ₹{new_projected_spend}."
                carry_forward_rule = "N/A"
                
            # Scenario 4 & 7: The Power User (Heavy Exhaustion + Many Boosters)
            elif exhaust_days >= 20 and boosters >= 7:
                # The jump to ₹440 feels costly (Scenario 7 problem). We bundle Spotify to fix it.
                offer_category = "Plan Upgrade + Value Benefit"
                offer_details = "Recommend ₹440 Plan (3GB/day). Bundled with 1-month Spotify Premium to justify price jump."
                new_projected_spend = 440
                carry_forward_rule = "N/A"
                
            # Scenario 3 & 6: The Regular Booster Buyer
            elif exhaust_days >= 5 and boosters >= 3 and boosters < 7:
                # Offer the ₹49 bundle instead of separate ₹19 boosters
                offer_category = "Paid Booster Bundle"
                offer_details = "Offer 5GB Booster Bundle for ₹49. Cheaper than buying individual boosters."
                new_projected_spend = current_plan + 49
                # Addressing Scenario 6: They paid for this, so unused data must carry forward
                carry_forward_rule = "YES - Partial Carry-Forward applies to unused paid booster data."
                
            # Scenario 2 & 5: The Price-Sensitive Needer (High exhaust, 0 boosters)
            elif exhaust_days >= 10 and boosters == 0:
                # Estimate extra need (simple heuristic: 0.3GB per day of exhaustion)
                free_gb = min(10, round(exhaust_days * 0.3))
                offer_category = "Limited Free Booster"
                offer_details = f"Provide {free_gb}GB Free Booster Data. No extra cost."
                new_projected_spend = current_plan
                # Addressing Scenario 5: It's free, so it vanishes if unused.
                carry_forward_rule = "STRICT NO - Free data vanishes. Unused data reduces next month's offer limit."
                
            else:
                # Fallback for borderline or complex mixed cases
                offer_category = "Service Check-In"
                offer_details = "Send generic check-in SMS. Could be network/service quality issue."
                carry_forward_rule = "N/A"
        
        recommendations.append({
            "Customer_ID": cust_id,
            "Risk_Level": risk_label,
            "Hidden_Persona": row.get('actual_scenario', 'Unknown'),
            "Offer_Category": offer_category,
            "Offer_Details": offer_details,
            "Data_Leftover_Rule": carry_forward_rule,
            "Projected_Next_Spend": new_projected_spend
        })
        
    res_df = pd.DataFrame(recommendations)
    res_df.to_csv(output_csv, index=False)
    
    # Generate Output Summary
    print("\n=======================================================")
    print(" TELECOM CHURN AI - RECOMMENDATION ENGINE RESULTS ")
    print("=======================================================\n")
    print(f"Total customers processed: {len(df)}")
    print(f"Interventions generated: {len(res_df[res_df['Offer_Category'] != 'No Action Needed'])}\n")
    
    print("--- Offer Distribution ---")
    offer_counts = res_df['Offer_Category'].value_counts()
    for category, count in offer_counts.items():
        print(f"{category.ljust(30)} : {count} customers")
        
    print(f"\nDetailed recommendations saved to: {output_csv}")
    print("=======================================================")

if __name__ == "__main__":
    run_recommendation_engine()
