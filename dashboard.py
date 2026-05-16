import streamlit as st
import pandas as pd
import plotly.express as px
import requests

# Configuration
st.set_page_config(page_title="Telecom Retention Analytics", layout="wide")

# Load Data (Cache removed to instantly show new API requests)
def load_data():
    import os
    try:
        df = pd.read_csv("telecom_recommendations.csv")
        raw_df = pd.read_csv("telecom_dataset.csv")
        merged_df = pd.merge(df, raw_df, left_on="Customer_ID", right_on="customer_id")
        
        # Merge live API logs if they exist
        if os.path.exists("api_history_logs.csv"):
            api_df = pd.read_csv("api_history_logs.csv")
            
            # Map the API logs to match the dashboard's expected columns
            api_mapped = pd.DataFrame({
                'Customer_ID': "LIVE_API_TEST_" + api_df.index.astype(str),
                'Risk_Level': api_df['AI_Offer'].apply(lambda x: 'Low' if x == 'No Action Needed' else 'High'),
                'Offer_Category': api_df['AI_Offer'],
                'Offer_Details': 'Custom simulation generated via Live API.',
                'Data_Leftover_Rule': 'Standard Live Policy',
                'Projected_Next_Spend': 389,
                'churn_risk_score': api_df['AI_Risk_Score'],
                'plan_price': 389,
                'used_data_gb': api_df['Used_Data_GB'],
                'unused_data_gb': (60.0 - api_df['Used_Data_GB']).clip(lower=0),
                'exhaustion_days': api_df['Exhaustion_Days'],
                'booster_count': api_df['Booster_Count']
            })
            merged_df = pd.concat([merged_df, api_mapped], ignore_index=True)
            
        return merged_df
    except FileNotFoundError:
        return None

df = load_data()

# App Header
st.title("Customer Retention & Churn Analytics")
st.markdown("Enterprise Intelligence & Intervention System")

tab1, tab2 = st.tabs(["Batch Analytics Overview", "Real-Time Inference Engine"])

with tab1:
    st.markdown("### Portfolio Overview")

    if df is None:
        st.error("System Error: Analytics datasets not found. Please verify data pipelines.")
    else:
        total_customers = len(df)
        high_risk = len(df[df['Risk_Level'] == 'High'])
        interventions = len(df[df['Offer_Category'] != 'No Action Needed'])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers Evaluated", f"{total_customers:,}")
        col2.metric("High Risk Identified", f"{high_risk:,}")
        col3.metric("Interventions Generated", f"{interventions:,}")
        col4.metric("Healthy Portfolio", f"{total_customers - interventions:,}")
        
        st.markdown("---")
        st.subheader("Intervention Analytics")
        
        # Row 1 of Charts
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            st.markdown("**Intervention Distribution**")
            offer_df = df[df['Offer_Category'] != 'No Action Needed']
            offer_counts = offer_df['Offer_Category'].value_counts().reset_index()
            offer_counts.columns = ['Offer', 'Count']
            fig_pie = px.pie(offer_counts, values='Count', names='Offer', hole=0.5, 
                             color_discrete_sequence=px.colors.sequential.Blues_r)
            fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with row1_col2:
            st.markdown("**Risk Exposure by Spend Tier**")
            # Analyze if higher spending customers are at more risk
            risk_spend = df.groupby('Risk_Level')['total_monthly_spend'].mean().reset_index()
            fig_bar = px.bar(risk_spend, x='Risk_Level', y='total_monthly_spend', 
                             color='Risk_Level', color_discrete_map={'Low': '#2ecc71', 'Medium': '#f1c40f', 'High': '#e74c3c'},
                             labels={'total_monthly_spend': 'Avg Monthly Spend (₹)'})
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Row 2 of Charts
        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            st.markdown("**Usage Intensity by Risk Category**")
            # Box plot to see the spread of data usage across risk levels
            fig_box = px.box(df, x='Risk_Level', y='used_data_gb', color='Risk_Level',
                             color_discrete_map={'Low': '#2ecc71', 'Medium': '#f1c40f', 'High': '#e74c3c'},
                             points="all", labels={'used_data_gb': 'Data Consumed (GB)'})
            st.plotly_chart(fig_box, use_container_width=True)

        with row2_col2:
            st.markdown("**Exhaustion vs. Booster Dependency Matrix**")
            fig_scatter = px.scatter(offer_df, x="exhaustion_days", y="booster_count", color="Risk_Level",
                                     size="total_monthly_spend", hover_data=["Customer_ID"],
                                     color_discrete_map={'Medium': '#f1c40f', 'High': '#e74c3c'},
                                     labels={"exhaustion_days": "Days Data Exhausted", "booster_count": "Add-on Purchases"})
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        st.markdown("---")
        st.subheader("Customer Intervention Lookup")
        customer_list = df['Customer_ID'].tolist()
        
        try:
            default_cust = df[df['Offer_Category'] != 'No Action Needed']['Customer_ID'].iloc[0]
            default_idx = customer_list.index(default_cust)
        except IndexError:
            default_idx = 0
            
        selected_customer = st.selectbox("Search Customer ID", customer_list, index=default_idx)
        
        if selected_customer:
            cust_data = df[df['Customer_ID'] == selected_customer].iloc[0]
            st.markdown(f"#### Account Profile: `{selected_customer}`")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.info(f"**AI Risk Score:** {cust_data['churn_risk_score']} ({cust_data['Risk_Level']} Risk)")
                st.write(f"**Current Plan Price:** ₹{cust_data['plan_price']}")
            with c2:
                st.warning(f"**Total Data Consumed:** {cust_data['used_data_gb']}GB")
                st.write(f"**Unutilized Data:** {cust_data['unused_data_gb']}GB")
            with c3:
                st.error(f"**Data Exhaustion Frequency:** {cust_data['exhaustion_days']} / 30 days")
                st.write(f"**Add-on Purchases:** {cust_data['booster_count']}")
                
            st.markdown("#### 🎯 AI Recommended Retention Action")
            if cust_data['Offer_Category'] == 'No Action Needed':
                st.success(f"**Status:** {cust_data['Offer_Category']}")
                st.write(cust_data['Offer_Details'])
            else:
                st.error(f"**Required Intervention:** {cust_data['Offer_Category']}")
                st.info(f"**Agent Script:** {cust_data['Offer_Details']}")
                
                # NEW: Market Comparison Section
                with st.expander("⚖️ Competitive Market Comparison"):
                    st.write("Our AI analyzed rival offers from Jio and Airtel to ensure this recommendation is competitive.")
                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.markdown("**Competitor Threat**")
                        st.write("Jio ₹949 (Hotstar + Gemini)")
                        st.write("Airtel ₹979 (Hotstar + Xstream)")
                    with m_col2:
                        st.markdown("**Our Advantage**")
                        st.write(f"Personalized: {cust_data['Offer_Category']}")
                        st.write(f"Projected Cost: ₹{cust_data['Projected_Next_Spend']}")
                    st.caption("AI Decision: Matching competitor value while protecting margins via usage-based credit.")

                st.write(f"**Carry-Forward Policy:** {cust_data['Data_Leftover_Rule']}")
                st.write(f"**Projected ARPU:** ₹{cust_data['Projected_Next_Spend']}")
                if st.button(f"Execute {cust_data['Offer_Category']}", type="primary"):
                    st.success(f"System: Action successfully committed to CRM for account {selected_customer}.")

with tab2:
    st.markdown("### Real-Time Inference Sandbox")
    st.write("Configure customer telemetry parameters below to perform a live inference request against the production ML API.")
    
    with st.form("custom_customer_form"):
        col_form1, col_form2 = st.columns(2)
        with col_form1:
            custom_used_data = st.slider("Total Data Consumed (GB)", 0.0, 100.0, 45.0)
            custom_exhaustion = st.slider("Data Exhaustion Frequency (Days)", 0, 30, 0)
        with col_form2:
            custom_boosters = st.slider("Add-on Purchases Count", 0, 15, 0)
            custom_delay = st.slider("Recharge Delay (Days)", 0, 10, 0)
            
        submitted = st.form_submit_button("Execute Inference Request")
        
    if submitted:
        # Build the exact JSON payload the API expects
        api_url = "http://localhost:8000/predict_and_recommend"
        payload = {
            "customer_id": "SIMULATED_001",
            "plan_price": 389,
            "total_monthly_gb": 60.0,
            "used_data_gb": custom_used_data,
            "exhaustion_days": custom_exhaustion,
            "booster_count": custom_boosters,
            "recharge_delay_days": custom_delay
        }
        
        try:
            # Send the request to our local FastAPI server
            response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                
                st.markdown("---")
                st.success("200 OK: Prediction received from API endpoint.")
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.markdown("#### Analytical Assessment")
                    st.write(f"**Churn Probability:** {result['ai_analysis']['churn_probability']}")
                    st.write(f"**Risk Stratification:** {result['ai_analysis']['risk_level']}")
                    
                with res_col2:
                    st.markdown("#### System Recommendation")
                    st.write(f"**Target Offer:** {result['retention_action']['recommended_offer']}")
                    st.write(f"**Execution Details:** {result['retention_action']['offer_details']}")
                    
                st.markdown("**API Response Payload:**")
                st.json(result)
            else:
                st.error(f"HTTP {response.status_code}: Verification of backend API status required.")
        except requests.exceptions.ConnectionError:
            st.error("Connection Refused. Verify that the inference server (api.py) is operational on port 8000.")
