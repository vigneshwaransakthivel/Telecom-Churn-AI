import streamlit as st
import pandas as pd
import plotly.express as px

# Configuration
st.set_page_config(page_title="Telecom Churn AI Dashboard", layout="wide", page_icon="📡")

# Load Data
@st.cache_data
def load_data():
    try:
        # Load the recommendations generated in Step 2
        df = pd.read_csv("telecom_recommendations.csv")
        # Load the raw usage data from Step 1
        raw_df = pd.read_csv("telecom_dataset.csv")
        
        # Merge them together so we have everything in one place
        merged_df = pd.merge(df, raw_df, left_on="Customer_ID", right_on="customer_id")
        return merged_df
    except FileNotFoundError:
        return None

df = load_data()

# App Header
st.title("📡 Telecom Churn AI & Retention Dashboard")
st.markdown("This dashboard acts as the **Customer Care Agent interface**. It shows the AI's real-time churn predictions and the exact personalized retention offer calculated for each customer.")

if df is None:
    st.error("Data files not found. Please ensure 'telecom_dataset.csv' and 'telecom_recommendations.csv' exist in this folder.")
else:
    # -------------------------------------------------------------
    # TOP LEVEL METRICS
    # -------------------------------------------------------------
    total_customers = len(df)
    high_risk = len(df[df['Risk_Level'] == 'High'])
    interventions = len(df[df['Offer_Category'] != 'No Action Needed'])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers Evaluated", total_customers)
    col2.metric("High Risk Customers", high_risk)
    col3.metric("Retention Interventions Generated", interventions)
    col4.metric("Healthy Customers (No Action)", total_customers - interventions)
    
    st.markdown("---")
    
    # -------------------------------------------------------------
    # MACRO VISUALIZATIONS
    # -------------------------------------------------------------
    col_graph1, col_graph2 = st.columns(2)
    
    with col_graph1:
        st.subheader("Distribution of Recommended Offers")
        # Filter out the 'No Action Needed' to see the actual offers
        offer_df = df[df['Offer_Category'] != 'No Action Needed']
        offer_counts = offer_df['Offer_Category'].value_counts().reset_index()
        offer_counts.columns = ['Offer', 'Count']
        
        fig1 = px.pie(offer_counts, values='Count', names='Offer', hole=0.4, 
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig1, use_container_width=True)
        
    with col_graph2:
        st.subheader("Data Exhaustion vs Booster Purchases")
        # Scatter plot to show how usage triggers different offers
        fig2 = px.scatter(offer_df, x="exhaustion_days", y="booster_count", color="Offer_Category",
                          hover_data=["Customer_ID", "unused_data_gb"],
                          labels={"exhaustion_days": "Days Data Exhausted", "booster_count": "Boosters Bought"},
                          opacity=0.7)
        st.plotly_chart(fig2, use_container_width=True)
        
    st.markdown("---")
    
    # -------------------------------------------------------------
    # CUSTOMER DEEP DIVE (AGENT VIEW)
    # -------------------------------------------------------------
    st.subheader("🔍 Customer Agent View (Lookup Customer)")
    
    customer_list = df['Customer_ID'].tolist()
    
    # Find the first user who got an offer to use as the default selection
    try:
        default_cust = df[df['Offer_Category'] != 'No Action Needed']['Customer_ID'].iloc[0]
        default_idx = customer_list.index(default_cust)
    except IndexError:
        default_idx = 0
        
    selected_customer = st.selectbox("Select or Type Customer ID to view their AI-recommended action plan", 
                                     customer_list, index=default_idx)
    
    if selected_customer:
        cust_data = df[df['Customer_ID'] == selected_customer].iloc[0]
        
        # Build a visual profile card
        st.markdown(f"### Customer Profile: `{selected_customer}`")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(f"**AI Risk Score:** {cust_data['churn_risk_score']} ({cust_data['Risk_Level']} Risk)")
            st.write(f"**Current Plan Price:** ₹{cust_data['plan_price']}")
            st.write(f"**Total Spend This Month:** ₹{cust_data['total_monthly_spend']}")
        with c2:
            st.warning(f"**Total Data Used:** {cust_data['used_data_gb']}GB")
            st.write(f"**Total Data Given:** {cust_data['total_monthly_gb']}GB")
            st.write(f"**Data Leftover (Wasted):** {cust_data['unused_data_gb']}GB")
        with c3:
            st.error(f"**Days Data Exhausted:** {cust_data['exhaustion_days']} / 30 days")
            st.write(f"**Boosters Purchased:** {cust_data['booster_count']}")
            st.write(f"**Money Spent on Boosters:** ₹{cust_data['booster_spend']}")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # The Recommended Action Box
        st.markdown("#### 🎯 AI Recommended Retention Action")
        
        if cust_data['Offer_Category'] == 'No Action Needed':
            st.success(f"**Action:** {cust_data['Offer_Category']}")
            st.write(cust_data['Offer_Details'])
        else:
            # Highlight the offer with a colored box
            st.error(f"**Action Required:** {cust_data['Offer_Category']}")
            
            st.markdown(f"**What to tell the customer:**")
            st.info(f"*{cust_data['Offer_Details']}*")
            
            st.write(f"**Data Carry-Forward Rule:** {cust_data['Data_Leftover_Rule']}")
            st.write(f"**Projected Customer Cost:** ₹{cust_data['Projected_Next_Spend']}")
            
            if st.button(f"Apply {cust_data['Offer_Category']} to Account", type="primary"):
                # Simulating an API call to the CRM / Billing system
                st.success(f"✅ Offer successfully queued in CRM for Customer **{selected_customer}**!")
                st.info("SMS and App Notification will be dispatched shortly.")
                st.balloons()
