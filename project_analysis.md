# 📡 Telecom Churn Prediction & Personalized Retention System — Full Analysis

---

## 🧩 Problem Statement (Deep Analysis)

### The Core Gap in Traditional Telecom Retention

Traditional churn prevention operates on a **one-size-fits-all** model:
- Detect churn → Send a generic offer (discount, extra data, free SMS)
- This fails because it **ignores the "why"** behind churn

The fundamental insight here is:

> **Churn is a symptom. The cause is a mismatch between what the plan offers and what the customer actually needs.**

### Why Fixed Plans Fail Individual Customers

| Customer Type | Current Plan | Their Reality | Generic Offer Result |
|---|---|---|---|
| Light user | ₹389 – 2GB/day | Uses 400MB/day, wastes 1.6GB | Extra data = useless |
| Heavy user | ₹389 – 2GB/day | Runs out by noon, buys boosters | Discount = still not enough data |
| Booster-dependent | ₹389 – 2GB/day | Buys ₹20 boosters daily | Better booster bundle is what they need |
| Cost-sensitive | ₹699 – 3GB/day | Paying more than needed | A value-added perk (e.g., Spotify) justifies the price |
| Complainer | Any plan | Network/service issues | Data offers don't solve their frustration |
| Late recharger | Any plan | Keeps running out of balance | Autopay nudge or grace period plan |

---

## 🎯 The Three-Layer Solution Architecture

### Layer 1 — Churn Prediction
**Goal:** Determine *if* a customer is at risk of leaving.

**Risk Levels:**
- 🟢 **Low Risk** — Engaged, satisfied, consistent usage
- 🟡 **Medium Risk** — Some signs of dissatisfaction or inconsistency
- 🔴 **High Risk** — Clear behavioral signals of likely departure

### Layer 2 — Churn Reason Identification
**Goal:** Determine *why* the customer might leave.

This is the **differentiating layer** — most systems stop at Layer 1. This system goes deeper.

**Churn Reason Categories:**
| Reason Code | Description | Key Signals |
|---|---|---|
| `DATA_WASTE` | Plan has more data than needed | Low data usage %, high leftover |
| `DATA_INSUFFICIENT` | Plan doesn't cover daily needs | Daily exhaustion, frequent boosters |
| `COST_SENSITIVE` | Plan feels too expensive | Low income tier, late recharges |
| `SERVICE_ISSUE` | Network or service complaints | High complaint count |
| `ENGAGEMENT_DROP` | Reduced overall usage | Drop in calls, data, SMS over time |
| `BOOSTER_DEPENDENT` | Relies on add-ons for daily usage | High booster purchase frequency |
| `RECHARGE_IRREGULAR` | Inconsistent or late recharges | Recharge delay pattern |
| `COMPETITOR_PULL` | Likely switching to a competitor | Port requests, sudden inactivity |

### Layer 3 — Personalized Offer Recommendation
**Goal:** Recommend the *right* intervention for the *right* reason.

**Offer-to-Reason Mapping (Initial Framework):**
| Churn Reason | Recommended Offer Type | Example |
|---|---|---|
| `DATA_WASTE` | Downgrade to a more affordable plan | Move from ₹389 to ₹239 with 1GB/day |
| `DATA_INSUFFICIENT` | Upgrade plan or offer data-boost bundle | ₹489 with 3GB/day or ₹449 with booster pack |
| `COST_SENSITIVE` | Discount, cashback, or value add-on | ₹50 off next recharge / Spotify Free |
| `SERVICE_ISSUE` | Service credit + escalation flag | ₹30 credit + priority support |
| `ENGAGEMENT_DROP` | Re-engagement offer | "Come back" bonus data for 7 days |
| `BOOSTER_DEPENDENT` | Booster bundle plan | ₹50 unlimited booster pack for 28 days |
| `RECHARGE_IRREGULAR` | Auto-recharge incentive | 5% cashback on autopay setup |
| `COMPETITOR_PULL` | High-value retention offer | Best plan + OTT bundle |

---

## 📊 Behavioral Signal Framework

### Input Features (Customer Behavior Signals)

**Data Usage Signals:**
- `avg_daily_data_used_gb` — Average GB consumed per day
- `plan_daily_data_limit_gb` — What the plan offers
- `data_utilization_pct` — % of daily data actually used
- `data_exhaustion_days` — Days per month where data ran out
- `data_leftover_avg_gb` — Average unused data per day

**Booster & Add-on Signals:**
- `booster_purchases_per_month` — How many boosters bought
- `booster_spend_per_month` — Money spent on boosters
- `addon_frequency` — Frequency of purchasing add-ons

**Recharge Behavior Signals:**
- `avg_recharge_delay_days` — How late the customer recharges after expiry
- `recharge_consistency_score` — Regularity of recharging
- `last_recharge_days_ago` — Recency of last recharge

**Complaint & Support Signals:**
- `complaint_count_last_3_months` — Number of complaints raised
- `complaint_type` — Category of complaints (network, billing, speed)
- `unresolved_complaint_flag` — Whether issues remain unresolved

**Engagement Signals:**
- `avg_daily_call_minutes` — Call usage
- `avg_monthly_sms` — SMS usage
- `usage_trend` — Is usage increasing or decreasing over time?
- `active_days_per_month` — Days with any activity

**Plan & Financial Signals:**
- `current_plan_price` — What they're paying
- `plan_duration_days` — Validity period
- `customer_tenure_months` — How long they've been with the operator
- `plan_change_frequency` — How often they switch plans

---

## 🔄 System Flow (End-to-End)

```
Customer Behavior Data (last 30–90 days)
            │
            ▼
    ┌───────────────────┐
    │  Feature Engineering │  ← Compute derived signals
    └───────────────────┘
            │
            ▼
    ┌───────────────────┐
    │  Churn Risk Model  │  ← ML classifier (Low / Medium / High)
    └───────────────────┘
            │
            ▼
    ┌───────────────────────┐
    │  Churn Reason Engine   │  ← Rule-based + ML explainability
    └───────────────────────┘
            │
            ▼
    ┌────────────────────────────┐
    │  Offer Recommendation Engine│  ← Reason → Offer mapping + ranking
    └────────────────────────────┘
            │
            ▼
    Personalized Retention Action
    (CRM push / Agent alert / App notification)
```

---

## 🏗️ System Design Considerations

### ML Model Choice
- **Churn Prediction:** Gradient Boosting (XGBoost / LightGBM) — handles tabular data well, interpretable with SHAP
- **Reason Identification:** Rule-based logic (transparent, auditable) layered with SHAP feature importance
- **Offer Ranking:** Could evolve into a multi-armed bandit or recommendation system as feedback data grows

### Explainability (Critical for Telecom Use Cases)
- Customer care agents need to **understand why** an offer is being made
- SHAP values can surface top 3 reasons for any churn prediction
- Example: *"This customer is at high churn risk because: (1) data exhausted 18/30 days, (2) bought 12 boosters this month, (3) delayed recharge by 4 days"*

### Feedback Loop
- Track whether recommended offers were accepted
- Track post-offer churn rate
- Continuously retrain models with new outcomes

---

## ⚠️ Key Design Challenges

| Challenge | Description | Approach |
|---|---|---|
| **Cold Start** | New customers with little behavioral history | Use plan + demographic signals for early prediction |
| **Reason Overlap** | A customer may have multiple churn reasons | Multi-label reason assignment with priority ranking |
| **Offer Fatigue** | Customers ignore repeated offers | Cap offer frequency; rotate offer types |
| **Data Quality** | Missing or inconsistent usage logs | Imputation + anomaly detection in preprocessing |
| **Business Constraints** | Not all offers can be given to all customers | Rule-layer filters (margin, eligibility, tenure) |

---

## 📌 What's Needed Next

> The user will provide specific **cases and scenarios** that define how the system should behave for different customer profiles. These will be used to:
> 1. Define **decision rules** for the Churn Reason Engine
> 2. Map **offer logic** for each scenario
> 3. Validate the **end-to-end recommendation accuracy**

**Ready to receive scenarios. ✅**

---

*Analysis by Antigravity | Project: Telecom Churn AI System | May 2026*
