# Retain-AI

### AI-Powered Customer Retention Decision Platform

Retain-AI is an end-to-end customer retention platform that combines machine learning, explainable AI, business economics, and LLMs to answer a practical business question:

> **Which customers are most likely to churn, why are they at risk, and which customers should the business intervene with first?**

Rather than treating churn prediction as a standalone classification problem, Retain-AI connects prediction to business decision-making.

The platform moves through:

**Customer Data → Feature Engineering → Churn Prediction → Probability Calibration → SHAP Explainability → Revenue at Risk → Intervention Economics → Retention Actions → Executive Dashboard**

---

## 🚀 Live Application

**Streamlit App:**  
[Open Retain-AI](https://retain-ai.streamlit.app/)

**GitHub Repository:**  
[Retain-AI](https://github.com/Is-not-avaible-was-available/Retain-AI)

---

## 🎯 What Problem Does Retain-AI Solve?

Traditional churn dashboards often stop at:

> "This customer has a 72% probability of churning."

That is useful, but it does not answer the business questions that follow:

- How much revenue is at risk?
- Why is the customer at risk?
- Is intervention economically worthwhile?
- Which customers should the retention team contact first?
- What kind of intervention should be used?
- How should executives understand the overall retention exposure?

Retain-AI addresses these questions by combining predictive modeling with a business decision engine.

---

# 🧠 Platform Architecture

```text
                    CUSTOMER DATA
                         │
                         ▼
              ┌─────────────────────┐
              │ Data Intelligence   │
              │ Validation /        │
              │ Feature Engineering │
              └──────────┬──────────┘
                         │
                         ▼
                ┌────────────────┐
                │ Feature Store  │
                └───────┬────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │ Predictive Layer    │
              │ Calibrated XGBoost  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Explainability      │
              │ SHAP                │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Business Decision   │
              │ Engine              │
              │                     │
              │ Revenue at Risk     │
              │ Expected Save Value │
              │ Intervention Cost   │
              │ Net Expected Value  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ LLM Communication   │
              │ Layer               │
              │                     │
              │ Qwen3:4B            │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Streamlit Decision  │
              │ Dashboard            │
              └─────────────────────┘


📊 Dashboard

Retain-AI provides five major decision-oriented views.

1. Executive Overview

Provides a portfolio-level view of:

Customers at risk
Revenue at risk
Expected save value
Intervention economics
Risk distribution
Portfolio trends
Segment-level exposure

Designed for executives and retention leaders.

2. Customer Risk Explorer

Allows users to:

Identify high-risk customers
Filter by segment and risk tier
Examine churn probability
Compare revenue exposure
Prioritize customers for intervention

The ranking is driven by business impact rather than churn probability alone.

3. Customer 360

Provides an individual customer view containing:

Customer profile
Contract information
Health score
Health trajectory
Usage behaviour
Support signals
Churn probability
Revenue at risk
Risk tier
Renewal urgency
SHAP-based explanations
Recommended retention action
4. Intervention Planner

Connects model predictions to economics.

For each customer:

Revenue at Risk
       ↓
Expected Save Value
       ↓
Intervention Cost
       ↓
Net Expected Value
       ↓
Priority Rank

This allows retention teams to focus their limited intervention capacity on customers where intervention is expected to create the greatest economic value.

5. Model & Governance

Provides visibility into:

Model performance
Validation methodology
Temporal train/validation/test split
Calibration
Precision / Recall
ROC-AUC
PR-AUC
Model comparison
Feature importance
SHAP explainability
Business assumptions
🤖 Machine Learning

Three model families were evaluated:

Model	ROC-AUC	PR-AUC	Brier Score
Logistic Regression	0.7943	0.2460	0.0464
Random Forest	0.7932	0.2352	0.0469
XGBoost	0.7699	0.1942	0.0808
Regularized XGBoost + Calibration	0.8016	0.2466	0.0467

The final production model is:

Regularized XGBoost + probability calibration

Logistic Regression is retained as the baseline model.

⏱️ Temporal Validation

Because churn prediction is a time-dependent business problem, Retain-AI uses a temporal validation strategy rather than a random train/test split.

The data is separated chronologically into:

TRAIN
2024-01-01 → 2024-10-28

PURGE WINDOW

VALIDATION
2025-02-03 → 2025-04-07

PURGE WINDOW

TEST
2025-07-14 → 2025-10-06

Purging prevents future churn-label information from leaking into model development.

The test set is kept frozen and is not used for model tuning.

🔍 Explainable AI

Retain-AI uses SHAP to explain why individual customers receive their risk scores.

The strongest global drivers include:

Health trend
Auto-renewal
Health score
Company size
Annual contract value
Total contract value
Product usage signals

An important insight from the model is that customer trajectory matters significantly.

A declining health trajectory can be more informative than a customer's current health score alone.

💰 Business Decision Engine

The platform does not prioritize customers using churn probability alone.

Instead, it calculates:

Revenue at Risk
Revenue at Risk =
Churn Probability × Annual Contract Value
Expected Save Value
Expected Save Value =
Revenue at Risk × Intervention Success Rate
Net Expected Value
Net Expected Value =
Expected Save Value − Intervention Cost

Customers are then prioritized by economic value.

The default business assumptions are:

Intervention Success Rate = 30%

Intervention Cost = ₹25,000

Default Intervention Capacity = 10%

At 10% intervention capacity, the decision engine selects the highest-value customers rather than simply selecting the customers with the highest churn probability.

📈 Example Portfolio Economics

Using the modeled customer portfolio:

Metric	Value
Customers evaluated	7,243
Total Revenue at Risk	₹1.216B
Expected Save Value	₹364.9M
Total Intervention Cost	₹181.1M
Net Expected Value	₹183.9M

At the default 10% intervention capacity:

Metric	Value
Customers selected	724
Revenue at Risk	₹760.1M
Expected Save Value	₹228.0M
Intervention Cost	₹18.1M
Net Expected Value	₹209.9M
Revenue-at-risk coverage	62.5%

These figures are generated from the project's synthetic dataset and business assumptions.

🧩 Risk Framework

Customers are classified into four churn-risk tiers:

Probability	Risk Tier
< 5%	Low
5–10%	Moderate
10–20%	High
≥ 20%	Critical

Revenue exposure is independently classified to avoid confusing:

Probability of churn

with

Financial impact of churn

This allows the platform to distinguish between a high-probability, low-value customer and a moderate-probability, high-value enterprise customer.

🤖 LLM Layer

Retain-AI uses Qwen3:4B as a communication and reasoning layer.

The LLM does not predict churn.

Instead:

ML Model
   ↓
Churn Probability
   ↓
SHAP Explanation
   ↓
Business Metrics
   ↓
Decision Engine
   ↓
Structured Context
   ↓
LLM
   ↓
Human-readable Insight

The LLM is responsible for translating structured analytical outputs into understandable business recommendations.

The authoritative churn probability, financial calculations, and intervention priority remain deterministic outputs from the ML and business decision layers.

🏗️ Project Structure
Retain-AI/
│
├── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│       └── features.parquet
│
├── models/
│   ├── xgb_calibrated.joblib
│   ├── xgb_preprocessor.joblib
│   ├── xgb_regularized.joblib
│   ├── business_config.joblib
│   └── model_metadata.json
│
├── src/
│   ├── customer_state.py
│   ├── usage.py
│   ├── support.py
│   ├── churn.py
│   ├── labels.py
│   ├── features.py
│   ├── inference.py
│   ├── decision_engine.py
│   ├── explainability.py
│   ├── llm_insights.py
│   └── executive_insights.py
│
├── configs/
│   └── split_config.json
│
├── notebooks/
│
├── requirements.txt
├── runtime.txt
├── .gitignore
└── README.md
🛠️ Technology Stack
Data & Engineering
Python
Pandas
NumPy
PyArrow
Parquet
Machine Learning
Scikit-learn
XGBoost
Joblib
Explainable AI
SHAP
Generative AI
Qwen3:4B
Ollama
Application
Streamlit
Altair
Deployment
GitHub
Streamlit Community Cloud
🔬 Key Engineering Decisions
Point-in-time feature construction

Features are constructed using only information available at the prediction snapshot.

Usage and support information is shifted to prevent future information from entering the model.

Temporal validation

The model is evaluated using chronological splits rather than random sampling.

Probability calibration

Raw class-weighted probabilities were overconfident, so calibration was introduced before using probabilities for business decisions.

Business-aware prioritization

The highest-risk customer is not necessarily the highest-priority customer.

Priority depends on:

Churn Probability
×
Annual Contract Value

followed by expected intervention economics.

LLM separation

The LLM is intentionally separated from prediction and financial calculations to reduce hallucination risk and preserve analytical control.

📦 Running Locally

Clone the repository:

git clone https://github.com/Is-not-avaible-was-available/Retain-AI.git
cd Retain-AI

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Run Streamlit:

streamlit run app.py

The application will open at:

http://localhost:8501
🧪 Dataset

Retain-AI uses a synthetic B2B SaaS customer dataset designed to simulate:

Customer accounts
Contracts
Subscription/product adoption
Weekly product usage
Customer health
Support interactions
Renewal behaviour
Churn events

The synthetic nature of the dataset makes the project suitable for demonstrating the complete architecture without exposing real customer information.

⚠️ Disclaimer

This project is a portfolio / demonstration system using synthetic data.

The financial values, intervention assumptions, churn outcomes, and business recommendations should not be interpreted as predictions about any real company or customer portfolio.

👤 Author

Rajat Yadav

Built as an end-to-end exploration of how machine learning and generative AI can move beyond prediction into business decision intelligence.

⭐ If you find this project useful

Feel free to explore the repository, try the Streamlit application, and connect with me on LinkedIn.
