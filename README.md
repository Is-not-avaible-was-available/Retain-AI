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
