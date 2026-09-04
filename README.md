Retain-AI

AI-Powered Customer Retention Decision Platform.

Retain-AI is an end-to-end customer retention intelligence platform that moves beyond churn prediction. It estimates 90-day churn risk, explains why an account is at risk, quantifies financial exposure, prioritises limited retention capacity, and generates grounded executive and customer-level AI briefs.

Product promise

Identify risk → quantify exposure → explain the drivers → prioritise intervention → communicate the decision.

Architecture

Raw customer / contract / usage / support data
                    │
                    ▼
             Feature Store
                    │
                    ▼
        Calibrated XGBoost Model
                    │
            90-day churn probability
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
        SHAP             Decision Engine
     Why risk?        What should we do?
          │                   │
          └─────────┬─────────┘
                    ▼
          Revenue at Risk / EV
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
      Customer 360       Executive AI
          │                   │
          └─────────┬─────────┘
                    ▼
                 Qwen3:4B
                    │
                    ▼
               Streamlit UI

The LLM is a communication layer. It does not generate the churn probability, replace SHAP, or override the Decision Engine.

Application modules

1. Executive Overview

Portfolio-level KPIs, risk concentration, trends, top retention priorities and default intervention economics.

2. Customer Risk Explorer

Filter and rank customers by risk tier, financial exposure, behaviour, segment and renewal urgency.

3. Intervention Planner

Configure intervention capacity and review expected benefit/cost, risk coverage, recommended actions and the selected intervention portfolio.

4. Customer 360

Customer-level context, health/usage/support/renewal signals, Decision Engine recommendation, SHAP explanation and AI Retention Insight.

5. Executive AI

Qwen3:4B synthesises portfolio analytics into an executive brief covering portfolio posture, key findings, leadership focus, recommended actions and watchouts.

6. Model & Governance

Model performance, frozen-test results, operating-point metrics, assumptions, governance guardrails, architecture and deployment readiness.

Model

Production champion: calibrated regularized XGBoost

Prediction horizon: 90 days

Calibration: sigmoid calibration with 5-fold cross-validation on the training period

Baseline: calibrated Logistic Regression

Frozen test performance

Metric

Result

ROC-AUC

0.8082

PR-AUC

0.2245

Brier Score

0.0468

Actual positive rate

5.49%

10% intervention operating point

Metric

Result

Customers targeted

9,035

Precision

23.88%

Recall

43.48%

Lift

4.35×

The test set is frozen and is not used for tuning. The 10% capacity is an operating point selected from validation analysis and remains configurable in the Intervention Planner.

Business decision logic

Retain-AI deliberately avoids an arbitrary weighted risk score.

Churn probability
        ↓
Revenue at Risk = P(churn) × Annual Contract Value
        ↓
Expected Save Value = Revenue at Risk × intervention success assumption
        ↓
Net Expected Value = Expected Save Value − intervention cost
        ↓
Rank economically viable accounts
        ↓
Select within available intervention capacity

Current scenario assumptions

Assumption

Value

Intervention success rate

30%

Intervention cost

₹25,000 / account

Default intervention capacity

10%

These are explicit scenario assumptions, not learned causal treatment effects. Once real intervention outcomes are collected, they should be replaced or calibrated using observed treatment/control evidence.

AI governance

Prediction authority: calibrated XGBoost.

Explanation authority: SHAP explains model behaviour; it is not treated as causal evidence.

Business authority: Decision Engine controls risk tier, economic prioritisation and recommended action.

LLM authority: Qwen3:4B communicates governed signals and cannot override upstream decisions.

Missing data: missing support observations are not interpreted as negative customer experience.

Temporal integrity: training, validation and test periods are separated with purge windows.

Test integrity: the final test set is frozen.

Human oversight: retention recommendations are decision support for accountable business teams, not autonomous customer actions.

Project structure

Retain-AI/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   └── processed/
│       └── features.parquet
├── models/
│   ├── xgb_calibrated.joblib
│   ├── xgb_regularized.joblib
│   ├── xgb_preprocessor.joblib
│   ├── business_config.joblib
│   └── model_metadata.json
├── src/
│   ├── inference.py
│   ├── decision_engine.py
│   ├── explainability.py
│   ├── llm_insights.py
│   └── executive_insights.py
└── notebooks/
    └── modelling / validation work

Local setup

1. Create the environment

Windows PowerShell:

python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

2. Install Ollama

Install Ollama locally and pull the configured model:

ollama pull qwen3:4b
ollama list

Keep Ollama running before using Customer 360 AI or Executive AI.

3. Run the application

streamlit run app.py

Reproducibility

The repository expects the frozen feature store and persisted model artifacts to be present. The production application loads the persisted calibrated model and preprocessor rather than retraining at startup.

The final test set is used only for final evaluation. Threshold and capacity decisions are based on validation analysis.

Deployment readiness

Ready

Persisted production model artifacts

Isolated inference module

Isolated Decision Engine

SHAP explainability layer

Local Qwen3:4B integration

Streamlit dashboard

Caching and startup error handling

Model/governance documentation

Business assumptions documented

Next after live deployment

Once real retention interventions and outcomes are available, add:

prediction drift monitoring

calibration monitoring

intervention acceptance / treatment tracking

realised retention lift

realised revenue saved

intervention-cost tracking

periodic model retraining and champion/challenger evaluation

This distinction is intentional: production monitoring of intervention effectiveness requires real post-decision outcome data that a synthetic development dataset cannot provide.

Portfolio positioning

Retain-AI is designed to demonstrate the complete path from machine learning to business decisioning:

Prediction is not the product. The product is the decision.

The platform combines predictive ML, explainability, financial prioritisation and governed generative AI into a single retention workflow.
