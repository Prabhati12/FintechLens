#  FintechLens — Partner Analytics Engine
### Data Analytics & Platform Insights

---

## What It Does

FintechLens ingests an anonymised user-event dataset, segments fintech end-users by behaviour, and predicts churn risk per segment — surfacing the top-3 churn drivers as testable hypotheses.

| Module | Technique | Output |
|--------|-----------|--------|
| **Event Ingestion** | Synthetic 10K user event log (login, txn, support, feature) | Clean structured dataset |
| **Segmentation** | K-Means on RFM + behavioural features + PCA | 4 segments: Champions, Loyal, At-Risk, Dormant |
| **Churn Risk** | LightGBM per segment + Kaplan-Meier survival curves | Churn probability + expected days-to-churn |
| **Hypothesis Engine** | SHAP global + per-segment → testable hypotheses | Top-3 falsifiable churn drivers with A/B test design |
| **Partner Analytics** | Cross-partner comparison dashboard | Per-partner health scorecard |
| **Upload Engine** | Drop any CSV → instant predictions | Generalises to any partner's data |

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Models train automatically on first run (~30 seconds). Cached after that.

---

## Project Structure

```
fintechlens/
├── app.py                        # Entry point + routing
├── requirements.txt
├── .streamlit/config.toml        # Blue theme
├── data/                         # Auto-generated dataset
├── models/                       # Cached trained models
├── utils/
│   ├── data_gen.py               # Synthetic fintech event generator
│   └── ml_engine.py              # Segmentation, LightGBM, SHAP, hypotheses
└── pages/
    ├── overview.py               # KPI dashboard + heatmap
    ├── segmentation.py           # Cluster map, RFM profiles, product mix
    ├── churn_risk.py             # Per-segment risk, survival curves, user lookup
    ├── hypotheses.py             # SHAP + top-3 testable hypothesis cards
    ├── partner.py                # Cross-partner comparison
    └── upload.py                 # Upload any CSV
```

---

## Dataset Schema

The synthetic dataset simulates a real fintech event log:

| Feature | Description |
|---------|-------------|
| `user_id` | Anonymised user identifier |
| `partner_id` | One of 5 fintech partners |
| `signup_days` | Days since account creation |
| `kyc_status` | verified / pending / failed |
| `logins_30d` | Login count in last 30 days |
| `txn_count_30d` | Transaction count in last 30 days |
| `txn_volume_30d` | Transaction volume in last 30 days |
| `support_tickets_30d` | Support ticket count |
| `feature_clicks_30d` | In-app feature interaction count |
| `days_since_last_login` | Recency of login |
| `days_since_last_txn` | Recency of transaction (RFM Recency) |
| `product_count` | Number of products used (1–4) |
| `uses_payments/lending/investing/insurance` | Product mix flags |
| `lifecycle_stage` | Rule-based: Onboarding / Active / Champion / At-Risk / Dormant / Churned |
| `churned` | Binary churn label |
| `days_to_churn` | Used for survival analysis |

---

## ML Details

### Segmentation
- **Features:** recency, frequency, monetary, logins_30d, feature_clicks_30d, product_count, days_since_last_login, support_tickets_30d
- **Algorithm:** K-Means (k=4, n_init=15), auto-labelled by ascending churn rate
- **Visualisation:** PCA 2D scatter (colour = segment, shape = lifecycle)

### Churn Prediction (per segment)
- **Algorithm:** LightGBM with `scale_pos_weight` for class imbalance
- **Trained separately per segment** — At-Risk users churn for different reasons than Dormant ones
- **Survival:** Kaplan-Meier estimator per segment, median days-to-churn

### Hypothesis Engine
- SHAP TreeExplainer on each segment model
- Mean |SHAP| aggregated across segments → top-3 features
- Each feature mapped to a structured hypothesis with: statement, A/B test design, direction, confidence
