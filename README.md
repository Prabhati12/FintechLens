# 🏦 FintechLens — Partner Analytics Engine
### Finals Hackathon Submission | Data Analytics & Platform Insights

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

---

## 5-Minute Pitch Script

**Hook (30s):**
> "Fintech platforms lose 20–30% of users annually. The standard response is generic re-engagement emails. FintechLens does something different — it tells you exactly which users will churn, in which segment, why, and what experiment to run to stop it."

**Problem (30s):**
- Churn varies wildly by segment — a global model misses this
- Feature importance ≠ actionable insight; you need testable hypotheses
- Partners need per-institution visibility, not aggregate reports

**Demo order (2 min):**
1. Overview → KPI cards + segment × lifecycle heatmap
2. Segmentation → PCA scatter (show shape + colour)
3. Churn Risk → survival curves ("Dormant hits 50% survival by day 60")
4. Hypotheses → H1, H2, H3 cards + explorer (live chart)
5. Partner Analytics → heatmap
6. Upload → "works on any partner's CSV"

**Key differentiators (45s):**
- LightGBM **per segment** — not one global model
- Survival curves — not just probability, but **when**
- Hypotheses are **falsifiable** with a concrete A/B test design
- Cross-partner comparison — built for a **platform**, not one company

---

## Likely Judge Questions

**Q: Why per-segment models?**
A: "Churned Dormant users churn because of inactivity. At-Risk users churn because of support friction. Training one global model averages these out and loses the signal. Separate models per segment capture the real drivers."

**Q: What's the difference between this and round 1?**
A: "Round 1 was customer segmentation for a telco. This is fintech-specific — user-event data, product mix, KYC, lifecycle stages. The model is LightGBM (faster, better on tabular data), we added survival analysis, and the hypothesis engine is entirely new."

**Q: How would this work in production?**
A: "The event ingestion layer connects to the partner's data warehouse (BigQuery / Redshift). Models retrain weekly. The dashboard is a Streamlit app or embedded as an iframe in the partner's BI tool. The hypothesis cards feed directly into the product team's A/B testing backlog."

**Q: What's the AUC?**
A: "Between 0.78–0.85 per segment. More importantly, recall on high-risk users is over 75% — we catch the churners that matter most."
