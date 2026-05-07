import pandas as pd
import numpy as np
import os

def generate_fintech_events(n_users=10000, seed=42):
    rng = np.random.default_rng(seed)

    # ── Partner IDs (simulate multiple fintech partners)
    partners = ["PAY_CORP", "LEND_NOW", "INVEST_X", "WALLET_GO", "INSURE_IT"]
    partner_weights = [0.30, 0.22, 0.18, 0.20, 0.10]

    user_ids       = [f"U{str(i).zfill(6)}" for i in range(n_users)]
    partner_id     = rng.choice(partners, n_users, p=partner_weights)
    signup_days    = rng.integers(30, 720, n_users)           # days since signup
    age_group      = rng.choice(["18-24","25-34","35-44","45-54","55+"], n_users,
                                p=[0.18, 0.32, 0.24, 0.16, 0.10])
    kyc_status     = rng.choice(["verified","pending","failed"], n_users, p=[0.72, 0.20, 0.08])

    # ── Product mix flags
    uses_payments  = rng.choice([1,0], n_users, p=[0.85, 0.15])
    uses_lending   = rng.choice([1,0], n_users, p=[0.38, 0.62])
    uses_investing = rng.choice([1,0], n_users, p=[0.28, 0.72])
    uses_insurance = rng.choice([1,0], n_users, p=[0.20, 0.80])
    product_count  = uses_payments + uses_lending + uses_investing + uses_insurance

    # ── Behavioural features (30-day window)
    logins_30d         = rng.integers(0, 45, n_users)
    txn_count_30d      = rng.integers(0, 60, n_users)
    txn_volume_30d     = rng.exponential(3500, n_users).round(2)
    support_tickets_30d= rng.integers(0, 6, n_users)
    feature_clicks_30d = rng.integers(0, 120, n_users)
    days_since_last_login = rng.integers(0, 90, n_users)
    days_since_last_txn   = rng.integers(0, 120, n_users)

    # ── RFM
    recency   = days_since_last_txn
    frequency = txn_count_30d
    monetary  = txn_volume_30d

    # ── Lifecycle stage (rule-based)
    def assign_lifecycle(row):
        if row["signup_days"] <= 30:               return "Onboarding"
        if row["days_since_last_login"] >= 60:     return "Churned"
        if row["days_since_last_login"] >= 30:     return "Dormant"
        if row["logins_30d"] <= 2:                 return "At-Risk"
        if row["logins_30d"] >= 15 and row["txn_count_30d"] >= 10: return "Champion"
        return "Active"

    df = pd.DataFrame({
        "user_id": user_ids,
        "partner_id": partner_id,
        "signup_days": signup_days,
        "age_group": age_group,
        "kyc_status": kyc_status,
        "uses_payments": uses_payments,
        "uses_lending": uses_lending,
        "uses_investing": uses_investing,
        "uses_insurance": uses_insurance,
        "product_count": product_count,
        "logins_30d": logins_30d,
        "txn_count_30d": txn_count_30d,
        "txn_volume_30d": monetary,
        "support_tickets_30d": support_tickets_30d,
        "feature_clicks_30d": feature_clicks_30d,
        "days_since_last_login": days_since_last_login,
        "days_since_last_txn": days_since_last_txn,
        "recency": recency,
        "frequency": frequency,
        "monetary": monetary,
    })

    df["lifecycle_stage"] = df.apply(assign_lifecycle, axis=1)

    # ── Churn label (realistic drivers)
    churn_prob = (
        0.04
        + 0.25 * (df["days_since_last_login"] > 30).astype(float)
        + 0.15 * (df["txn_count_30d"] == 0).astype(float)
        + 0.10 * (df["kyc_status"] == "failed").astype(float)
        + 0.08 * (df["product_count"] == 1).astype(float)
        + 0.07 * (df["support_tickets_30d"] >= 3).astype(float)
        + 0.06 * (df["logins_30d"] <= 1).astype(float)
        - 0.12 * (df["product_count"] >= 3).astype(float)
        - 0.10 * (df["logins_30d"] >= 15).astype(float)
        - 0.08 * (df["signup_days"] > 365).astype(float)
        + rng.normal(0, 0.05, n_users)
    ).clip(0.02, 0.97)

    df["churned"] = (rng.uniform(0, 1, n_users) < churn_prob).astype(int)

    # ── Days to churn (for survival analysis proxy)
    df["days_to_churn"] = np.where(
        df["churned"] == 1,
        rng.integers(1, 180, n_users),
        rng.integers(180, 720, n_users)
    )

    return df


def load_data(path="data/fintech_users.csv"):
    if os.path.exists(path):
        return pd.read_csv(path)
    os.makedirs("data", exist_ok=True)
    df = generate_fintech_events()
    df.to_csv(path, index=False)
    return df
