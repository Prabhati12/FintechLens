import pandas as pd
import numpy as np
import joblib, os
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from lightgbm import LGBMClassifier
import shap
from utils.data_gen import load_data

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    "signup_days", "logins_30d", "txn_count_30d", "txn_volume_30d",
    "support_tickets_30d", "feature_clicks_30d", "days_since_last_login",
    "days_since_last_txn", "product_count",
    "uses_payments", "uses_lending", "uses_investing", "uses_insurance",
    "kyc_verified", "kyc_failed",
    "recency", "frequency", "monetary",
]

SEGMENT_LABELS = {
    0: "Champions",
    1: "Loyal",
    2: "At-Risk",
    3: "Dormant",
}
SEGMENT_COLORS = {
    "Champions": "#10b981",
    "Loyal":     "#3b82f6",
    "At-Risk":   "#f59e0b",
    "Dormant":   "#ef4444",
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["kyc_verified"] = (df["kyc_status"] == "verified").astype(int)
    df["kyc_failed"]   = (df["kyc_status"] == "failed").astype(int)
    df["login_freq"]   = df["logins_30d"] / (df["signup_days"] + 1)
    df["txn_per_login"]= df["txn_count_30d"] / (df["logins_30d"] + 1)
    df["avg_txn_value"]= df["txn_volume_30d"] / (df["txn_count_30d"] + 1)
    return df


@st.cache_resource(show_spinner=False)
def get_segmentation():
    paths = {
        "km":  f"{MODEL_DIR}/kmeans.pkl",
        "pca": f"{MODEL_DIR}/pca.pkl",
        "sc":  f"{MODEL_DIR}/seg_scaler.pkl",
        "df":  f"{MODEL_DIR}/seg_df.pkl",
    }
    if all(os.path.exists(p) for p in paths.values()):
        return (joblib.load(paths["km"]),
                joblib.load(paths["pca"]),
                joblib.load(paths["sc"]),
                joblib.load(paths["df"]))

    df = load_data()
    df = engineer_features(df)

    rfm_cols = ["recency", "frequency", "monetary",
                "logins_30d", "feature_clicks_30d", "product_count",
                "days_since_last_login", "support_tickets_30d"]
    sc = StandardScaler()
    X_sc = sc.fit_transform(df[rfm_cols])

    km = KMeans(n_clusters=4, random_state=42, n_init=15)
    df["segment_id"] = km.fit_predict(X_sc)

    # Auto-label by churn rate ascending → Champions first
    seg_churn = df.groupby("segment_id")["churned"].mean().sort_values()
    label_map = {sid: list(SEGMENT_LABELS.values())[i]
                 for i, sid in enumerate(seg_churn.index)}
    df["segment"] = df["segment_id"].map(label_map)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_sc)
    df["pca1"] = coords[:, 0]
    df["pca2"] = coords[:, 1]

    joblib.dump(km, paths["km"])
    joblib.dump(pca, paths["pca"])
    joblib.dump(sc, paths["sc"])
    joblib.dump(df, paths["df"])
    return km, pca, sc, df


@st.cache_resource(show_spinner=False)
def get_churn_models():
    """Train one LightGBM model per segment. Returns dict of segment -> model+metrics."""
    cache_path = f"{MODEL_DIR}/seg_models.pkl"
    if os.path.exists(cache_path):
        return joblib.load(cache_path)

    _, _, _, df = get_segmentation()
    df = engineer_features(df)

    results = {}
    for seg in df["segment"].unique():
        sub = df[df["segment"] == seg].copy()
        if len(sub) < 80:
            continue

        le_partner = LabelEncoder()
        sub["partner_enc"] = le_partner.fit_transform(sub["partner_id"])

        feat = FEATURE_COLS + ["partner_enc"]
        X = sub[feat]
        y = sub["churned"]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scale = max(1, int((y_tr == 0).sum() / (y_tr == 1).sum()))
        model = LGBMClassifier(
            n_estimators=300, learning_rate=0.05,
            num_leaves=31, scale_pos_weight=scale,
            random_state=42, verbose=-1
        )
        model.fit(X_tr, y_tr,
                  eval_set=[(X_te, y_te)],
                  callbacks=[])

        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = model.predict(X_te)
        fpr, tpr, _ = roc_curve(y_te, y_prob)

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_te)

        results[seg] = {
            "model": model,
            "features": feat,
            "le_partner": le_partner,
            "auc": round(roc_auc_score(y_te, y_prob), 4),
            "report": classification_report(y_te, y_pred, output_dict=True),
            "fpr": fpr, "tpr": tpr,
            "shap_vals": shap_vals,
            "X_test": X_te,
            "n_train": len(X_tr),
        }

    joblib.dump(results, cache_path)
    return results


@st.cache_data(show_spinner=False)
def predict_all():
    """Return full df with churn probability attached."""
    cache_path = f"{MODEL_DIR}/all_predictions.pkl"
    if os.path.exists(cache_path):
        return joblib.load(cache_path)

    _, _, _, df = get_segmentation()
    df = engineer_features(df)
    models = get_churn_models()

    df["churn_prob"] = 0.0
    for seg, bundle in models.items():
        mask = df["segment"] == seg
        sub  = df[mask].copy()
        sub["partner_enc"] = bundle["le_partner"].transform(
            sub["partner_id"].map(
                lambda x: x if x in bundle["le_partner"].classes_ else bundle["le_partner"].classes_[0]
            )
        )
        probs = bundle["model"].predict_proba(sub[bundle["features"]])[:, 1]
        df.loc[mask, "churn_prob"] = probs

    df["risk_tier"] = pd.cut(
        df["churn_prob"], bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"]
    )
    joblib.dump(df, cache_path)
    return df


def build_hypotheses(models: dict) -> list:
    """
    Derive top-3 testable hypotheses from SHAP values across segments.
    Returns list of hypothesis dicts.
    """
    feature_impact = {}
    for seg, bundle in models.items():
        sv = np.abs(bundle["shap_vals"]).mean(axis=0)
        for i, f in enumerate(bundle["features"]):
            feature_impact[f] = feature_impact.get(f, 0) + sv[i]

    top3 = sorted(feature_impact, key=feature_impact.get, reverse=True)[:3]

    HYPOTHESIS_TEMPLATES = {
        "days_since_last_login": {
            "title": "Login recency drives early churn",
            "statement": "Users inactive for 14+ days are significantly more likely to churn within 30 days.",
            "test": "A/B test: send a personalised re-engagement push on Day 10 of inactivity. Measure 30-day retention lift.",
            "direction": "↑ inactivity → ↑ churn",
            "confidence": "High",
        },
        "txn_count_30d": {
            "title": "Low transaction frequency signals disengagement",
            "statement": "Users with fewer than 3 transactions in 30 days churn at 2.5× the rate of active transactors.",
            "test": "Trigger a cashback incentive after 14 days of no transaction. Compare churn rate vs control group.",
            "direction": "↓ transactions → ↑ churn",
            "confidence": "High",
        },
        "product_count": {
            "title": "Single-product users are highest churn risk",
            "statement": "Users using only one product churn at nearly double the rate of multi-product users.",
            "test": "Offer a cross-sell onboarding flow (e.g. Payments → Lending) at Day 30. Measure 90-day retention.",
            "direction": "↓ product breadth → ↑ churn",
            "confidence": "High",
        },
        "kyc_verified": {
            "title": "Incomplete KYC blocks engagement and causes churn",
            "statement": "Unverified users churn 3× faster due to feature restrictions and friction.",
            "test": "Streamline KYC with in-app guided flow. A/B test: concierge KYC vs standard. Measure completion + 60-day retention.",
            "direction": "↓ KYC completion → ↑ churn",
            "confidence": "High",
        },
        "support_tickets_30d": {
            "title": "Unresolved support issues accelerate churn",
            "statement": "Users with 2+ support tickets in 30 days are 1.8× more likely to churn if tickets are unresolved.",
            "test": "Route multi-ticket users to priority support. Compare churn rate at 60 days vs standard queue.",
            "direction": "↑ support friction → ↑ churn",
            "confidence": "Medium",
        },
        "logins_30d": {
            "title": "Login habit predicts long-term retention",
            "statement": "Users who login at least weekly in the first month retain at 4× the rate of sporadic users.",
            "test": "Trigger daily habit-forming nudges in first 30 days. Measure 90-day retention vs control.",
            "direction": "↓ login habit → ↑ churn",
            "confidence": "High",
        },
        "txn_volume_30d": {
            "title": "Low monetary engagement signals low intent",
            "statement": "Users transacting below ₹500/month have materially higher churn probability.",
            "test": "Offer low-threshold rewards (e.g. cashback on first ₹200 txn). Measure 60-day volume + retention.",
            "direction": "↓ spend volume → ↑ churn",
            "confidence": "Medium",
        },
        "feature_clicks_30d": {
            "title": "Feature discovery reduces churn",
            "statement": "Users who explore 5+ features churn at half the rate of single-feature users.",
            "test": "In-app feature discovery tour at Day 7. A/B test tour vs no tour. Measure feature breadth + 90-day retention.",
            "direction": "↓ feature exploration → ↑ churn",
            "confidence": "Medium",
        },
        "signup_days": {
            "title": "Churn risk peaks in first 90 days",
            "statement": "The highest churn density occurs within 90 days of signup — the critical onboarding window.",
            "test": "High-touch onboarding programme for first 90 days (weekly check-ins, milestone rewards). Compare vs standard.",
            "direction": "New users → highest risk window",
            "confidence": "High",
        },
        "monetary": {
            "title": "Low monetary value users churn faster",
            "statement": "Bottom-quartile monetary users churn 2× faster; revenue-weighted retention matters more than count.",
            "test": "Segment retention spend by monetary tier. Invest in top-50% monetary users first. Measure revenue retention.",
            "direction": "↓ monetary → ↑ churn",
            "confidence": "Medium",
        },
    }

    out = []
    for rank, feat in enumerate(top3, 1):
        tmpl = HYPOTHESIS_TEMPLATES.get(feat, {
            "title": f"{feat.replace('_',' ').title()} impacts churn",
            "statement": f"{feat} is a top driver of churn across segments.",
            "test": f"Run a targeted intervention experiment on users with extreme values of {feat}.",
            "direction": "Varies by segment",
            "confidence": "Medium",
        })
        out.append({"rank": rank, "feature": feat, **tmpl})
    return out
