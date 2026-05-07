import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.ml_engine import get_churn_models, SEGMENT_COLORS
from utils.data_gen import generate_fintech_events

REQUIRED = [
    "user_id","partner_id","signup_days","kyc_status","logins_30d",
    "txn_count_30d","txn_volume_30d","support_tickets_30d","feature_clicks_30d",
    "days_since_last_login","days_since_last_txn","product_count",
    "uses_payments","uses_lending","uses_investing","uses_insurance",
]


def show():
    st.markdown("## 📤 Upload Partner Data")
    st.markdown("Drop in any anonymised user-event CSV matching the schema below — predictions run instantly.")

    # Template download
    template = generate_fintech_events(n_users=5, seed=1)[REQUIRED]
    st.download_button("⬇ Download Template CSV", template.to_csv(index=False).encode(),
                       "template.csv", "text/csv")

    with st.expander("Required columns"):
        st.markdown("\n".join([f"- `{c}`" for c in REQUIRED]))

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        return

    try:
        df = pd.read_csv(uploaded)
        st.success(f"✅ Loaded **{len(df):,}** users.")
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return

    # Feature engineering inline
    df["kyc_verified"] = (df["kyc_status"] == "verified").astype(int)
    df["kyc_failed"]   = (df["kyc_status"] == "failed").astype(int)
    df["recency"]      = df["days_since_last_txn"]
    df["frequency"]    = df["txn_count_30d"]
    df["monetary"]     = df["txn_volume_30d"]

    # Use the At-Risk model as default (most cautious)
    models = get_churn_models()
    seg_used = "At-Risk" if "At-Risk" in models else list(models.keys())[0]
    bundle   = models[seg_used]

    try:
        df["partner_enc"] = df["partner_id"].map(
            lambda x: bundle["le_partner"].transform([x])[0]
            if x in bundle["le_partner"].classes_
            else 0
        )
        feats = [f for f in bundle["features"] if f in df.columns]
        for f in bundle["features"]:
            if f not in df.columns:
                df[f] = 0
        probs = bundle["model"].predict_proba(df[bundle["features"]])[:, 1]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return

    df["churn_prob"] = probs
    df["risk_tier"]  = pd.cut(probs, bins=[0,0.3,0.6,1.0],
                               labels=["Low","Medium","High"])

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Users",   f"{len(df):,}")
    c2.metric("High Risk",     f"{(df['risk_tier']=='High').sum():,}")
    c3.metric("Avg Churn Prob",f"{probs.mean():.1%}")
    c4.metric("Model Used",    seg_used)

    col1, col2 = st.columns(2)
    with col1:
        rt = df["risk_tier"].value_counts().reset_index()
        rt.columns = ["Risk","Count"]
        fig = px.pie(rt, names="Risk", values="Count", hole=0.5,
                     color="Risk",
                     color_discrete_map={"High":"#ef4444","Medium":"#f59e0b","Low":"#10b981"},
                     template="plotly_white", title="Risk Distribution")
        fig.update_layout(height=280, margin=dict(t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.histogram(df, x="churn_prob", nbins=30,
                            color="risk_tier",
                            color_discrete_map={"High":"#ef4444","Medium":"#f59e0b","Low":"#10b981"},
                            template="plotly_white", opacity=0.8,
                            title="Churn Probability Distribution",
                            labels={"churn_prob":"Churn Probability"})
        fig2.update_layout(height=280, margin=dict(t=40,b=10))
        st.plotly_chart(fig2, use_container_width=True)

    display = df[["user_id","partner_id","churn_prob","risk_tier",
                  "logins_30d","txn_count_30d","product_count"]].copy()
    display["churn_prob"] = display["churn_prob"].apply(lambda x: f"{x:.1%}")
    display = display.sort_values("churn_prob", ascending=False)
    st.dataframe(display, use_container_width=True, height=380, hide_index=True)

    csv = df.to_csv(index=False).encode()
    st.download_button("⬇ Download Full Results", csv, "upload_predictions.csv", "text/csv")
