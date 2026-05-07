import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.ml_engine import get_churn_models, predict_all, build_hypotheses, SEGMENT_COLORS

CONFIDENCE_COLOR = {"High": "#10b981", "Medium": "#f59e0b", "Low": "#ef4444"}


def show():
    st.markdown("## 💡 Churn Driver Hypotheses")
    st.markdown(
        "SHAP values across all segment models are aggregated to surface the **top-3 testable churn drivers** — "
        "each framed as a falsifiable hypothesis with a suggested experiment."
    )

    models = get_churn_models()
    df     = predict_all()

    hypotheses = build_hypotheses(models)

    # ── Top-3 Hypothesis Cards
    st.markdown("### Top-3 Testable Hypotheses")
    for h in hypotheses:
        conf_color = CONFIDENCE_COLOR.get(h["confidence"], "#6b7280")
        st.markdown(f"""
        <div style="border:1px solid var(--color-border-tertiary);border-radius:14px;
                    padding:1.2rem 1.4rem;margin-bottom:14px;
                    background:var(--color-background-primary)">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
                <span style="background:#ede9fe;color:#4c1d95;font-size:12px;font-weight:700;
                             padding:3px 10px;border-radius:20px">H{h['rank']}</span>
                <strong style="font-size:15px">{h['title']}</strong>
                <span style="margin-left:auto;background:{conf_color}22;color:{conf_color};
                             font-size:11px;font-weight:600;padding:2px 10px;border-radius:20px">
                    {h['confidence']} confidence</span>
            </div>
            <p style="margin:4px 0 8px;color:var(--color-text-secondary);font-size:13px;line-height:1.6">
                📌 <strong>Statement:</strong> {h['statement']}
            </p>
            <p style="margin:4px 0 8px;color:var(--color-text-secondary);font-size:13px;line-height:1.6">
                🧪 <strong>Test:</strong> {h['test']}
            </p>
            <p style="margin:4px 0 0;font-size:12px;color:#6b7280">
                Driver: <code>{h['feature']}</code> &nbsp;·&nbsp; Direction: {h['direction']}
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── SHAP Feature Importance per segment
    st.markdown("### SHAP Feature Importance — Per Segment")
    seg_sel = st.selectbox("Select segment", list(models.keys()))
    _shap_bar(models[seg_sel], seg_sel)

    st.divider()

    # ── Hypothesis validation explorer
    st.markdown("### 🔬 Hypothesis Explorer")
    st.markdown("Interactively validate each hypothesis against the dataset.")

    h_sel = st.selectbox("Choose hypothesis",
                         [f"H{h['rank']}: {h['title']}" for h in hypotheses])
    rank  = int(h_sel[1]) - 1
    feat  = hypotheses[rank]["feature"]

    _hypothesis_explorer(df, feat, hypotheses[rank])


def _shap_bar(bundle, seg):
    import numpy as np
    sv  = bundle["shap_vals"]
    mean_abs = np.abs(sv).mean(axis=0)
    fi_df = pd.DataFrame({"Feature": bundle["features"], "MeanAbsSHAP": mean_abs})
    fi_df = fi_df.sort_values("MeanAbsSHAP", ascending=True).tail(15)

    fig = px.bar(fi_df, x="MeanAbsSHAP", y="Feature", orientation="h",
                 color="MeanAbsSHAP", color_continuous_scale="Purples",
                 template="plotly_white",
                 labels={"MeanAbsSHAP": "Mean |SHAP| Value"})
    fig.update_layout(height=420, margin=dict(t=10,b=10), showlegend=False,
                      title=f"Feature Importance — {seg} segment")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Model AUC for {seg}: **{bundle['auc']}** · Trained on {bundle['n_train']:,} users")


def _hypothesis_explorer(df, feat, h):
    if feat not in df.columns:
        st.info(f"Feature `{feat}` not directly in dataset for plotting.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"#### `{feat}` distribution by churn label")
        fig = px.histogram(df, x=feat, color=df["churned"].map({1:"Churned",0:"Retained"}),
                           color_discrete_map={"Churned":"#ef4444","Retained":"#10b981"},
                           barmode="overlay", opacity=0.7, nbins=40,
                           template="plotly_white",
                           labels={"color":"Label", feat: feat.replace("_"," ").title()})
        fig.update_layout(height=300, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"#### Churn rate across `{feat}` quartiles")
        df2 = df.copy()
        try:
            df2["quartile"] = pd.qcut(df2[feat], q=4,
                                      labels=["Q1 (low)","Q2","Q3","Q4 (high)"],
                                      duplicates="drop")
            qr = df2.groupby("quartile", observed=True)["churned"].mean().reset_index()
            qr.columns = ["Quartile","ChurnRate"]
            fig2 = px.bar(qr, x="Quartile", y="ChurnRate",
                          color="ChurnRate", color_continuous_scale="Reds",
                          template="plotly_white",
                          labels={"ChurnRate":"Churn Rate"})
            fig2.update_layout(height=300, margin=dict(t=10,b=10), showlegend=False)
            fig2.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig2, use_container_width=True)
        except Exception:
            st.info("Feature has too few unique values for quartile split.")

    # Summary stat box
    churned_mean  = df[df["churned"]==1][feat].mean()
    retained_mean = df[df["churned"]==0][feat].mean()
    ratio = churned_mean / retained_mean if retained_mean > 0 else 0

    st.markdown(f"""
    <div style="background:var(--color-background-secondary);border-radius:10px;
                padding:1rem 1.2rem;margin-top:8px;font-size:13px">
        <strong>Data evidence for H{h['rank']}:</strong><br>
        Avg <code>{feat}</code> for <span style="color:#ef4444">churned users</span>:
        <strong>{churned_mean:.2f}</strong><br>
        Avg <code>{feat}</code> for <span style="color:#10b981">retained users</span>:
        <strong>{retained_mean:.2f}</strong><br>
        Ratio: <strong>{ratio:.2f}×</strong> — {h['direction']}
    </div>
    """, unsafe_allow_html=True)
