import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.ml_engine import get_churn_models, predict_all, SEGMENT_COLORS


def show():
    st.markdown("## 🔍 Churn Risk Analysis")
    st.markdown("Separate LightGBM model per segment — churn probability, survival curves, and per-user deep-dive.")

    tab1, tab2, tab3 = st.tabs(["📊 Risk Overview", "📈 Survival Curves", "🔎 User Lookup"])

    with tab1:
        _risk_overview()
    with tab2:
        _survival_curves()
    with tab3:
        _user_lookup()


def _risk_overview():
    df    = predict_all()
    models= get_churn_models()

    st.markdown("#### Per-Segment Model Performance")
    perf_rows = []
    for seg, bundle in models.items():
        rep = bundle["report"]
        perf_rows.append({
            "Segment":   seg,
            "Train Size": f"{bundle['n_train']:,}",
            "ROC-AUC":   bundle["auc"],
            "Precision": round(rep["1"]["precision"], 3),
            "Recall":    round(rep["1"]["recall"], 3),
            "F1":        round(rep["1"]["f1-score"], 3),
        })
    perf_df = pd.DataFrame(perf_rows).sort_values("ROC-AUC", ascending=False)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    st.markdown("#### Churn Probability Distribution by Segment")
    fig = px.violin(df, x="segment", y="churn_prob", color="segment",
                    color_discrete_map=SEGMENT_COLORS, box=True,
                    template="plotly_white",
                    labels={"churn_prob": "Churn Probability", "segment": "Segment"})
    fig.update_layout(height=350, margin=dict(t=10,b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ROC Curves — All Segments")
        fig2 = go.Figure()
        colors_line = ["#10b981","#3b82f6","#f59e0b","#ef4444"]
        for (seg, bundle), col in zip(models.items(), colors_line):
            fig2.add_trace(go.Scatter(
                x=bundle["fpr"], y=bundle["tpr"], mode="lines",
                name=f"{seg} (AUC={bundle['auc']})",
                line=dict(color=col, width=2)
            ))
        fig2.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                   line=dict(dash="dash",color="gray"), name="Random"))
        fig2.update_layout(template="plotly_white", height=350,
                           margin=dict(t=20,b=10),
                           xaxis_title="False Positive Rate",
                           yaxis_title="True Positive Rate")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("#### Risk Tier Distribution")
        rt = df.groupby(["segment","risk_tier"]).size().reset_index(name="Count")
        fig3 = px.bar(rt, x="segment", y="Count", color="risk_tier",
                      color_discrete_map={"High":"#ef4444","Medium":"#f59e0b","Low":"#10b981"},
                      barmode="stack", template="plotly_white",
                      labels={"segment":"Segment","risk_tier":"Risk Tier"})
        fig3.update_layout(height=350, margin=dict(t=20,b=10))
        st.plotly_chart(fig3, use_container_width=True)

    # High-risk table
    st.markdown("#### High-Risk Users (churn prob > 60%)")
    high = df[df["risk_tier"]=="High"].sort_values("churn_prob", ascending=False)
    display = high[["user_id","partner_id","segment","lifecycle_stage",
                    "churn_prob","logins_30d","txn_count_30d","product_count"]].head(200)
    display["churn_prob"] = display["churn_prob"].apply(lambda x: f"{x:.1%}")
    st.dataframe(display, use_container_width=True, height=350, hide_index=True)
    csv = high.to_csv(index=False).encode()
    st.download_button("⬇ Download High-Risk List", csv, "high_risk_users.csv", "text/csv")


def _survival_curves():
    st.markdown("#### Kaplan-Meier Style Survival Curves — Days-to-Churn per Segment")
    st.markdown("Estimates the probability of a user *surviving* (not churning) over time for each segment.")

    df = predict_all()

    fig = go.Figure()
    colors_line = {"Champions":"#10b981","Loyal":"#3b82f6","At-Risk":"#f59e0b","Dormant":"#ef4444"}

    for seg in ["Champions", "Loyal", "At-Risk", "Dormant"]:
        sub = df[df["segment"] == seg].copy()
        if len(sub) == 0:
            continue

        # KM estimator
        times  = np.sort(sub["days_to_churn"].values)
        events = (
    sub.groupby("days_to_churn")["churned"]
    .sum()
    .reindex(times, fill_value=0)
    .values
)

        n      = len(times)
        surv   = 1.0
        surv_probs = []
        t_vals     = []
        for i, (t, e) in enumerate(zip(times, events)):
            if e == 1:
                surv *= (1 - 1 / (n - i))
            surv_probs.append(surv)
            t_vals.append(t)

        # Downsample for plot
        step = max(1, len(t_vals) // 200)
        fig.add_trace(go.Scatter(
            x=t_vals[::step], y=surv_probs[::step],
            mode="lines", name=seg,
            line=dict(color=colors_line[seg], width=2.5)
        ))

    fig.update_layout(
        template="plotly_white", height=420,
        margin=dict(t=20, b=10),
        xaxis_title="Days Since Signup",
        yaxis_title="Survival Probability (not churned)",
        yaxis=dict(tickformat=".0%", range=[0, 1.05]),
        legend_title_text="Segment"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Steeper drop = faster churn decay. Dormant users reach 50% survival fastest — intervene in first 60 days.")

    # Median survival table
    st.markdown("#### Median Survival Time by Segment")
    rows = []
    for seg in ["Champions","Loyal","At-Risk","Dormant"]:
        sub = df[df["segment"]==seg]
        median_days = sub["days_to_churn"].median()
        churn_in_90 = (sub[sub["churned"]==1]["days_to_churn"] <= 90).mean()
        rows.append({
            "Segment": seg,
            "Median Days to Churn": int(median_days),
            "% Churned within 90 days": f"{churn_in_90:.0%}",
            "Avg Churn Prob": f"{sub['churn_prob'].mean():.1%}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _user_lookup():
    df = predict_all()
    st.markdown("### 🔎 Individual User Deep-Dive")

    uid = st.text_input("Enter User ID", placeholder="e.g. U000042")
    if not uid:
        sample_ids = df["user_id"].sample(5, random_state=7).tolist()
        st.markdown(f"**Try:** {', '.join(sample_ids)}")
        return

    row = df[df["user_id"] == uid]
    if row.empty:
        st.error(f"User '{uid}' not found.")
        return

    r        = row.iloc[0]
    prob     = r["churn_prob"]
    seg      = r["segment"]
    color    = SEGMENT_COLORS.get(seg, "#6b7280")
    risk_lbl = "🔴 HIGH RISK" if prob > 0.6 else "🟡 MEDIUM RISK" if prob > 0.3 else "🟢 LOW RISK"
    bg       = "#fef2f2" if prob > 0.6 else "#fffbeb" if prob > 0.3 else "#f0fdf4"

    st.markdown(f"""
    <div style="background:{bg};border-radius:12px;padding:1.2rem;margin-bottom:1rem;">
        <h3 style="margin:0">{uid} &nbsp; {risk_lbl}</h3>
        <h2 style="margin:0.3rem 0 0">Churn Probability: {prob:.1%}</h2>
        <p style="margin:4px 0 0;color:#6b7280">Segment: <strong style="color:{color}">{seg}</strong>
        &nbsp;·&nbsp; Lifecycle: <strong>{r['lifecycle_stage']}</strong>
        &nbsp;·&nbsp; Partner: <strong>{r['partner_id']}</strong></p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Behavioural Profile**")
        items = {
            "Logins (30d)":          r["logins_30d"],
            "Transactions (30d)":    r["txn_count_30d"],
            "Txn Volume (30d)":      f"₹{r['txn_volume_30d']:,.0f}",
            "Days since last login": r["days_since_last_login"],
            "Days since last txn":   r["days_since_last_txn"],
            "Support tickets":       r["support_tickets_30d"],
            "Feature clicks":        r["feature_clicks_30d"],
            "Products used":         int(r["product_count"]),
            "KYC Status":            r["kyc_status"],
            "Signup days ago":       r["signup_days"],
        }
        for k, v in items.items():
            st.markdown(f"- **{k}:** {v}")

    with col2:
        st.markdown("**Recommended Action**")
        action, reason, act_color = _get_action(prob, r)
        st.markdown(f"""
        <div style="background:{act_color};border-radius:10px;padding:1rem;margin-bottom:1rem">
            <strong>{action}</strong><br>
            <small style="color:#6b7280">{reason}</small>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Churn Risk Score"},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ef4444" if prob > 0.6 else "#f59e0b" if prob > 0.3 else "#10b981"},
                "steps": [
                    {"range": [0, 30],  "color": "#f0fdf4"},
                    {"range": [30, 60], "color": "#fffbeb"},
                    {"range": [60, 100],"color": "#fef2f2"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 60},
            }
        ))
        fig.update_layout(height=260, margin=dict(t=30, b=0, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)


def _get_action(prob, r):
    if prob > 0.70:
        return ("🚨 Urgent Human Outreach",
                "Assign relationship manager. Offer personalised retention package within 48h.",
                "#fef2f2")
    elif prob > 0.50:
        return ("💳 Targeted Incentive Offer",
                "Send personalised discount or cashback reward to re-engage.",
                "#fffbeb")
    elif prob > 0.30:
        return ("📧 Proactive Engagement Campaign",
                "Feature discovery email + in-app nudge to explore unused products.",
                "#eff6ff")
    else:
        return ("✅ Maintain & Reward",
                "Include in loyalty programme. No intervention needed.",
                "#f0fdf4")
