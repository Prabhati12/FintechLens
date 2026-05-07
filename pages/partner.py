import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.ml_engine import predict_all, SEGMENT_COLORS


def show():
    st.markdown("## 🏦 Partner Analytics")
    st.markdown("Compare churn risk, segment health, and revenue exposure across all fintech partner institutions.")

    df = predict_all()
    partners = sorted(df["partner_id"].unique())

    # Partner selector
    selected = st.multiselect("Filter Partners", partners, default=partners)
    df = df[df["partner_id"].isin(selected)]

    st.divider()

    # ── Partner KPI table
    st.markdown("#### Partner Health Scorecard")
    rows = []
    for p in selected:
        sub = df[df["partner_id"] == p]
        rows.append({
            "Partner":          p,
            "Users":            len(sub),
            "Churn Rate":       f"{sub['churned'].mean():.1%}",
            "High Risk Users":  (sub["risk_tier"] == "High").sum(),
            "Avg Products":     round(sub["product_count"].mean(), 2),
            "Avg Monthly Vol":  f"₹{sub['txn_volume_30d'].mean():,.0f}",
            "Revenue at Risk":  f"₹{sub[sub['risk_tier']=='High']['monetary'].sum()/1e3:,.0f}K",
            "Dominant Segment": sub["segment"].value_counts().idxmax(),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Churn Rate by Partner")
        cr = df.groupby("partner_id")["churned"].mean().reset_index()
        cr.columns = ["Partner", "ChurnRate"]
        cr = cr.sort_values("ChurnRate", ascending=True)
        fig = px.bar(cr, x="ChurnRate", y="Partner", orientation="h",
                     color="ChurnRate", color_continuous_scale="Reds",
                     template="plotly_white",
                     labels={"ChurnRate": "Churn Rate", "Partner": ""},
                     text=cr["ChurnRate"].apply(lambda x: f"{x:.1%}"))
        fig.update_layout(height=300, margin=dict(t=10,b=10), showlegend=False)
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Segment Mix by Partner")
        sm = df.groupby(["partner_id","segment"]).size().reset_index(name="Count")
        sm_pct = sm.copy()
        totals = sm_pct.groupby("partner_id")["Count"].transform("sum")
        sm_pct["Pct"] = sm_pct["Count"] / totals
        fig2 = px.bar(sm_pct, x="partner_id", y="Pct", color="segment",
                      color_discrete_map=SEGMENT_COLORS,
                      template="plotly_white", barmode="stack",
                      labels={"partner_id":"Partner","Pct":"Share","segment":"Segment"})
        fig2.update_layout(height=300, margin=dict(t=10,b=10))
        fig2.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Avg Churn Probability by Partner × Segment")
        hm = df.groupby(["partner_id","segment"])["churn_prob"].mean().reset_index()
        hm_pivot = hm.pivot(index="partner_id", columns="segment", values="churn_prob").fillna(0)
        fig3 = px.imshow(hm_pivot.round(2), text_auto=".0%",
                         color_continuous_scale="RdYlGn_r",
                         zmin=0, zmax=0.8, template="plotly_white", aspect="auto")
        fig3.update_layout(height=300, margin=dict(t=20,b=10))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("#### Revenue at Risk per Partner")
        rev = df[df["risk_tier"]=="High"].groupby("partner_id")["monetary"].sum().reset_index()
        rev.columns = ["Partner","RevenueAtRisk"]
        rev = rev.sort_values("RevenueAtRisk", ascending=True)
        fig4 = px.bar(rev, x="RevenueAtRisk", y="Partner", orientation="h",
                      color="RevenueAtRisk", color_continuous_scale="Oranges",
                      template="plotly_white",
                      labels={"RevenueAtRisk":"Revenue at Risk (₹)","Partner":""})
        fig4.update_layout(height=300, margin=dict(t=10,b=10), showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    # ── Drill-down
    st.divider()
    st.markdown("#### Partner Deep-Dive")
    partner_sel = st.selectbox("Select a partner to drill into", selected)
    sub = df[df["partner_id"] == partner_sel]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Users",    f"{len(sub):,}")
    c2.metric("Churn Rate",     f"{sub['churned'].mean():.1%}")
    c3.metric("High-Risk",      f"{(sub['risk_tier']=='High').sum():,}")
    c4.metric("Avg Products",   f"{sub['product_count'].mean():.1f}")

    col5, col6 = st.columns(2)
    with col5:
        lc = sub["lifecycle_stage"].value_counts().reset_index()
        lc.columns = ["Stage","Count"]
        stage_colors = {"Onboarding":"#a78bfa","Active":"#10b981","Champion":"#3b82f6",
                        "At-Risk":"#f59e0b","Dormant":"#ef4444","Churned":"#6b7280"}
        fig5 = px.pie(lc, names="Stage", values="Count", hole=0.45,
                      color="Stage", color_discrete_map=stage_colors,
                      template="plotly_white", title="Lifecycle Distribution")
        fig5.update_layout(height=280, margin=dict(t=40,b=10))
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        fig6 = px.histogram(sub, x="churn_prob", nbins=30, color="segment",
                            color_discrete_map=SEGMENT_COLORS,
                            template="plotly_white", opacity=0.75,
                            barmode="overlay",
                            title="Churn Probability Distribution",
                            labels={"churn_prob":"Churn Probability"})
        fig6.update_layout(height=280, margin=dict(t=40,b=10))
        st.plotly_chart(fig6, use_container_width=True)

    csv = sub.to_csv(index=False).encode()
    st.download_button(f"⬇ Download {partner_sel} Data", csv,
                       f"{partner_sel}_users.csv", "text/csv")
