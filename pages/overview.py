import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils.ml_engine import predict_all, SEGMENT_COLORS


def show():
    st.markdown("## 🏠 Platform Overview")
    st.markdown("10,000 fintech end-users across 5 partner institutions — behaviour, churn risk, and segment health at a glance.")

    df = predict_all()

    churn_rate  = df["churned"].mean()
    at_risk     = (df["risk_tier"] == "High").sum()
    revenue_risk= df[df["risk_tier"] == "High"]["monetary"].sum()
    avg_products= df["product_count"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Users",        f"{len(df):,}")
    c2.metric("Churn Rate",         f"{churn_rate:.1%}", delta_color="inverse", delta=f"{churn_rate:.1%}")
    c3.metric("High-Risk Users",    f"{at_risk:,}",  delta_color="inverse", delta="need action")
    c4.metric("Revenue at Risk",    f"₹{revenue_risk/1e6:.1f}M")
    c5.metric("Avg Products / User",f"{avg_products:.1f}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Segment Distribution")
        sc = df["segment"].value_counts().reset_index()
        sc.columns = ["Segment", "Count"]
        fig = px.pie(sc, names="Segment", values="Count", hole=0.5,
                     color="Segment", color_discrete_map=SEGMENT_COLORS,
                     template="plotly_white")
        fig.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Churn Rate by Segment")
        cr = df.groupby("segment")["churned"].mean().reset_index()
        cr.columns = ["Segment", "ChurnRate"]
        cr = cr.sort_values("ChurnRate", ascending=True)
        fig2 = px.bar(cr, x="ChurnRate", y="Segment", orientation="h",
                      color="Segment", color_discrete_map=SEGMENT_COLORS,
                      template="plotly_white", text=cr["ChurnRate"].apply(lambda x: f"{x:.0%}"))
        fig2.update_layout(height=300, margin=dict(t=10, b=10), showlegend=False)
        fig2.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Users by Partner & Risk Tier")
        pt = df.groupby(["partner_id", "risk_tier"]).size().reset_index(name="Count")
        fig3 = px.bar(pt, x="partner_id", y="Count", color="risk_tier",
                      color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
                      barmode="stack", template="plotly_white",
                      labels={"partner_id": "Partner", "risk_tier": "Risk"})
        fig3.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("#### Lifecycle Stage Breakdown")
        lc = df["lifecycle_stage"].value_counts().reset_index()
        lc.columns = ["Stage", "Count"]
        stage_colors = {"Onboarding": "#a78bfa", "Active": "#10b981", "Champion": "#3b82f6",
                        "At-Risk": "#f59e0b", "Dormant": "#ef4444", "Churned": "#6b7280"}
        fig4 = px.bar(lc, x="Stage", y="Count", color="Stage",
                      color_discrete_map=stage_colors, template="plotly_white")
        fig4.update_layout(height=300, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    st.markdown("#### Segment × Lifecycle Heatmap")
    hm = df.groupby(["segment", "lifecycle_stage"]).size().reset_index(name="Count")
    hm_pivot = hm.pivot(index="segment", columns="lifecycle_stage", values="Count").fillna(0)
    fig5 = px.imshow(hm_pivot, text_auto=True, color_continuous_scale="Blues",
                     template="plotly_white", aspect="auto")
    fig5.update_layout(height=300, margin=dict(t=20, b=10))
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("Rows = ML segments  ·  Columns = rule-based lifecycle stage  ·  Values = user count")
