import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.ml_engine import get_segmentation, predict_all, SEGMENT_COLORS

SEGMENT_DESC = {
    "Champions": "High frequency, high monetary, low recency gap. Your best users — protect and reward.",
    "Loyal":     "Consistent engagement, moderate spend. Upsell opportunities and referral potential.",
    "At-Risk":   "Dropping login frequency and transaction count. Proactive intervention needed now.",
    "Dormant":   "Low engagement, high recency gap. Hard to recover — focus on high-value subset.",
}


def show():
    st.markdown("## 👥 User Segmentation")
    st.markdown("K-Means clustering on RFM + behavioural features. Four actionable segments with distinct churn profiles.")

    km, pca, sc, df_seg = get_segmentation()
    df = predict_all()

    # Segment cards
    cols = st.columns(4)
    for i, seg in enumerate(["Champions", "Loyal", "At-Risk", "Dormant"]):
        sub  = df[df["segment"] == seg]
        count= len(sub)
        pct  = count / len(df) * 100
        cr   = sub["churned"].mean()
        color= SEGMENT_COLORS[seg]
        with cols[i]:
            st.markdown(f"""
            <div style="border-left:4px solid {color};padding:10px 14px;
                        background:var(--color-background-secondary);
                        border-radius:0 10px 10px 0;margin-bottom:8px">
                <strong style="color:{color}">{seg}</strong><br>
                <span style="font-size:1.4rem;font-weight:700">{count:,}</span>
                <span style="color:#6b7280;font-size:0.8rem"> ({pct:.0f}%)</span><br>
                <small style="color:#6b7280">Churn rate: <b>{cr:.0%}</b></small>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    tab1, tab2, tab3 = st.tabs(["🗺 Cluster Map", "📊 RFM Profiles", "🧩 Product Mix"])

    with tab1:
        _cluster_map(df)
    with tab2:
        _rfm_profiles(df)
    with tab3:
        _product_mix(df)


def _cluster_map(df):
    st.markdown("#### PCA Cluster Map — RFM + Behavioural Space")
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig = px.scatter(sample, x="pca1", y="pca2", color="segment",
                     color_discrete_map=SEGMENT_COLORS,
                     opacity=0.55, template="plotly_white",
                     hover_data=["user_id", "partner_id", "churn_prob"],
                     labels={"pca1": "Principal Component 1", "pca2": "Principal Component 2"},
                     symbol="lifecycle_stage")
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(height=500, margin=dict(t=20, b=10), legend_title_text="Segment")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Shape = lifecycle stage · Colour = ML segment · Axes = PCA projection of 8-feature RFM space")


def _rfm_profiles(df):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Avg Recency by Segment (lower = better)")
        rc = df.groupby("segment")["recency"].mean().reset_index()
        rc.columns = ["Segment", "AvgRecency"]
        fig = px.bar(rc, x="Segment", y="AvgRecency", color="Segment",
                     color_discrete_map=SEGMENT_COLORS, template="plotly_white",
                     labels={"AvgRecency": "Avg Days Since Last Txn"})
        fig.update_layout(height=280, margin=dict(t=10,b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Avg Monetary by Segment")
        mn = df.groupby("segment")["monetary"].mean().reset_index()
        mn.columns = ["Segment", "AvgMonetary"]
        fig2 = px.bar(mn, x="Segment", y="AvgMonetary", color="Segment",
                      color_discrete_map=SEGMENT_COLORS, template="plotly_white",
                      labels={"AvgMonetary": "Avg 30d Volume (₹)"})
        fig2.update_layout(height=280, margin=dict(t=10,b=10), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Login Frequency Distribution")
        fig3 = px.box(df, x="segment", y="logins_30d", color="segment",
                      color_discrete_map=SEGMENT_COLORS, template="plotly_white",
                      labels={"logins_30d": "Logins (30d)", "segment": "Segment"})
        fig3.update_layout(height=280, margin=dict(t=10,b=10), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("#### Transaction Count Distribution")
        fig4 = px.violin(df, x="segment", y="txn_count_30d", color="segment",
                         color_discrete_map=SEGMENT_COLORS, template="plotly_white",
                         box=True, labels={"txn_count_30d": "Txn Count (30d)"})
        fig4.update_layout(height=280, margin=dict(t=10,b=10), showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("#### Segment Descriptions")
    for seg, desc in SEGMENT_DESC.items():
        color = SEGMENT_COLORS[seg]
        st.markdown(f"""
        <div style="border-left:3px solid {color};padding:6px 14px;
                    margin-bottom:8px;background:var(--color-background-secondary);
                    border-radius:0 8px 8px 0;font-size:13px">
            <strong style="color:{color}">{seg}</strong> — {desc}
        </div>
        """, unsafe_allow_html=True)


def _product_mix(df):
    st.markdown("#### Product Adoption by Segment")
    products = ["uses_payments", "uses_lending", "uses_investing", "uses_insurance"]
    prod_labels = ["Payments", "Lending", "Investing", "Insurance"]

    rows = []
    for seg in ["Champions", "Loyal", "At-Risk", "Dormant"]:
        sub = df[df["segment"] == seg]
        for p, label in zip(products, prod_labels):
            rows.append({"Segment": seg, "Product": label,
                         "AdoptionRate": sub[p].mean()})
    prod_df = pd.DataFrame(rows)

    fig = px.bar(prod_df, x="Product", y="AdoptionRate", color="Segment",
                 color_discrete_map=SEGMENT_COLORS, barmode="group",
                 template="plotly_white", labels={"AdoptionRate": "Adoption Rate"})
    fig.update_layout(height=350, margin=dict(t=10,b=10))
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Avg Products Used vs Churn Rate")
    pc = df.groupby("product_count").agg(
        churn_rate=("churned", "mean"),
        users=("user_id", "count")
    ).reset_index()
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=pc["product_count"], y=pc["users"],
                          name="Users", marker_color="#c4b5fd",
                          yaxis="y"))
    fig2.add_trace(go.Scatter(x=pc["product_count"], y=pc["churn_rate"],
                               name="Churn Rate", yaxis="y2",
                               mode="lines+markers",
                               line=dict(color="#ef4444", width=2.5),
                               marker=dict(size=8)))
    fig2.update_layout(
        template="plotly_white", height=320,
        margin=dict(t=20, b=10),
        yaxis=dict(title="User Count"),
        yaxis2=dict(overlaying="y", side="right",
                    tickformat=".0%", title="Churn Rate"),
        legend=dict(x=0.7, y=1)
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Multi-product users churn at a fraction of single-product users — cross-sell is the strongest retention lever.")
