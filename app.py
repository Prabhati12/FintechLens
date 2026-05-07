import streamlit as st

st.set_page_config(
    page_title="FintechLens",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(135deg, #1d4ed8 0%, #7c3aed 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    div[data-testid="stSidebarNav"] { display: none; }
    .stMetric { background: var(--color-background-secondary);
                border-radius: 10px; padding: 0.6rem 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar
st.sidebar.markdown("## 🏦 FintechLens")
st.sidebar.markdown("*Partner Analytics Engine*")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "👥 Segmentation", "🔍 Churn Risk",
     "💡 Hypotheses", "🏦 Partner Analytics", "📤 Upload Data"],
    label_visibility="collapsed"
)

st.sidebar.divider()
st.sidebar.markdown("**Users:** 10,000")
st.sidebar.markdown("**Partners:** 5")
st.sidebar.markdown("**Model:** LightGBM × 4 segments")
st.sidebar.markdown("**Clustering:** K-Means (RFM + behaviour)")

# ── Warm up models on first load
if "models_ready" not in st.session_state:
    with st.spinner("🔧 Training segment models on first run — takes ~30 seconds..."):
        from utils.ml_engine import get_segmentation, get_churn_models, predict_all
        get_segmentation()
        get_churn_models()
        predict_all()
    st.session_state["models_ready"] = True

# ── Route
if page == "🏠 Overview":
    from pages.overview import show; show()
elif page == "👥 Segmentation":
    from pages.segmentation import show; show()
elif page == "🔍 Churn Risk":
    from pages.churn_risk import show; show()
elif page == "💡 Hypotheses":
    from pages.hypotheses import show; show()
elif page == "🏦 Partner Analytics":
    from pages.partner import show; show()
elif page == "📤 Upload Data":
    from pages.upload import show; show()
