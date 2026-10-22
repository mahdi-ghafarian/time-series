from pathlib import Path
import runpy

import streamlit as st


st.set_page_config(
    page_title="Market Analysis Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

repository_root = Path(__file__).resolve().parent.parent
apps = {
    "Swing Statistics": repository_root / "swing-statistics-app" / "app.py",
    "Mean Reversion Profile": repository_root / "mean-reversion-app" / "app.py",
    "Financial Markets Explorer": (
        repository_root / "financial-markets-explorer" / "app.py"
    ),
}

st.sidebar.title("Market Analysis Suite")
selected_app = st.sidebar.radio("Choose an analysis", list(apps))
st.session_state["_market_analysis_suite"] = True
runpy.run_path(str(apps[selected_app]), run_name="__main__")
