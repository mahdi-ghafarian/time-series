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
    "Market Explorer": (
        repository_root / "financial-markets-explorer" / "app.py"
    ),
    "Mean Reversion Profile": repository_root / "mean-reversion-app" / "app.py",
    "Swing Statistics": repository_root / "swing-statistics-app" / "app.py",
}
app_prefixes = {
    "Market Explorer": "explorer_",
    "Mean Reversion Profile": "mean_reversion_",
    "Swing Statistics": "swing_",
}
app_search_keys = {
    app: f"{prefix}search_ticker"
    for app, prefix in app_prefixes.items()
}
app_search_label_keys = {
    app: f"{prefix}search_ticker_label"
    for app, prefix in app_prefixes.items()
}


def save_current_app_preferences() -> None:
    previous_app = st.session_state.get("_suite_current_app")
    if previous_app not in app_prefixes:
        return

    prefix = app_prefixes[previous_app]
    search_key = app_search_keys[previous_app]
    search_label_key = app_search_label_keys[previous_app]
    if search_key in st.session_state:
        search_value = st.session_state[search_key]
        selected_label = st.session_state.get(search_label_key, "")
        if selected_label and search_value == selected_label:
            search_value = ""
            st.session_state[search_key] = search_value
            st.session_state[search_label_key] = ""
        st.session_state["_suite_ticker_search"] = search_value
    if search_label_key in st.session_state:
        st.session_state["_suite_ticker_search_label"] = st.session_state[
            search_label_key
        ]

    preferences = dict(
        st.session_state.get("_suite_saved_preferences", {})
    )
    for key in st.session_state:
        if key.startswith(prefix):
            preferences[key] = st.session_state[key]
    st.session_state["_suite_saved_preferences"] = preferences

st.sidebar.title("Market Analysis Suite")
st.sidebar.markdown(
    "by [Mahdi Ghafarian](https://www.linkedin.com/in/mahdi-ghafarian)"
)
selected_app = st.sidebar.radio(
    "Choose an analysis",
    list(apps),
    key="_suite_selected_app",
    on_change=save_current_app_preferences,
)
preferences = st.session_state.get("_suite_saved_preferences", {})
selected_prefix = app_prefixes[selected_app]
for key, value in preferences.items():
    if key.startswith(selected_prefix):
        st.session_state[key] = value
search_key = app_search_keys[selected_app]
search_label_key = app_search_label_keys[selected_app]
if "_suite_ticker_search" in st.session_state:
    st.session_state[search_key] = st.session_state["_suite_ticker_search"]
if "_suite_ticker_search_label" in st.session_state:
    st.session_state[search_label_key] = st.session_state[
        "_suite_ticker_search_label"
    ]

st.session_state["_suite_current_app"] = selected_app
st.session_state.setdefault("_suite_shared_ticker", "BTC-USD")
st.session_state[f"{selected_prefix}ticker"] = st.session_state[
    "_suite_shared_ticker"
]
st.session_state["_market_analysis_suite"] = True
runpy.run_path(str(apps[selected_app]), run_name="__main__")
