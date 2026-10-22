import streamlit as st


st.set_page_config(
    page_title="Market Analysis Suite",
    page_icon="📊",
    layout="wide",
)

st.title("Market Analysis Suite")
st.write(
    "Choose an analysis from the sidebar: Swing Statistics, Mean Reversion "
    "Profile, or Financial Markets Explorer."
)
