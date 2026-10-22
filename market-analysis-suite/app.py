from pathlib import Path

import streamlit as st


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

pages = {
    "Market analysis": [
        st.Page(
            REPOSITORY_ROOT / "swing-statistics-app" / "app.py",
            title="Swing Statistics",
            icon="📊",
        ),
        st.Page(
            REPOSITORY_ROOT / "mean-reversion-app" / "app.py",
            title="Mean Reversion Profile",
            icon="📈",
        ),
        st.Page(
            REPOSITORY_ROOT / "financial-markets-explorer" / "app.py",
            title="Financial Markets Explorer",
            icon="🔎",
        ),
    ]
}

page = st.navigation(pages, position="sidebar")
page.run()
