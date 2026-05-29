"""
Streamlit Main Application Entry Point.

Multi-page app with sidebar navigation.
Run with: streamlit run app/main.py
"""

import sys
import os

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

from app.components.sidebar import render_sidebar
from app.pages import home, data_explorer, predict, analytics, training

# Page config
st.set_page_config(
    page_title="Equipment Success Predictor",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide default Streamlit menu/footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Render sidebar and get selected page
page = render_sidebar()

# Route to page
if page == "🏠 Home":
    home.render()
elif page == "📊 Data Explorer":
    data_explorer.render()
elif page == "🔮 Predict":
    predict.render()
elif page == "📈 Analytics":
    analytics.render()
elif page == "🧠 Training":
    training.render()
