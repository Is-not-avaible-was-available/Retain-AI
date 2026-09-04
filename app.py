import streamlit as st

st.set_page_config(
    page_title="Retain-AI Test",
    page_icon="🔬",
)

st.title("Retain-AI Deployment Test")

st.success("Streamlit application started successfully.")

st.write("Python version test")

import sys

st.code(sys.version)

st.write("Basic application health check passed.")