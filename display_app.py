import streamlit as st
from PIL import Image

st.title("HPC AI Deep Learning Benchmark")
st.header("Project Results")

image = Image.open('hpc_performance_analysis.png')
st.image(image, caption="Performance Analysis", use_container_width=True)

with open('project_report.txt') as f:
    report = f.read()
st.header("Detailed Report")
st.text(report)
