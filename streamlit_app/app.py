import streamlit as st
import pandas as pd
from PIL import Image
import os

# Titolo dell'app
st.title("📊 Analyzing Income Model on Synthetic Data")
st.write(
    "This app shows how a machine learning model predicts income "
    " using data generated with CTGAN."
)

# === Percorsi senza '..' perché streamlit parte dalla root ===
data_path = "data/synthetic_adult_clean.csv"
shap_bar_path = "outputs/shapadultincome.png"
shap_summary_path = "outputs/shap_summary_plot.png"

# Caricamento dei dati
st.subheader("📁 Used Dataset")
try:
    df = pd.read_csv(data_path)
    st.success("✅ Dataset uploaded successfully!")
    st.dataframe(df.head())
    
    # Download dei dati
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download synthetic data", csv, "synthetic_income_dataset.csv", "text/csv")
except FileNotFoundError:
    st.error("❌ File not found! Make sure the file is in the 'data' folder.")

# Classification report
st.subheader("📄 XGBoost model results")
st.markdown("**Classification Report:**")
st.code('''
              precision    recall  f1-score   support
       <=50K       0.88      0.84      0.86       136
        >50K       0.62      0.69      0.65        52

    accuracy                           0.80       188
   macro avg       0.75      0.77      0.76       188
weighted avg       0.81      0.80      0.80       188
''')

st.info("🟡 The model has a good overall accuracy (80%), but there is still some imbalance in the >50K class.")

# SHAP images
st.subheader("🔍 SHAP: Model Explanation")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**📊 Average importance of features**")
    st.caption("🎓 This chart shows which features influence the forecast the most.")
    if os.path.exists(shap_bar_path):
        shap_bar = Image.open(shap_bar_path)
        st.image(shap_bar, caption="Bar Chart SHAP", use_column_width=True)
    else:
        st.warning(f"⚠️ File not found: {shap_bar_path}")

with col2:
    st.markdown("**📈 Distribution of SHAP impacts**")
    st.caption("🎯 Each point represents a person: color = feature value, position = impact.")
    if os.path.exists(shap_summary_path):
        shap_summary = Image.open(shap_summary_path)
        st.image(shap_summary, caption="SHAP Summary Plot", use_column_width=True)
    else:
        st.warning(f"⚠️ File not found: {shap_summary_path}")

# Footer
st.markdown("---")
st.markdown("WebApp developed by **Dante Trabassi** © 2025")