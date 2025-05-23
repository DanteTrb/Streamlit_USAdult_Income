# Streamlit_USAdult_Income

### 🔗 Online App

The application is publicly available on Streamlit Community Cloud at the following link:  
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://appusadultincome-bqv9cycx8ovr3c2j8kpnn3.streamlit.app)

🐍![Python Version](https://img.shields.io/badge/python-3.11-blue)
📄![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
A model to predict individual income thresholds in the US (`<=50K` or `>50K`) using synthetically generated data via CTGAN and classified with XGBoost.

---

## 🧠 Project Objective

To simulate a realistic dataset of US adults and build a machine learning model capable of predicting annual income, then **visualize everything interactively** using Streamlit.

---

## 🛠️ Technologies Used

- 📊 **Original Dataset:** UCI Adult Dataset  
- 🧬 **Synthetic Data Generator:** [CTGAN](https://sdv.dev/SDV/user_guides/single_table/ctgan.html)  
- 🤖 **Classifier:** XGBoost  
- 🔍 **Model Explainability:** SHAP values  
- 🌐 **Interactive Dashboard:** Streamlit

---

## 🧪 How the App Works

The Streamlit app includes the following sections:

1. **Synthetic dataset** generated with CTGAN  
2. **XGBoost model performance report**  
3. **SHAP plots** to explain feature importance  
4. **Download button** to export synthetic data

👉 Try it now by clicking the badge above!

---

## 📁 Project Structure

Streamlit_USAdult_Income/
├── data/ # Original and synthetic data
├── outputs/ # SHAP plots and model results
├── streamlit_app/ # Streamlit app code
│ └── app.py
├── requirements.txt # Python dependencies
└── README.md


---

## 💻 Run Locally

```bash
git clone https://github.com/DanteTrb/Streamlit_USAdult_Income.git
cd Streamlit_USAdult_Income
pip install -r requirements.txt
streamlit run streamlit_app/app.py


👤 Author
Dante Trabassi
LinkedIn: https://www.linkedin.com/in/dante-trabassi-663b3718b/
