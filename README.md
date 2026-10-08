# ☀️ AI Solar Planning & Financial Analysis System v2.0

An Enterprise-Grade, AI-Powered Solar Planning, Power Forecasting & Financial Simulation platform built with **Python, Streamlit, Scikit-Learn, Plotly, and Joblib**.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Model Accuracy](https://img.shields.io/badge/Model%20R%C2%B2-95.4%25-brightgreen.svg)]()

---

## ⚡ Key Highlights (v2.0 Performance & Design Upgrade)
- **Zero-Delay Startup:** Pre-trained binary model (`solar_model.joblib`) loads in **< 5 milliseconds**, completely eliminating the startup loading delay and freezing issues on cloud platforms.
- **Glassmorphic Modern UI:** Dark-mode cyber aesthetic with glowing KPI cards, animated hover effects, responsive typography, and customized Plotly charts.
- **Production Deployment Ready:** Equipped with `requirements.txt`, `.streamlit/config.toml`, and universal `app.py` entrypoint for 1-click deployment on Streamlit Community Cloud, Render, or Hugging Face.

---

## 🚀 Features & Capabilities

### 1. 🏠 Enterprise Dashboard
- Live generation KPI tracking and system efficiency calculator.
- Monthly financial savings estimation and CO₂ carbon mitigation metrics.
- 25-year ROI and investment amortization benchmarks.
- Clean dataset explorer with real-time health diagnostics.

### 2. 📊 Solar Farm Data Visualization
- **Diurnal Profiles:** Hourly average DC generation curves.
- **Seasonal Analysis:** Monthly energy yield breakdown.
- **Physics Correlations:** Irradiance ($G$) vs. Cell Temperature ($T$) vs. DC Power scatter plots with interactive hover tooltips.

### 3. 🔮 AI Prediction & Planning Hub
- **🏠 Residential Rooftop Planner:**
  - Automated system capacity sizing based on roof area and consumption.
  - Government subsidy calculation (PM Surya Ghar Muft Bijli Yojana, up to ₹78,000 cap).
  - 25-Year profit and bill reduction simulation.
  - One-click downloadable **Official PDF Solar Report**.
- **☀️ Real-Time ML DC Power Prediction:**
  - Fast inference using Polynomial Regression ($deg=2$) and Decision Tree Regressor.
  - Historical comparison against baseline generation for the same hour and month.
  - Interactive speedometer gauge meter.
  - Actionable diagnostic and maintenance alerts.
  - Session prediction logs with CSV export.
- **🏭 Commercial & Industrial Solar Planner:**
  - Multi-acre and large-scale MW/kW facility design.
  - Detailed CAPEX ledger (PV Modules, Inverter Clusters, HT Transformers, BESS Battery Storage, Cabling, Civil Foundations, GST).
  - Dual-axis 25-year cashflow simulation.
  - Downloadable **Detailed Project Report (PDF)** and **Full Financial Model (Excel .xlsx)**.

---

## 🛠️ Technology Stack
- **Frontend & App:** [Streamlit](https://streamlit.io/) with Custom CSS (Glassmorphism, animations)
- **Machine Learning:** [Scikit-learn](https://scikit-learn.org/) (PolynomialFeatures, DecisionTreeRegressor)
- **Model Serialization:** [Joblib](https://joblib.readthedocs.io/)
- **Data Engineering:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Visualizations:** [Plotly Express & Graph Objects](https://plotly.com/python/)
- **Document Export Engines:** [FPDF2](https://py-pdf.github.io/fpdf2/) (PDF generation), [OpenPyXL](https://openpyxl.readthedocs.io/) (Excel spreadsheets)

---

## 💻 Quick Start (Run Locally)

### 1. Clone the repository
```bash
git clone https://github.com/Pareshprajapati-777/AI-Solar-Planning-Financial-Analysis-System.git
cd AI-Solar-Planning-Financial-Analysis-System
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
streamlit run app.py
```
*(or run `streamlit run ML_Solar.py`)*

The application will launch immediately at `http://localhost:8501`.

---

## 🌐 Deploy Live on Streamlit Community Cloud (Free)

1. Fork or push this repository to your GitHub account.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New app"**.
4. Select your repository: `Pareshprajapati-777/AI-Solar-Planning-Financial-Analysis-System`
5. Branch: `main`
6. Main file path: `app.py` (or `ML_Solar.py`)
7. Click **"Deploy"**!

Your app will be live and fast in seconds with zero startup bottleneck!
