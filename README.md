# 📈 AI Stock Prediction Dashboard

A production-ready **AI-powered stock analysis & prediction dashboard** built with **Streamlit**, **PyTorch (LSTM)**, and **real-time market data APIs**.  
The application combines **machine learning price forecasting**, **interactive financial visualizations**, and **professional trading widgets** into a single unified interface.

> ⚠️ **Disclaimer:** This application is for educational and research purposes only.  
> It does **not** constitute financial or investment advice.

---

## 🚀 Live Demo

🔗 **Deployed App:** https://sstockpredictor.streamlit.app/

---

## 🖼️ Screenshots

### Dashboard Overview
> <img width="1919" height="887" alt="image" src="https://github.com/user-attachments/assets/764b9385-c4c5-4583-a455-7b552dc29e00" />


### AI Price Prediction & Training
> <img width="1918" height="884" alt="image" src="https://github.com/user-attachments/assets/42bf11f0-c289-4087-b2c1-40c7b64d7c84" />


### Market Widgets & Technical Analysis
> <img width="1913" height="879" alt="image" src="https://github.com/user-attachments/assets/851509eb-a094-4840-882c-385b04f1fe44" />


---

## 🧠 Key Features

### 🔍 Intelligent Stock Search
- Real-time **company search powered by Algolia**
- Fast symbol lookup with instant suggestions
- Automatic exchange mapping (NSE, BSE, NASDAQ, NYSE)

---

### 🤖 AI Price Prediction (LSTM)
- Custom **PyTorch LSTM model**
- Trains dynamically on **2 years of historical price data**
- Predicts **future closing prices (1–30 trading days)**
- Adjustable:
  - Sequence length
  - Training epochs
  - Retraining behavior per run

---

### 📊 Advanced Market Visualizations
- Interactive **Plotly price charts**
- Technical overlays:
  - Moving averages (MA20, MA50)
- Historical vs predicted price comparison
- Downloadable prediction results (CSV)

---

### 📈 Professional Trading Widgets
Embedded **TradingView & Trendlyne widgets**:
- Technical Analysis summary
- Financial statements
- Company profile
- SWOT analysis
- Market heatmaps
- Live ticker tape
- Stock screener
- Market news feed

---

### ⚡ Smart Data Handling
- Primary source: **TwelveData API**
- Fallback: **Yahoo Finance**
- Intelligent caching using `st.cache_data`
- Trading-day aware future date generation

---

## 🏗️ Tech Stack

| Category | Technology |
|--------|-----------|
| Frontend | Streamlit |
| ML Framework | PyTorch |
| Model | LSTM Neural Network |
| Data APIs | Yahoo Finance, TwelveData |
| Search | Algolia |
| Charts | Plotly |
| Deployment | Docker |
| Language | Python 3.13 |

---

## 📁 Project Structure

├── app.py

├── Dockerfile

├── requirements.txt

├── .dockerignore

└── README.md

---

## 🐳 Docker Deployment (Production)

This project is **fully Dockerized** for consistent and reliable deployment.

### Build Image
```bash
docker build -t ai-stock-dashboard .


