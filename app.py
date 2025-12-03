import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from datetime import timedelta
import requests

torch.set_float32_matmul_precision("medium")  # Faster on newer CPUs
np.set_printoptions(suppress=True)
pd.set_option("display.float_format", "{:.2f}".format)

# ---------------------------
# Page config & CSS
# ---------------------------
st.set_page_config(page_title="Stock AI Predictor", layout="wide", page_icon="📈")

st.markdown(
    """
<style>
.header { font-size:28px; font-weight:700; background: linear-gradient(90deg,#005bea,#00c6fb);
-webkit-background-clip: text; -webkit-text-fill-color: transparent; padding-bottom: 4px; }
.subtle { color: #6b7280; margin-top: -10px; margin-bottom: 10px; }
.card { background: white; border-radius: 12px; padding: 18px 20px; height:0px; 
box-shadow: 0 6px 18px rgba(0,0,0,0.06); min-height: 120px; display:flex; flex-direction:column; 
justify-content:space-between; border:1px solid #f3f4f6; }
.metric-title { color:#6b7280; font-size:12px; margin-bottom:6px; }
.card h2,.card h3{ margin:0; padding:0; }
[data-testid="column"]{ padding-right:10px; }
.status-row{ display:flex; align-items:center; gap:8px; }
.status-dot{ width:12px; height:12px; border-radius:50%; background:#22c55e; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='header'>AI Stock Prediction Dashboard</div>", unsafe_allow_html=True
)
st.markdown(
    "<div class='subtle'>Yahoo Finance + simple LSTM — small demo (not financial advice)</div>",
    unsafe_allow_html=True,
)


# ---------------------------
# Model Status Renderer
# ---------------------------
def render_model_status(status: str):
    if status == "pending":
        return "<div class='card'><div class='metric-title'>Model Status</div><h4>🟡 Pending</h4><p style='opacity:0.75;margin-top:-12px;'>Click <b>Run Prediction</b> to start.</p></div>"
    elif status == "training":
        return """
        <style>@keyframes blink {0% {opacity:0.2;} 20% {opacity:1;} 100% {opacity:0.2;}}
        .blink span {animation: blink 1.4s infinite both;}
        .blink span:nth-child(2) { animation-delay:.2s; } .blink span:nth-child(3) { animation-delay:.4s; }
        </style>
        <div class='card'><div class='metric-title'>Model Status</div>
        <h4 class='blink'>🟢 Training<span>.</span><span>.</span><span>.</span></h4>
        <p style='opacity:0.75;margin-top:-12px;'>Learning from 2 years of data...</p></div>
        """
    elif status == "completed":
        return "<div class='card'><div class='metric-title'>Model Status</div><h4 style='color:#4caf50;'>✅ Completed</h4><p style='opacity:0.75;margin-top:-12px;'>Predictions generated.</p></div>"
    elif status == "failed":
        return "<div class='card'><div class='metric-title'>Model Status</div><h4 style='color:#ff5252;'>❌ Failed</h4><p style='opacity:0.75;margin-top:-12px;'>Check logs & retry.</p></div>"
    else:
        return "<p>Invalid model status</p>"


# ---------------------------
# LSTM Model
# ---------------------------
class Brain(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        h0 = torch.zeros(1, x.size(0), 50)
        c0 = torch.zeros(1, x.size(0), 50)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def download_data(symbol: str, period: str = "2y"):
    df = yf.download(symbol, period=period, progress=False)
    return df


def prepare_sequences(close_values: np.ndarray, seq: int = 60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_values.astype(np.float32))
    X, y = [], []
    for i in range(seq, len(scaled)):
        X.append(scaled[i - seq : i, :])
        y.append(scaled[i, :])
    return np.array(X), np.array(y), scaler, scaled


def get_future_dates(last_date, n):
    future_dates = []
    tmp = last_date
    while len(future_dates) < n:
        tmp += timedelta(days=1)
        if tmp.weekday() < 5:
            future_dates.append(tmp)
    return future_dates


# ---------------------------
# Sidebar 
# ---------------------------
with st.sidebar:
    st.header("Controls")

    query = st.text_input("Search company", placeholder="Type company name...")

    company_options = []
    algolia_raw = {}

    if query and len(query) >= 2:
        try:
            ALGOLIA_APP_ID = "XM6V4TK0GI"
            ALGOLIA_API_KEY = "f2739da0f5b32535dc8b656db109394c"
            ALGOLIA_INDEX_NAME = "sec_list"
            url = f"https://{ALGOLIA_APP_ID}-dsn.algolia.net/1/indexes/{ALGOLIA_INDEX_NAME}/query"
            headers = {
                "X-Algolia-API-Key": ALGOLIA_API_KEY,
                "X-Algolia-Application-Id": ALGOLIA_APP_ID,
                "Content-Type": "application/json",
            }
            payload = {"params": f"query={query}&hitsPerPage=10"}
            resp = requests.post(url, json=payload, headers=headers, timeout=5)

            if resp.status_code == 200:
                algolia_raw = resp.json()  # Save raw response for debugging
                hits = algolia_raw.get("hits", [])
                company_options = [
                    (h.get("Security Name", ""), h.get("Symbol", ""))
                    for h in hits
                    if h.get("Security Name") and h.get("Symbol")
                ]
            else:
                st.error(f"Algolia request failed with status {resp.status_code}")
        except Exception as e:
            st.error(f"Algolia search error: {e}")

    # Show results if available
    if company_options:
        choice = st.selectbox("Choose company", [c[0] for c in company_options])
        symbol = dict(company_options)[choice]
        yf_symbol=symbol+'.NS'
    else:
        st.info("Start typing to see company suggestions...")
        symbol = None
        yf_symbol=None

    # ---- Debug panel ----
    with st.expander("Debug: Algolia raw response"):
        st.json(algolia_raw)  # Shows the raw JSON returned by Algolia

    # Other controls
    days_ahead = st.slider("Predict next how many trading days?", 1, 30, 10)
    retrain_every_run = st.checkbox("Retrain model every run", value=True)
    seq_len = st.number_input(
        "Sequence length (timesteps)", min_value=10, max_value=120, value=60, step=10
    )
    epochs = st.slider("Training epochs", 5, 200, 35)
    st.markdown("---")
    st.caption("Demo trains on CPU. Not financial advice.")


# ---------------------------
# Stop if no company is selected
# ---------------------------
if symbol is None:
    st.warning(
        "Please select a company from the search box before running predictions."
    )
    st.stop()

# ---------------------------
# Prediction code
# ---------------------------
run = st.button("Run Prediction")
if run:
    with st.spinner("Fetching data..."):
        df = download_data(yf_symbol, period="2y")

    if df is None or df.empty:
        st.error("No data returned for this symbol.")
        st.stop()

    if "Close" not in df.columns:
        st.error("Close column not available in downloaded data.")
        st.stop()

    close = df["Close"].dropna()
    if len(close) < 100:
        st.error("Not enough data (need at least 100 close prices).")
        st.stop()
    dates = close.index
    close_vals = close.values.reshape(-1, 1).astype(np.float32)
    latest_price = float(close_vals[-1, 0])
    X, y, scaler, scaled_all = prepare_sequences(close_vals, seq=seq_len)

    if X.ndim != 3 or X.shape[2] != 1:
        st.error(f"Unexpected X shape: {X.shape}. Expect (N, {seq_len}, 1).")
        st.stop()

    # --- Status cards ---
    col_a, col_b, col_c = st.columns([1, 1, 1])
    col_a.markdown(
        f"<div class='card'><div class='metric-title'>Latest Close</div><h2>₹{latest_price:.2f}</h2></div>",
        unsafe_allow_html=True,
    )
    col_b.markdown(
        f"<div class='card'><div class='metric-title'>Naive (last) Price</div><h3>₹{latest_price:.2f}</h3></div>",
        unsafe_allow_html=True,
    )
    status_placeholder = col_c.empty()
    status_placeholder.markdown(render_model_status("pending"), unsafe_allow_html=True)

    left, right = st.columns([1, 2])

    with left:
        st.subheader("Model training")
        st.write("Training a small LSTM on the last 2 years of closing prices.")

        st.session_state.model_status = "training"
        status_placeholder.markdown(
            render_model_status("training"), unsafe_allow_html=True
        )

        try:
            model = Brain(input_size=1, hidden_size=50, num_layers=1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            loss_fn = nn.MSELoss()

            X_t = torch.from_numpy(X).float()  # (N, seq, 1)
            y_t = torch.from_numpy(y).float()  # (N, 1)

            progress_text = st.empty()
            progress_bar = st.progress(0)

            model.train()
            for ep in range(epochs):
                optimizer.zero_grad()
                out = model(X_t)  # (N,1)
                loss = loss_fn(out, y_t)
                loss.backward()
                optimizer.step()
                progress_fraction = (ep + 1) / epochs
                progress_bar.progress(progress_fraction)
                progress_text.text(f"Epoch {ep+1}/{epochs} — loss {loss.item():.6f}")

            progress_text.text("Training complete.")
            st.success("Model trained ✅")

            st.session_state.model_status = "completed"
            status_placeholder.markdown(
                render_model_status("completed"), unsafe_allow_html=True
            )

        except Exception as e:
            st.session_state.model_status = "failed"
            status_placeholder.markdown(
                render_model_status("failed"), unsafe_allow_html=True
            )
            st.error(f"Training failed: {e}")
            st.stop()

        # switch to eval mode for prediction
        model.eval()

        pred_container = st.container()
        with pred_container:
            st.write("Generating future predictions...")
            pred_progress = st.progress(0)
            pred_status = st.empty()

        preds = []
        current = scaled_all[-seq_len:].copy()

        with torch.no_grad():
            for i in range(days_ahead):
                x = torch.from_numpy(current).float().unsqueeze(0)
                p_t = model(x)
                p = float(p_t.item())
                preds.append(p)
                current = np.vstack([current[1:], [[p]]]).astype(np.float32)

                # Update progress
                progress = (i + 1) / days_ahead
                pred_progress.progress(progress)
                pred_status.markdown(
                    f"Predicting day **{i+1}/{days_ahead}** → ₹{p:.2f}"
                )

        pred_progress.progress(1.0)
        pred_status.success("All predictions ready!")
        pred_container.empty()
        pred_prices = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        future_dates = get_future_dates(dates[-1], days_ahead)
        pred_df = pd.DataFrame(
            {
                "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
                "Predicted Price": pred_prices,
            }
        )
        csv = pred_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download predictions (CSV)",
            data=csv,
            file_name=f"{symbol}_predictions.csv",
            mime="text/csv",
        )

    with right:
        st.subheader(f"{choice} — Price Chart")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=close_vals.flatten(),
                mode="lines",
                name="Close",
                line=dict(color="#1f77b4"),
            )
        )
        try:
            df_ma = pd.DataFrame({"Close": close_vals.flatten()}, index=dates)
            df_ma["MA20"] = df_ma["Close"].rolling(20).mean()
            df_ma["MA50"] = df_ma["Close"].rolling(50).mean()
            fig.add_trace(
                go.Scatter(
                    x=df_ma.index,
                    y=df_ma["MA20"],
                    mode="lines",
                    name="MA20",
                    line=dict(dash="dash"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_ma.index,
                    y=df_ma["MA50"],
                    mode="lines",
                    name="MA50",
                    line=dict(dash="dot"),
                )
            )
        except Exception:
            pass

        # plot predictions
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=pred_prices,
                mode="markers+lines",
                name="Prediction",
                marker=dict(color="#ef553b", size=8),
            )
        )
        fig.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("## Predictions")
    col1, col2 = st.columns(2)
    try:
        latest_pred = float(pred_prices[-1])
    except Exception:
        latest_pred = None

    pct_change = None
    if latest_pred is not None:
        pct_change = (latest_pred - latest_price) / latest_price * 100

    col1.metric("Latest Closing Price", f"₹{latest_price:.2f}")
    if latest_pred is not None:
        col2.metric(
            f"Predicted in {days_ahead} days",
            f"₹{latest_pred:.2f}",
            f"{pct_change:.2f}%",
        )

    display_df = pd.DataFrame(
        {
            "Date": [d.strftime("%b %d, %Y") for d in future_dates],
            "Predicted Price (₹)": [f"{p:.2f}" for p in pred_prices],
        }
    )
    st.table(display_df)

st.caption("Demo app — Not financial advice. Data via Yahoo Finance.")
