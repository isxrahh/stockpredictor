import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from datetime import timedelta

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
    .header {
        font-size:28px;
        font-weight:700;
        background: linear-gradient(90deg,#005bea,#00c6fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 4px;
    }
    .subtle {
        color: #6b7280;
        margin-top: -10px;
        margin-bottom: 10px;
    }
    .card {
        background: white;
        border-radius: 12px;
        padding: 18px 20px;
        height:0px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
        min-height: 120px;
        display:flex;
        flex-direction:column;
        justify-content:space-between;
        border:1px solid #f3f4f6;
    }
    .metric-title { color:#6b7280; font-size:12px; margin-bottom:6px; }

    .card h2,.card h3{
    margin:0;
    padding:0;}

    [data-testid="column"]{
    padding-right:10px;}

    .status-row{
    display:flex;
    align-items:center;
    gap:8px;
    }

    .status-dot{
    width:12px;
    height:12px;
    border-radius:50%;
    background:#22c55e;
    }
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
# Model Status Renderer (single card)
# ---------------------------
def render_model_status(status: str):
    if status == "pending":
        return """
        <div class='card'>
            <div class='metric-title'>Model Status</div>
            <h4>🟡 Pending</h4>
            <p style='opacity:0.75;margin-top:-12px;'>Click <b>Run Prediction</b> to start.</p>
        </div>
        """

    elif status == "training":
        return """
        <style>
        @keyframes blink {
          0% { opacity: 0.2; } 
          20% { opacity: 1; } 
          100% { opacity: 0.2; }
        }
        .blink span { 
          animation: blink 1.4s infinite both;
        }
        .blink span:nth-child(2) { animation-delay:.2s; }
        .blink span:nth-child(3) { animation-delay:.4s; }
        </style>

        <div class='card'>
            <div class='metric-title'>Model Status</div>
            <h4 class='blink'>🟢 Training<span>.</span><span>.</span><span>.</span></h4>
            <p style='opacity:0.75;margin-top:-12px;'>Learning from 2 years of data...</p>
        </div>
        """

    elif status == "completed":
        return """
        <div class='card'>
            <div class='metric-title'>Model Status</div>
            <h4 style='color:#4caf50;'>✅ Completed</h4>
            <p style='opacity:0.75;margin-top:-12px;'>Predictions generated.</p>
        </div>
        """

    elif status == "failed":
        return """
        <div class='card'>
            <div class='metric-title'>Model Status</div>
            <h4 style='color:#ff5252;'>❌ Failed</h4>
            <p style='opacity:0.75;margin-top:-12px;'>Check logs & retry.</p>
        </div>
        """

    else:
        return "<p>Invalid model status</p>"


# ---------------------------
# Model (core logic)
# ---------------------------
class Brain(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        # Accept (seq, input) or (batch, seq, input)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        batch = x.size(0)
        h0 = torch.zeros(self.num_layers, batch, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch, self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


# ---------------------------
# Helpers (cached)
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
        X.append(scaled[i - seq : i, :])  # (seq,1)
        y.append(scaled[i, :])  # (1,)
    X = np.array(X)
    y = np.array(y)
    return X, y, scaler, scaled


def get_future_dates(last_date, n):
    future_dates = []
    tmp = last_date
    while len(future_dates) < n:
        tmp = tmp + timedelta(days=1)
        if tmp.weekday() < 5:  # skip weekends
            future_dates.append(tmp)
    return future_dates


# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("Controls")
    companies = {
        "Reliance Industries": "RELIANCE.NS",
        "Tata Motors": "TATAMOTORS.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "Infosys": "INFY.NS",
        "TCS": "TCS.NS",
        "SBI": "SBIN.NS",
        "ICICI Bank": "ICICIBANK.NS",
        "Bajaj Finance": "BAJFINANCE.NS",
        "Adani Enterprises": "ADANIENT.NS",
        "Wipro": "WIPRO.NS",
        "L&T": "LT.NS",
        "Maruti Suzuki": "MARUTI.NS",
    }
    choice = st.selectbox("Choose company", list(companies.keys()))
    symbol = companies[choice]
    days_ahead = st.slider("Predict next how many trading days?", 1, 30, 10)
    retrain_every_run = st.checkbox("Retrain model every run", value=True)
    seq_len = st.number_input(
        "Sequence length (timesteps)", min_value=10, max_value=120, value=60, step=10
    )
    epochs = st.slider("Training epochs", 5, 200, 35)
    st.markdown("---")
    st.caption(
        "Note: This demo trains on CPU and is intentionally simple. Not financial advice."
    )

# ---------------------------
# Main UI / Workflow
# ---------------------------
# Session state for model status
if "model_status" not in st.session_state:
    st.session_state.model_status = "pending"

run = st.button("Run Prediction")

if run or True:
    with st.spinner("Fetching data..."):
        df = download_data(symbol, period="2y")
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

    col_a, col_b, col_c = st.columns([1, 1, 1])
    col_a.markdown(
        "<div class='card'><div class='metric-title'>Latest Close</div><h2>₹{:.2f}</h2></div>".format(
            latest_price
        ),
        unsafe_allow_html=True,
    )
    naive_next = latest_price
    col_b.markdown(
        "<div class='card'><div class='metric-title'>Naive (last) Price</div><h3>₹{:.2f}</h3></div>".format(
            naive_next
        ),
        unsafe_allow_html=True,
    )

    status_placeholder = col_c.empty()
    status_placeholder.markdown(
        render_model_status(st.session_state.model_status), unsafe_allow_html=True
    )

    left, right = st.columns([1, 2])

    with left:
        st.subheader("Model training")
        st.write("Training a small LSTM on the last 2 years of closing prices.")

        # Set status to training (update the single placeholder)
        st.session_state.model_status = "training"
        status_placeholder.markdown(render_model_status("training"), unsafe_allow_html=True)

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
                # update UI
                progress_fraction = (ep + 1) / epochs
                progress_bar.progress(progress_fraction)
                progress_text.text(f"Epoch {ep+1}/{epochs} — loss {loss.item():.6f}")

            progress_text.text("Training complete.")
            st.success("Model trained ✅")

            # update status to completed (single placeholder)
            st.session_state.model_status = "completed"
            status_placeholder.markdown(render_model_status("completed"), unsafe_allow_html=True)

        except Exception as e:
            st.session_state.model_status = "failed"
            status_placeholder.markdown(render_model_status("failed"), unsafe_allow_html=True)
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
