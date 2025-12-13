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
from profile_widget import yahoo_card

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
.card { background: transparent; border: 1px solid #343a42 !important; border-radius: 12px; padding: 18px 20px; height:0px; margin:30px 0;
box-shadow: 0 6px 18px rgba(0,0,0,0.06); min-height: 120px; display:flex; flex-direction:column; 
justify-content:space-between; border:1px solid #f3f4f6; }
.metric-title { color:#6b7280; font-size:12px; margin-bottom:6px; }
.card h2,.card h3{ margin:0; padding:0; }
[data-testid="column"]{ padding-right:10px; }
.status-row{ display:flex; align-items:center; gap:8px; }
.status-dot{ width:12px; height:12px; border-radius:50%; background:#22c55e; }
.big-font { font-size: 46px !important; font-weight: bold; background: linear-gradient(90deg,#667eea,#764ba2);
                 -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin:0; }
    </style>
""",
    unsafe_allow_html=True,
)

st.session_state.show_widgets = False

st.markdown(
    "<div class='header big-font'>AI Stock Prediction Dashboard</div>",
    unsafe_allow_html=True,
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
@st.cache_data(show_spinner=False, ttl=3600)
def download_data(symbol: str, period: str = "2y"):
    TWELVE_DATA_KEY = st.secrets.get("TWELVE_DATA_KEY") or "your_key_here"
    if TWELVE_DATA_KEY != "5a1f3871569543aca6034279f126b3c8":
        url = f"https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol.replace(
                ".NS", ""
            ),  # Twelve Data uses RELIANCE, not RELIANCE.NS
            "interval": "1day",
            "outputsize": 730,  # ~2 years
            "apikey": TWELVE_DATA_KEY,
            "dp": 5,
            "format": "JSON",
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            if "values" in data:
                df = pd.DataFrame(data["values"])
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime")
                df = df[["close"]].rename(columns={"close": "Close"})
                df["Close"] = df["Close"].astype(float)
                df = df.sort_index()
                return df
        except:
            pass

    # Fallback to yfinance
    ticker = symbol + ".NS" if not symbol.endswith((".NS", ".BO")) else symbol
    df = yf.download(ticker, period=period, progress=False)
    if df.empty or "Close" not in df.columns:
        return None
    return df[["Close"]].dropna()


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
    st.header("Controls & Search")

    # ---- Single search input ----
    query = st.text_input(
        "Search company", placeholder="Type company name...", key="search_inp"
    )

    st.markdown(
        """
<style>
[data-testid="stTextInput"] > div > div > input {
    padding-left: 38px !important;   /* space for logo */
}

/* Logo container */
.logo-icon {
    position: absolute;
    left: 10px;
    bottom: 29px;
    width: 20px;
    height: 20px;
    opacity: 0.7;
}

/* Position wrapper */
.stTextInput {
    position: relative;
}
</style>

<div class="logo-icon">
    <img src="https://logosandtypes.com/wp-content/uploads/2023/11/algolia.svg" width="20"/>
</div>
""",
        unsafe_allow_html=True,
    )

    # ---- Initialize session state ----
    if "company_name" not in st.session_state:
        st.session_state["company_name"] = None
    if "company_symbol" not in st.session_state:
        st.session_state["company_symbol"] = None

    company_options = []
    symbol = None
    yf_symbol = None

    # ---- Algolia search ----
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
            hits = resp.json().get("hits", []) if resp.status_code == 200 else []

            company_options = [
                {"name": h.get("Security Name", ""), "symbol": h.get("Symbol", "")}
                for h in hits
            ]
            for opt in company_options:
                if st.button(f"{opt['name']} ({opt['symbol']})", width="stretch"):
                    st.session_state.company_name = opt["name"]
                    st.session_state.company_symbol = opt["symbol"]
                    st.rerun()
        except Exception as e:
            st.error(f"Algolia search error: {e}")

    # ---- Show suggestions as buttons ----
    if company_options:
        st.markdown("**Suggestions:**")
        for c in company_options:
            if st.button(f"{c['name']} ({c['symbol']})"):
                st.session_state["company_name"] = c["name"]
                st.session_state["company_symbol"] = c["symbol"]

        # ---- Dropdown to choose company ----
        choice = st.selectbox(
            "Choose company",
            [c["name"] for c in company_options],
            index=next(
                (
                    i
                    for i, c in enumerate(company_options)
                    if c["name"] == st.session_state.get("company_name")
                ),
                0,
            ),
        )

        # Update session state if dropdown changed
        selected_company = next(
            (c for c in company_options if c["name"] == choice), None
        )
        if selected_company:
            st.session_state["company_name"] = selected_company["name"]
            st.session_state["company_symbol"] = selected_company["symbol"]

    # ---- Final selection ----
    if st.session_state.get("company_symbol"):
        symbol = st.session_state["company_symbol"]
        yf_symbol = symbol + ".NS"
        st.success(f"Selected: {st.session_state['company_name']} ({symbol})")
    else:
        st.info("Start typing to see company suggestions...")
        symbol = None
        yf_symbol = None

    # Other controls
    days_ahead = st.slider("Predict next how many trading days?", 1, 30, 10)
    retrain_every_run = st.checkbox("Retrain model every run", value=True)
    seq_len = st.number_input(
        "Sequence length (timesteps)", min_value=10, max_value=120, value=60, step=10
    )
    epochs = st.slider("Training epochs", 5, 200, 35)
    st.markdown("---")
    st.caption("Demo trains on CPU. Not financial advice.")
    # ────────────────────── SAFETY CHECK – DO NOT PUT ANYTHING ABOVE THIS ──────────────────────
    if "symbol" not in locals() and st.session_state.get("company_symbol"):
        symbol = st.session_state["company_symbol"]
    elif "symbol" not in locals():
        symbol = None
    # ────────────────────── NOW "symbol" ALWAYS EXISTS ──────────────────────

# ---------- Robust TradingView symbol selection ----------
# `symbol` here is what you get from Algolia, e.g. "TCS" or "RELIANCE" or "AAPL"
base_sym = symbol.upper() if symbol else ""

def build_tv_candidates(base):
    # Prioritized guess-list — prefer Indian exchanges first (NSE/BSE),
    # then US exchanges, then plain base as last resort.
    if not base:
        return [""]

    candidates = []
    # Indian style (user likely searching Indian stocks)
    candidates.append(f"NSE:{base}")
    candidates.append(f"BSE:{base}")
    # Common global prefixes
    candidates.append(f"NASDAQ:{base}")
    candidates.append(f"NYSE:{base}")
    # Plain ticker (some widgets accept just the base)
    candidates.append(base)
    # dedupe while preserving order
    seen = set(); uniq = []
    for c in candidates:
        if c not in seen:
            uniq.append(c); seen.add(c)
    return uniq

tv_candidates = build_tv_candidates(base_sym)

# Show the candidates and let user pick — default to first (NSE:...) so Indian tickers map nicely.
st.write("Suggested TradingView candidates (pick one that renders):")
chosen_tv = st.selectbox("TradingView symbol", tv_candidates, index=0, help="If widget shows 'invalid symbol' try a different candidate here")

# set final variables used by widgets
tv_symbol = chosen_tv
trend_symbol = base_sym 

# Stop if no company is selected
# ---------------------------
if symbol is None:
    st.warning(
        "Please select a company from the search box before running predictions."
    )
    ## OTHER WIDGETS OF TRADING VIEW

    # Live S&P 500 Heatmap — Embed TradingView
    st.markdown("### Live Nifty 50 Heatmap")

    st.components.v1.html(
        """
        <div style="width: 100%; height: 660px; margin: 0; padding: 0;">
            <div class="tradingview-widget-container">
                <div class="tradingview-widget-container__widget"></div>
                    <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/heatmap/stock/" rel="noopener nofollow" target="_blank"><span class="blue-text">Stock Heatmap</span></a><span class="trademark"> by TradingView</span></div>
                        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-stock-heatmap.js" async>
                            {
                                "dataSource": "SPX500",
                                "blockSize": "market_cap_basic",
                                "blockColor": "change",
                                "grouping": "sector",
                                "locale": "en",
                                "symbolUrl": "",
                                "colorTheme": "light", 
                                "exchanges": [],
                                "hasTopBar": false,
                                "isDataSetEnabled": false,
                                "isZoomEnabled": true,
                                "hasSymbolTooltip": true,
                                "isMonoSize": false,
                                "width": "100%",
                                "height": "100%"
                            }
                        </script>
                    </div>
                </div>
            </div>
        </div>
        """,
        height=700,
        scrolling=False,
    )

    st.components.v1.html(
        """
            <div style="width: 100%; margin: 0; padding: 0;">
                <!-- TradingView Widget BEGIN -->
                <div class="tradingview-widget-container">
                  <div class="tradingview-widget-container__widget"></div>
                  <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/markets/" rel="noopener nofollow" target="_blank"><span class="blue-text">Ticker tape</span></a><span class="trademark"> by TradingView</span></div>
                  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
                  {
                  "symbols": [
                    {
                      "proName": "FOREXCOM:SPXUSD",
                      "title": "S&P 500 Index"
                    },
                    {
                      "proName": "FOREXCOM:NSXUSD",
                      "title": "US 100 Cash CFD"
                    },
                    {
                      "proName": "FX_IDC:EURUSD",
                      "title": "EUR to USD"
                    },
                    {
                      "proName": "BITSTAMP:BTCUSD",
                      "title": "Bitcoin"
                    },
                    {
                      "proName": "BITSTAMP:ETHUSD",
                      "title": "Ethereum"
                    },
                    {
                      "proName": "CAPITALCOM:GOLD",
                      "title": "Gold"
                    },
                    {
                      "proName": "NSE:NIFTY",
                      "title": "Nifty 50 Index"
                    },
                    {
                      "proName": "BSE:SENSEX",
                      "title": "Sensex"
                    },
                    {
                      "proName": "NSE:BSE",
                      "title": "BSE"
                    },
                    {
                      "proName": "NASDAQ:TSLA",
                      "title": "Tesla"
                    },
                    {
                      "proName": "NASDAQ:APPL",
                      "title": "Apple"
                    },
                    {
                      "proName": "KRX:005930",
                      "title": "Samsung"
                    },
                    {
                      "proName": "NASDAQ:NVDA",
                      "title": "Nvidia"
                    },
                    {
                      "proName": "FX_IDC:USDINR",
                      "title": "USDINR"
                    },
                    {
                      "proName": "NSE:TCS",
                      "title": "TCS"
                    },
                    {
                      "proName": "NSE:BAJFINANCE",
                      "title": "Bajaj Finance"
                    }
                  ],
                  "colorTheme": "light",
                  "locale": "en",
                  "largeChartUrl": "",
                  "isTransparent": false,
                  "showSymbolLogo": true,
                  "displayMode": "adaptive"
                }
                  </script>
                </div>
                <!-- TradingView Widget END -->
            </div>""",
        scrolling=False,
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("### Stock Screener")
        st.components.v1.html(
            """ <!-- TradingView Widget BEGIN -->
           <div style="width: 100%; height: 660px; margin: 0; padding: 0;">
               <!-- TradingView Widget BEGIN -->
                <div class="tradingview-widget-container">
                  <div class="tradingview-widget-container__widget"></div>
                  <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/screener/" rel="noopener nofollow" target="_blank"><span class="blue-text">Stock Screener</span></a><span class="trademark"> by TradingView</span></div>
                  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-screener.js" async>
                  {
                  "market": "india",
                  "showToolbar": true,
                  "defaultColumn": "overview",
                  "defaultScreen": "most_capitalized",
                  "isTransparent": false,
                  "locale": "en",
                  "colorTheme": "light",
                  "width": "100%",
                  "height": 550
                }
                  </script>
                </div>
               <!-- TradingView Widget END -->
            </div>
            """,
            height=720,
            scrolling=False,
        )
    with col2:
        st.markdown("### Top Stories")
        st.components.v1.html(
            """
                <!-- TradingView Widget BEGIN -->
                <div style="width: 100%; height: 660px; margin: 0; padding: 0;">
                <div class="tradingview-widget-container">
                  <div class="tradingview-widget-container__widget"></div>
                  <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/news/top-providers/tradingview/" rel="noopener nofollow" target="_blank"><span class="blue-text">Top stories</span></a><span class="trademark"> by TradingView</span></div>
                  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-timeline.js" async>
                  {
                  "displayMode": "regular",
                  "feedMode": "all_symbols",
                  "colorTheme": "light",
                  "isTransparent": false,
                  "locale": "en",
                  "width": 520,
                  "height": 550
                }
                  </script>
                </div>
                </div>
                <!-- TradingView Widget END -->
            """,
            height=700,
            scrolling=False,
        )
    st.stop()

else:
    st.markdown(f"### {symbol} — AI Prediction")


# ---------------------------
# Prediction code
# ---------------------------
run = st.button("Run Prediction")
if run:
    st.session_state.show_widgets = True

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
    # ─────── 4 SAFE METRIC CARDS ───────
    c1, c2, c3, c4 = st.columns(4)

    # Ensure scalars
    latest_price = float(latest_price)
    close_21d = close.iloc[-21].item()

    # 1M return
    ret_1m = 0.0
    if len(close) > 21:
        ret_1m = (latest_price / close_21d - 1) * 100

    # Volatility (safe)
    vol = 0.0
    returns = close.pct_change().dropna()
    if len(returns) > 20:
        last_vol = returns.rolling(20).std().iloc[-1]  # ensure scalar
        vol = last_vol.item() * np.sqrt(252) * 100

    c1.metric(
        "1M Return",
        f"{ret_1m:+.2f}%",
        delta=f"{ret_1m:+.1f}%" if abs(ret_1m) > 0.01 else None,
    )
    c2.metric("Volatility", f"{vol:.1f}%" if vol > 0 else "—")
    c3.metric("52W High", f"₹{close.max().item():.2f}")
    c4.metric("Latest Price", f"₹{latest_price:.2f}")

    X, y, scaler, scaled_all = prepare_sequences(close_vals, seq=seq_len)

    if X.ndim != 3 or X.shape[2] != 1:
        st.error(f"Unexpected X shape: {X.shape}. Expect (N, {seq_len}, 1).")
        st.stop()

    # --- Status cards ---
    col_a, col_b, col_c = st.columns([1, 1, 1], gap="medium")
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
        st.subheader(f"{symbol} — Price Chart")
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
        st.plotly_chart(fig, width="stretch")
    st.markdown("## Predictions")
    col1, col2 = st.columns(2)
    try:
        latest_pred = (
            pred_prices[-1].item()
            if hasattr(pred_prices, "item")
            else float(pred_prices[-1])
        )
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

if st.session_state.get("show_widgets", False):
    
    # ---------- MAIN ROW ----------
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Technical Analysis Widget
        st.components.v1.html(
            f"""
            <div style=" height:368px; border-radius:12px; padding:0 4px;">
              <div class="tradingview-widget-container">
                <div class="tradingview-widget-container__widget"></div>
                <script src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
                {{
                  "symbol": "{tv_symbol}",
                  "colorTheme": "light",
                  "displayMode": "single",
                  "locale": "en",
                  "width": "100%",
                  "height": "100%"
                }}
                </script>
              </div>
            </div>
            """,
            height=380
        )

        # SWOT Analysis
        st.components.v1.html(
            f"""
            <div style="border-radius:12px; margin-left:5px; ">
            
              <blockquote 
                class="trendlyne-widgets"
                data-get-url="https://trendlyne.com/web-widget/swot-widget/Poppins/{trend_symbol}/?posCol=60a5fa&primaryCol=3b82f6&negCol=ef4444&neuCol=f59e0b" 
                data-theme="light"
                style="color:cornflowerblue;"
                ">
              </blockquote>
              <script async src="https://cdn-static.trendlyne.com/static/js/webwidgets/tl-widgets.js"></script>
            </div>
            """,
            height=310
        )
    
    
        # Profile Widget
        st.components.v1.html(
            f"""
            <div style=" height:276px; border-radius:12px;">
              <div class="tradingview-widget-container">
                <div class="tradingview-widget-container__widget"></div>
                <script src="https://s3.tradingview.com/external-embedding/embed-widget-symbol-profile.js" async>
                {{
                  "symbol": "{tv_symbol}",
                  "colorTheme": "light",
                  "locale": "en",
                  "width": "100%",
                  "height": "100%"
                }}
                </script>
              </div>
            </div>
            """,
            height=290
        )
    
    with col2:
        st.components.v1.html(
            f"""
            <div style="height:1000px; border-radius:12px;">
              <div class="tradingview-widget-container">
                <div class="tradingview-widget-container__widget"></div>
                <script src="https://s3.tradingview.com/external-embedding/embed-widget-financials.js" async>
                {{
                  "symbol": "{tv_symbol}",
                  "colorTheme": "light",
                  "displayMode": "regular",
                  "locale": "en",
                  "width": "100%",
                  "height": "100%"
                }}
                </script>
              </div>
            </div>
            """,
            height=1020
        )
    

st.caption("Demo app — Not financial advice. Data via Yahoo Finance & TwelveData API.")
