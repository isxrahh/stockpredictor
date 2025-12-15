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

from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

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
            background: transparent;
            border: 1px solid #343a42 !important;
            border-radius: 12px;
            padding: 18px 20px;
            height:0px;
            margin:30px 0;
            box-shadow: 0 6px 18px rgba(0,0,0,0.06);
            min-height: 140px;
            display:flex;
            flex-direction:column; 
            justify-content:space-between;
            border:1px solid #f3f4f6;
        }

        .metric-title {
            color:#6b7280;
            font-size:18px;
            margin-bottom:6px;
        }

        .card h2,.card h3 {
            margin:0; padding:0;
        }

        [data-testid="column"] {
            padding-right:10px;
        }

        .status-row {
            display:flex;
            align-items:center;
            gap:8px;
        }

        .status-dot {
            width:12px;
            height:12px;
            border-radius:50%;
            background:#22c55e;
        }

        .big-font {
            font-size: 46px !important;
            font-weight: bold;
            background: linear-gradient(90deg,#667eea,#764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin:0;
        }
        
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
        return (
            "<div class='card'>"
            "<div class='metric-title'>Model Status</div>"
            "<h4>🟡 Pending</h4>"
            "<p style='opacity:0.75;margin-top:-12px;'>Click <b>Run Prediction</b> to start.</p>"
            "</div>"
        )

    elif status == "training":
        return """
        <style>
        @keyframes blink {0% {opacity:0.2;} 20% {opacity:1;} 100% {opacity:0.2;}}
        .blink span {
            animation: blink 1.4s infinite both;
        }
        .blink span:nth-child(2) {
            animation-delay:.2s; 
        } 
        .blink span:nth-child(3) { 
            animation-delay:.4s; 
        }
        </style>
        <div class='card'><div class='metric-title'>Model Status</div>
        <h4 class='blink'>🟢 Training<span>.</span><span>.</span><span>.</span></h4>
        <p style='opacity:0.75;margin-top:-12px;'>Learning from 5 years of data...</p></div>
        """

    elif status == "completed":
        return (
            "<div class='card'>"
            "<div class='metric-title'>Model Status</div>"
            "<h4 style='color:#4caf50;'>✅ Completed</h4>"
            "<p style='opacity:0.75;margin-top:-12px;'>Predictions generated.</p>"
            "</div>"
        )

    elif status == "failed":
        return (
            "<div class='card'>"
            "<div class='metric-title'>Model Status</div>"
            "<h4 style='color:#ff5252;'>❌ Failed</h4>"
            "<p style='opacity:0.75;margin-top:-12px;'>Check logs & retry.</p>"
            "</div>"
        )

    else:
        return "<p>Invalid model status</p>"


# ---------------------------
# LSTM Model (Improved for stability)
# ---------------------------
class Brain(nn.Module):
    
    def __init__(self, input_size=2, hidden_size=256, num_layers=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ---------------------------
# Helpers
# ---------------------------


@st.cache_data(show_spinner=False, ttl=3600)
def download_data(symbol: str, period: str = "5y"):
    TWELVE_DATA_KEY = (
        st.secrets.get("TWELVE_DATA_KEY") or "5a1f3871569543aca6034279f126b3c8"
    )
    if TWELVE_DATA_KEY != "5a1f3871569543aca6034279f126b3c8":
        url = f"https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol.replace(".NS", ""),
            "interval": "1day",
            "outputsize": 1825,  # ~5 years
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


def prepare_return_sequences(close_prices: np.ndarray, seq_len: int = 90):
    # Log returns
    log_returns = np.diff(np.log(close_prices))
    log_returns = np.concatenate([[0.0], log_returns])

    returns_series = pd.Series(log_returns)
    volatility = returns_series.rolling(20).std().fillna(0.0).values

    features = np.column_stack([log_returns, volatility])

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    X, y = [], []
    for i in range(seq_len, len(scaled_features)):
        X.append(scaled_features[i - seq_len : i])
        y.append(scaled_features[i, 0])

    X = np.array(X)
    y = np.array(y)

    last_seq = scaled_features[-seq_len:]

    return X, y, scaler, last_seq, scaled_features.shape[1]


def get_future_dates(last_date, n):
    future_dates = []
    tmp = last_date
    while len(future_dates) < n:
        tmp += timedelta(days=1)
        if tmp.weekday() < 5:
            future_dates.append(tmp)
    return future_dates


# ---------------------------
# Financial Advisor Helpers
# ---------------------------


def generate_advice(pred_prices, latest_price, historical_returns, volatility):

    pred_return_1d = (pred_prices[0] / latest_price - 1) * 100
    pred_return_5d = (
        (pred_prices[4] / latest_price - 1) * 100
        if len(pred_prices) >= 5
        else pred_return_1d
    )
    pred_return_30d = (pred_prices[-1] / latest_price - 1) * 100

    advice = []
    advice.append(f"Short-term (1 day): Expected return {pred_return_1d:+.2f}%. ")
    if pred_return_1d > 2:
        advice.append("Strong buy signal if aligned with your risk tolerance.")
    elif pred_return_1d < -2:
        advice.append("Consider selling or hedging positions.")
    else:
        advice.append("Hold position; minimal movement expected.")

    advice.append(f"Medium-term (5 days): Expected return {pred_return_5d:+.2f}%. ")
    if pred_return_5d > 5:
        advice.append("Positive momentum; potential entry point for swing trades.")
    elif pred_return_5d < -5:
        advice.append("Caution: Downtrend may persist.")

    advice.append(f"Long-term (30 days): Expected return {pred_return_30d:+.2f}%. ")
    if pred_return_30d > 10:
        advice.append("Bullish outlook; consider accumulating shares.")
    elif pred_return_30d < -10:
        advice.append("Bearish; diversify or exit if overexposed.")
    else:
        advice.append("Stable; monitor for external factors.")

    avg_hist_return = float(historical_returns.mean()) * 252 * 100
    advice.append(
        f"Compared to historical annualized return of {avg_hist_return:.2f}%, the forecast suggests {'stronger' if pred_return_30d / 30 * 252 > avg_hist_return else 'weaker'} performance."
    )

    if volatility > 30:
        advice.append("High volatility stock; use stop-loss orders to manage risk.")
    else:
        advice.append("Moderate volatility; suitable for conservative investors.")

    return {
    "1d": pred_return_1d,
    "5d": pred_return_5d,
    "30d": pred_return_30d,
    "comparison": avg_hist_return,
    "volatility": volatility,
    }


def realistic_advisor(lstm_ret):
    if lstm_ret  > 0:
        return "🟢 Bullish bias (models agree)"
    elif lstm_ret < 0 :
        return "🔴 Bearish bias (models agree)"
    else:
        return "🟡 Mixed signals (high uncertainty)"


# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("Controls & Search")
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
            ALGOLIA_APP_ID = st.secrets.get("ALGOLIA_APP_ID")
            ALGOLIA_API_KEY = st.secrets.get("ALGOLIA_API_KEY")
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
                hits = resp.json().get("hits", [])
                company_options = [
                    {"name": h.get("Security Name", "Unknown"), "symbol": h.get("Symbol", "N/A")}
                    for h in hits
                    if h.get("Security Name") and h.get("Symbol")  
                ]
                
                if company_options:
                    display_options = [f"{c['name']} ({c['symbol']})" for c in company_options]
                    current_name = st.session_state.get("company_name")
                    default_index = 0
                    if current_name:
                        try:
                            default_index = next(i for i, c in enumerate(company_options) if c["name"] == current_name)
                        except StopIteration:
                            default_index = 0
                    
                    selected_display = st.selectbox(
                        "Search Results",
                        options=display_options,
                        index=default_index,
                        help="Choose a company to analyze"
                    )
                    
                    selected_index = display_options.index(selected_display)
                    chosen = company_options[selected_index]
                    
                    if (st.session_state.get("company_name") != chosen["name"] or
                        st.session_state.get("company_symbol") != chosen["symbol"]):
                        st.session_state.company_name = chosen["name"]
                        st.session_state.company_symbol = chosen["symbol"]
                        st.rerun()  
                    
                else:
                    st.warning("No companies found for your search.")
            else:
                st.error(f"Search failed (status {resp.status_code})")
                
        except Exception as e:
            st.error(f"Search error: {e}")
    
    if st.session_state.get("company_symbol"):
        symbol = st.session_state["company_symbol"]
        yf_symbol = symbol + ".NS"  
        st.success(f"**Selected:** {st.session_state['company_name']} ({symbol})")
    else:
        st.info("🔍 Start typing a company name to see suggestions...")
        symbol = None
        yf_symbol = None


    days_ahead = st.slider("Predict next how many trading days?", 1, 30, 10)
    retrain_every_run = st.checkbox("Retrain model every run", value=True)
    seq_len = st.number_input(
        "Sequence length (timesteps)", min_value=10, max_value=120, value=60, step=10
    )
    epochs = st.slider("Training epochs", 5, 200, 35)
    st.markdown("---")
    st.caption("Demo trains on CPU. Not financial advice.")

    if "symbol" not in locals() and st.session_state.get("company_symbol"):
        symbol = st.session_state["company_symbol"]
    elif "symbol" not in locals():
        symbol = None
base_sym = symbol.upper() if symbol else ""


def build_tv_candidates(base):
    if not base:
        return [""]

    candidates = []
    candidates.append(f"NSE:{base}")
    candidates.append(f"BSE:{base}")
    candidates.append(f"NASDAQ:{base}")
    candidates.append(f"NYSE:{base}")
    candidates.append(base)
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


tv_candidates = build_tv_candidates(base_sym)

chosen_tv = st.selectbox(
    "TradingView Symbol",
    tv_candidates,
    label_visibility="hidden",
    index=0,
    help="If widget shows 'invalid symbol' try a different candidate here",
)

tv_symbol = chosen_tv
trend_symbol = base_sym

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

run = st.button("Run Prediction", type="primary", width="stretch")
if run:
    st.session_state.show_widgets = True

    with st.spinner("Fetching data..."):
        df = download_data(yf_symbol, period="5y")

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
    close_vals = close.values.astype(np.float32).flatten()
    latest_price = float(close_vals[-1])

    c1, c2, c3, c4 = st.columns(4)
    close_21d = close.iloc[-21].item()
    ret_1m = (latest_price / close_21d - 1) * 100 if len(close) > 21 else 0.0

    returns = close.pct_change().dropna()
    vol = (
        (returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100).item()
        if len(returns) > 20
        else 0.0
    )

    with c1:
        st.markdown(
            f"""
            <div class='card'>
                <div class='metric-title'>1M Return</div>
                <h2>{ret_1m:+.2f}%</h2>
                <p style='
                    color:{"#22c55e" if ret_1m > 0.01 else "#ef4444"}; 
                    background:{"#22c55e0f" if ret_1m > 0.01 else "#ef444421"}; 
                    border-radius: 10px; 
                    padding: 0px 12px; 
                    width: fit-content;
                    margin:7px 0px;
                '
                >
                    {"↑" if ret_1m > 0.01 else "↓"} {abs(ret_1m):.1f}%
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class='card'>
                <div class='metric-title'>Volatility</div>
                <h2>{vol:.1f}%</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class='card'>
                <div class='metric-title'>52W High</div>
                <h2>₹{close.max().item():.2f}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class='card'>
                <div class='metric-title'>Latest Price</div>
                <h2>₹{latest_price:.2f}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    X, y, scaler, last_sequence_scaled, input_size = prepare_return_sequences(
        close_vals, seq_len=seq_len
    )

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
        st.write("Training LSTM on log returns for stable predictions...")

        status_placeholder.markdown(
            render_model_status("training"), unsafe_allow_html=True
        )

        try:
            if retrain_every_run or "trained_model" not in st.session_state:
                st.info("Training LSTM model...")
        
                model = Brain(hidden_size=128, num_layers=2, dropout=0.3)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=0.0005, weight_decay=1e-5
                )
                loss_fn = nn.MSELoss()
        
                dataset = TensorDataset(
                    torch.from_numpy(X).float(), torch.from_numpy(y).float()
                )
                loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
                progress_bar = st.progress(0)
                progress_text = st.empty()
        
                model.train()
                for epoch in range(epochs):
                    epoch_loss = 0.0
                    for bx, by in loader:
                        optimizer.zero_grad()
                        pred = model(bx)
                        loss = loss_fn(pred, by)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
        
                    progress_bar.progress((epoch + 1) / epochs)
                    progress_text.text(
                        f"Epoch {epoch+1}/{epochs} – Loss: {epoch_loss/len(loader):.6f}"
                    )
        
                st.session_state.trained_model = model
                st.success("Model trained successfully!")

            else:
                model = st.session_state.trained_model
                st.success("Using cached LSTM model")
        
            status_placeholder.markdown(
                render_model_status("completed"), unsafe_allow_html=True
            )


        except Exception as e:
            status_placeholder.markdown(
                render_model_status("failed"), unsafe_allow_html=True
            )
            st.error(f"Training failed: {e}")
            st.stop()

        model.eval()
        pred_container = st.container()
        with pred_container:
            st.write("Generating predictions...")
            pred_progress = st.progress(0)
        
        predictions_scaled = []
        current_seq = last_sequence_scaled.copy()  
        
        with torch.no_grad():
            for i in range(days_ahead):
                input_t = torch.from_numpy(current_seq).float().unsqueeze(0)
                pred_scaled = model(input_t).item() 
                
                predictions_scaled.append(pred_scaled)
                
                new_row = np.array([[pred_scaled, 0.0]])
                current_seq = np.vstack([current_seq[1:], new_row])
                
                pred_progress.progress((i + 1) / days_ahead)
        
        pred_progress.progress(1.0)
        pred_container.empty()
        
        pred_padded = np.column_stack([
            np.array(predictions_scaled),
            np.zeros(len(predictions_scaled))
        ])
        pred_returns = scaler.inverse_transform(pred_padded)[:, 0]
        
        pred_prices = [latest_price]
        for r in pred_returns:
            pred_prices.append(pred_prices[-1] * np.exp(r))
        pred_prices = np.array(pred_prices[1:])
        future_dates = get_future_dates(dates[-1], days_ahead)

        # ---------------------------
        # Generate AI Financial Advice
        # ---------------------------


        pred_df = pd.DataFrame(
            {
                "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
                "Predicted Price": pred_prices,
            }
        )

        st.download_button(
            "Download predictions (CSV)",
            data=pred_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{symbol}_predictions.csv",
            mime="text/csv",
        )

    # Chart
    with right:
        st.subheader(f"{symbol} — Price Chart")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=close_vals,
                mode="lines",
                name="Close",
                line=dict(color="#1f77b4"),
            )
        )

        df_ma = pd.DataFrame({"Close": close_vals}, index=dates)
        fig.add_trace(
            go.Scatter(
                x=df_ma.index,
                y=df_ma["Close"].rolling(20).mean(),
                mode="lines",
                name="MA20",
                line=dict(dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_ma.index,
                y=df_ma["Close"].rolling(50).mean(),
                mode="lines",
                name="MA50",
                line=dict(dash="dot"),
            )
        )

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
    latest_pred = float(pred_prices[-1])
    pct_change = (latest_pred - latest_price) / latest_price * 100

    col1.metric("Latest Closing Price", f"₹{latest_price:.2f}")
    col2.metric(
        f"Predicted in {days_ahead} days", f"₹{latest_pred:.2f}", f"{pct_change:+.2f}%"
    )

    display_df = pd.DataFrame(
        {
            "Date": [d.strftime("%b %d, %Y") for d in future_dates],
            "Predicted Price (₹)": [f"{p:.2f}" for p in pred_prices],
        }
    )
    st.table(display_df)


if st.session_state.get("show_widgets", False):

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
            height=380,
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
            height=310,
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
            height=290,
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
            height=1020,
        )
    
    advice = generate_advice(
    pred_prices=pred_prices,
    latest_price=latest_price,
    historical_returns=returns,
    volatility=vol
    )

    st.subheader("📊 AI Financial Advisor")

    def badge(value, pos=2, neg=-2):
        if value > pos:
            return "🟢 BUY", "#22c55e"
        elif value < neg:
            return "🔴 SELL", "#ef4444"
        else:
            return "🟡 HOLD", "#f59e0b"
    
    b1, c1 = badge(advice["1d"])
    b5, c5 = badge(advice["5d"], 5, -5)
    b30, c30 = badge(advice["30d"], 10, -10)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">Short Term (1 Day)</div>
                <h2>{advice["1d"]:+.2f}%</h2>
                <span style="color:{c1}; font-weight:600;">{b1}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">Medium Term (5 Days)</div>
                <h2>{advice["5d"]:+.2f}%</h2>
                <span style="color:{c5}; font-weight:600;">{b5}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col3:
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">Long Term (30 Days)</div>
                <h2>{advice["30d"]:+.2f}%</h2>
                <span style="color:{c30}; font-weight:600;">{b30}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🧠 AI Insight Summary")

    relative_perf = (
        "stronger" if advice["30d"] / 30 * 252 > advice["comparison"] else "weaker"
    )
    
    risk_text = (
        "High volatility — suitable for aggressive traders."
        if advice["volatility"] > 30
        else "Moderate volatility — suitable for conservative investors."
    )
    
    st.markdown(
        f"""
        <div style="padding:30px; border-radius:12px; margin: 10px 7px; box-shadow: 0 6px 18px rgba(0,0,0,0.06); ">
        • 📉 Expected annualized performance appears <b>{relative_perf}</b> compared to historical average<br>
        <br/>
        • 📊 Historical annual return: <b>{advice["comparison"]:.2f}%</b><br>
        <br/>
        • ⚠️ {risk_text}
        </div>
        """,
        unsafe_allow_html=True,
    )



st.caption(
    "LSTM signals reflect short-term patterns. "
)
