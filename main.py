import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
st.set_page_config(page_title="股票分析", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Institutional Look
st.markdown("""
    <style>
    /* Global Background and Text */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161B22 !important;
        border-right: 1px solid #30363D;
    }
    
    /* Card/Container Styling */
    .data-card {
        background: #161B22;
        border: 1px solid #30363D;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transition: all 0.2s ease-in-out;
    }
    .data-card:hover {
        border-color: #58A6FF;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    }
    
    /* Header Customization */
    .main-header {
        font-weight: 800;
        letter-spacing: -0.04em;
        background: linear-gradient(90deg, #58A6FF 0%, #BC8CF2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
        margin-bottom: 10px !important;
    }
    
    /* Metric Styling */
    [data-testid="stMetric"] {
        background: #1C2128;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 15px !important;
        transition: border-color 0.2s;
    }
    [data-testid="stMetric"]:hover {
        border-color: #444C56;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        color: #8B949E !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #161B22;
        border-radius: 8px !important;
        border: 1px solid #30363D;
        color: #8B949E;
        padding: 0 24px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background-color: #58A6FF !important;
        color: #FFFFFF !important;
        border-color: #58A6FF !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background-color: #58A6FF;
    }
    
    /* Status Badges */
    .status-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-thumb {
        background: #30363D;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }

    /* Animation Classes */
    @keyframes slideIn {
        from { width: 0; opacity: 0; }
        to { opacity: 1; }
    }
    .progress-bar-fill {
        animation: slideIn 1s cubic-bezier(0.4, 0, 0.2, 1) forwards;
    }
    </style>
    """, unsafe_allow_html=True)

# Translations - Fixed to Chinese
t = {
    "title": "股票分析",
    "description": "提供專業級別的即時市場情報與深度分析。",
    "settings": "基礎設定",
    "ticker_label": "股票代碼",
    "period_label": "查詢範圍",
    "interval_label": "資料密度",
    "current_price": "目前股價",
    "change": "漲跌幅",
    "volume": "成交量",
    "market_cap": "市值",
    "tab_overview": "📈 即時概況",
    "tab_tech": "📊 技術指標",
    "tab_predict": "🔮 趨勢預測",
    "tab_signals": "🔔 交易信號",
    "price_action": "價格走勢",
    "key_stats": "關鍵統計",
    "no_data": "查無此代碼，請重新輸入 (如 2330.TW)。",
    "trailing_pe": "本益比",
    "forward_pe": "預測本益比",
    "div_yield": "殖利率",
    "high_52w": "52週高點",
    "low_52w": "52週低點",
    "beta": "Beta 係數",
    "prediction_header": "AI 趨勢推估",
    "prediction_days": "推估天數",
    "predicted_price": "推估價格",
    "tech_analysis": "技術指標詳解",
    "signal_label": "信號線",
    "backtest_header": "預測準確度回測",
    "backtest_desc": "使用過去 7 天的數據驗證模型準確性。",
    "target_price": "目標價參考資訊"
}

@st.cache_data
def get_stock_data(ticker, period, interval):
    ticker = ticker.strip().upper()
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

@st.cache_data
def get_financial_data(ticker):
    """Fetches annual financial data for trends."""
    try:
        stock = yf.Ticker(ticker)
        financials = stock.financials
        if financials is not None and not financials.empty:
            # Extract Revenue and Net Income
            data = financials.T[['Total Revenue', 'Net Income']].copy()
            data.index = [d.year for d in data.index]
            return data.sort_index()
        return None
    except Exception:
        return None

def get_signal_score(data, rsi_val, macd_val, signal_val):
    score = 0
    # RSI Signal
    if rsi_val < 30: score += 1  # Oversold
    elif rsi_val > 70: score -= 1 # Overbought
    
    # MACD Signal
    if macd_val > signal_val: score += 1
    else: score -= 1
    
    # Trend Signal (Price vs MA20)
    current_price = float(data['Close'].iloc[-1])
    ma20 = float(data['Close'].rolling(window=20).mean().iloc[-1])
    if current_price > ma20: score += 1
    else: score -= 1
    
    # Return rating and color
    if score >= 2: return "強勢買入", "#00897b", 85
    elif score == 1: return "買入", "#26a69a", 65
    elif score == -1: return "賣出", "#ff5252", 35
    elif score <= -2: return "強勢賣出", "#d32f2f", 15
    return "中性", "#787b86", 50

def get_expert_insight(ticker, price, rsi, rating, macd_val, signal_val, buy_sigs, sell_sigs, current_date):
    # RSI Analysis
    rsi_status = "超買" if rsi > 70 else "超賣" if rsi < 30 else "中性"
    rsi_color = "#ef5350" if rsi > 70 else "#26a69a" if rsi < 30 else "#8B949E"
    rsi_desc = "股價進入超買區，短期回檔風險增加。" if rsi > 70 else "股價進入超賣區，可能存在反彈機會。" if rsi < 30 else "RSI 處於中性區間，走勢相對穩定。"
    
    # MACD Analysis
    macd_diff = macd_val - signal_val
    macd_status = "多頭金叉" if macd_diff > 0 else "空頭死叉"
    macd_color = "#26a69a" if macd_diff > 0 else "#ef5350"
    macd_desc = "快線穿越慢線，短期動能偏多。" if macd_diff > 0 else "快線跌破慢線，短期動能轉弱。"
    
    # Action Advice
    action = "積極買進" if "STRONG BUY" in rating else "建議買進" if "BUY" in rating else "建議放空" if "SELL" in rating else "避開空頭" if "STRONG SELL" in rating else "中性觀望"
    action_color = "#26a69a" if "BUY" in rating else "#ef5350" if "SELL" in rating else "#58A6FF"
    
    # Signal check
    latest_signal = "目前無明確進場信號。"
    # Convert current_date to date for comparison if needed
    if current_date in buy_sigs:
        latest_signal = "🔥 今日觸發【買入信號】，技術面轉強！"
        action = "即刻買進"
        action_color = "#26a69a"
    elif current_date in sell_sigs:
        latest_signal = "⚠️ 今日觸發【賣出信號】，注意獲利了結。"
        action = "即刻賣出"
        action_color = "#ef5350"
    
    return {
        "rsi": {"val": f"{rsi:.1f}", "status": rsi_status, "color": rsi_color, "desc": rsi_desc},
        "macd": {"val": f"{macd_diff:+.2f}", "status": macd_status, "color": macd_color, "desc": macd_desc},
        "action": {"status": action, "color": action_color, "desc": latest_signal}
    }

def create_tv_gauge(score_val, label, color):
    """Creates a TradingView-style gauge chart."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score_val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': label, 'font': {'size': 18, 'color': color}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#434651"},
            'bar': {'color': color},
            'bgcolor': "#1e222d",
            'borderwidth': 2,
            'bordercolor': "#434651",
            'steps': [
                {'range': [0, 20], 'color': '#d32f2f'},
                {'range': [20, 40], 'color': '#ff5252'},
                {'range': [40, 60], 'color': '#787b86'},
                {'range': [60, 80], 'color': '#26a69a'},
                {'range': [80, 100], 'color': '#00897b'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': score_val
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#d1d4dc", 'family': "Inter, sans-serif"},
        height=220,
        margin=dict(l=20, r=20, t=40, b=20),
        transition={'duration': 1000, 'easing': 'elastic-in-out'}
    )
    return fig

def calculate_health_scores(info):
    """Calculates 1-10 scores for financial health based on info data."""
    # 1. Profitability Score (Margins)
    pm = info.get('profitMargins', 0) or 0
    roe = info.get('returnOnEquity', 0) or 0
    p_score = min(10, max(1, int(pm * 20 + roe * 15))) if pm and roe else 5
    
    # 2. Leverage Score (Debt to Equity)
    de = info.get('debtToEquity', 100) or 100
    l_score = min(10, max(1, 10 - int(de / 40))) if de else 8
    
    # 3. Cash Flow Score
    fcf = info.get('freeCashflow', 0) or 0
    rev = info.get('totalRevenue', 1) or 1
    c_score = min(10, max(1, int((fcf / rev) * 30 + 5))) if fcf and rev else 6
    
    return p_score, l_score, c_score

def get_ai_entry_strategy(data, rsi, ema20, ema50, bb_lower, win_prob):
    """Generates AI-driven entry strategy and price."""
    current_price = float(data['Close'].iloc[-1])
    
    # Logic:
    # 1. If Strong Uptrend (EMA20 > EMA50) & RSI not overbought: Entry near EMA20
    # 2. If Downtrend: Entry near BB Lower or 5% below current
    # 3. If High Win Prob: Aggressive entry
    
    if ema20 > ema50:
        if rsi < 60:
            suggested_price = max(ema20, current_price * 0.99)
            confidence = "高"
            action = "拉回買進"
            desc = "趨勢偏多且未過熱，建議於均線附近分批佈局。"
            color = "#3FB950"
        else:
            suggested_price = ema20
            confidence = "中"
            action = "等待回檔"
            desc = "短期漲幅已大，建議等候回測 20 日線再行進場。"
            color = "#D29922"
    else:
        suggested_price = bb_lower if bb_lower < current_price else current_price * 0.95
        confidence = "低"
        action = "保守觀望"
        desc = "目前趨勢偏弱，建議等候布林下軌或支撐確認。"
        color = "#f85149"
        
    # Adjust by Win Probability
    if win_prob > 60:
        confidence = "極高"
        desc = "模擬勝率極佳，可考慮適度提高進場位階。"
        
    return {
        "price": suggested_price,
        "confidence": confidence,
        "action": action,
        "desc": desc,
        "color": color
    }

# Sidebar settings
with st.sidebar:
    st.markdown(f'<h2 style="font-size: 1.2rem; color: #FFFFFF; margin-bottom: 20px;">{t["settings"]}</h2>', unsafe_allow_html=True)
    
    # Actual Ticker Input
    ticker_val = st.session_state.get('ticker_input', "2330.TW")
    ticker_input = st.text_input(
        t["ticker_label"], 
        value=ticker_val, 
        key="main_ticker_input", 
        help="台股請加後綴，例如：2330.TW (上市) 或 8069.TWO (上櫃)；美股直接輸入代碼，例如：AAPL"
    )
    if ticker_input != ticker_val:
        st.session_state.ticker_input = ticker_input
        st.rerun()

    st.markdown("---")
    period = st.selectbox(t["period_label"], options=["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    interval = st.selectbox(t["interval_label"], options=["1d", "1wk", "1mo"], index=0)

    st.markdown("---")
    st.markdown(f'<p style="color: #8B949E; font-size: 0.8rem; font-weight: 600;">{t["prediction_header"]}</p>', unsafe_allow_html=True)
    predict_days = st.slider(t["prediction_days"], 1, 30, 7)

# Header Section
col_title, col_status = st.columns([3, 2])
with col_title:
    st.markdown(f'<h1 class="main-header">{t["title"]}</h1>', unsafe_allow_html=True)
    stock_header_placeholder = st.empty() # For stock name and ticker

with col_status:
    st.markdown(f'''
        <div style="text-align: right; padding-top: 15px;">
            <div style="font-size: 0.85rem; color: #8B949E; margin-bottom: 8px; font-family: monospace;">
                SYNC_TIME: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            </div>
            <div class="status-badge" style="background: rgba(35, 134, 54, 0.15); color: #3FB950; border: 1px solid rgba(63, 185, 80, 0.3);">
                <span style="font-size: 1rem;">●</span> MARKET_CONNECTED
            </div>
        </div>
    ''', unsafe_allow_html=True)

if ticker_input:
    data = get_stock_data(ticker_input, period, interval)
    
    if data is not None and not data.empty:
        stock = yf.Ticker(ticker_input)
        info = stock.info
        p_score, l_score, c_score = calculate_health_scores(info)
        
        # Update header placeholder with stock name and ticker
        stock_name = info.get('shortName') or info.get('longName') or ticker_input
        stock_header_placeholder.markdown(f'<p style="color: #58A6FF; font-size: 1.2rem; font-weight: 600; margin-top: -5px;">{stock_name} <span style="color: #8B949E; font-weight: 400; font-size: 0.9rem;">({ticker_input})</span></p>', unsafe_allow_html=True)

        # Pre-calculations
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        macd_series = exp1 - exp2
        signal_series = macd_series.ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        ma20 = data['Close'].rolling(window=20).mean()
        std20 = data['Close'].rolling(window=20).std()
        bb_upper = ma20 + (std20 * 2)
        bb_lower = ma20 - (std20 * 2)
        
        # EMA
        ema20 = data['Close'].ewm(span=20, adjust=False).mean()
        ema50 = data['Close'].ewm(span=50, adjust=False).mean()
        ema200 = data['Close'].ewm(span=200, adjust=False).mean()
        
        # Calculate Buy/Sell Signals
        buy_signals = []
        sell_signals = []
        for i in range(1, len(data)):
            # Buy: MACD Golden Cross OR RSI recovery from oversold
            macd_gold = macd_series.iloc[i] > signal_series.iloc[i] and macd_series.iloc[i-1] <= signal_series.iloc[i-1]
            rsi_buy = rsi_series.iloc[i] > 30 and rsi_series.iloc[i-1] <= 30
            if macd_gold or rsi_buy:
                buy_signals.append(data.index[i])
            
            # Sell: MACD Death Cross OR RSI pullback from overbought
            macd_death = macd_series.iloc[i] < signal_series.iloc[i] and macd_series.iloc[i-1] >= signal_series.iloc[i-1]
            rsi_sell = rsi_series.iloc[i] < 70 and rsi_series.iloc[i-1] >= 70
            if macd_death or rsi_sell:
                sell_signals.append(data.index[i])

        # Prepare Signal Data for Global Use
        all_signals = []
        for d in buy_signals:
            price_val = float(data.loc[d, 'Close'].iloc[0] if isinstance(data.loc[d, 'Close'], pd.Series) else data.loc[d, 'Close'])
            all_signals.append({"date": d, "type": "買入", "price": price_val, "color": "#26a69a", "icon": "🔼"})
        for d in sell_signals:
            price_val = float(data.loc[d, 'Close'].iloc[0] if isinstance(data.loc[d, 'Close'], pd.Series) else data.loc[d, 'Close'])
            all_signals.append({"date": d, "type": "賣出", "price": price_val, "color": "#ef5350", "icon": "🔽"})
        
        # Sort by date descending
        latest_signals = sorted(all_signals, key=lambda x: x['date'], reverse=True)
        latest_sig = latest_signals[0] if latest_signals else None

        current_price = float(data['Close'].iloc[-1])
        
        # Top Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric(t["current_price"], f"{current_price:.2f}")
        with m2:
            prev_price = float(data['Close'].iloc[-2])
            change = current_price - prev_price
            st.metric(t["change"], f"{change:+.2f}", f"{(change/prev_price*100):+.2f}%")
        with m3:
            volume = float(data['Volume'].iloc[-1])
            st.metric(t["volume"], f"{volume/1e6:.2f}M")
        with m4:
            market_cap = info.get('marketCap', 'N/A')
            if isinstance(market_cap, (int, float)):
                st.metric(t["market_cap"], f"{market_cap/1e12:.2f}T")
            else:
                st.metric(t["market_cap"], "N/A")
        with m5:
            day_high = info.get('dayHigh', 'N/A')
            st.metric("當日最高", f"{day_high}" if isinstance(day_high, (int, float)) else "N/A")

        # Technical Rating Badge
        sig_text, sig_color, sig_val = get_signal_score(data, float(rsi_series.iloc[-1]), float(macd_series.iloc[-1]), float(signal_series.iloc[-1]))
        
        st.markdown("---")
        
        tab1, tab2, tab3, tab4 = st.tabs([t["tab_overview"], t["tab_tech"], t["tab_predict"], t["tab_signals"]])

        with tab1:
            # 1. Quick Recommendation & Prediction Summary
            res = get_expert_insight(ticker_input, current_price, float(rsi_series.iloc[-1]), sig_text, float(macd_series.iloc[-1]), float(signal_series.iloc[-1]), buy_signals, sell_signals, data.index[-1])
            
            # Prediction Calculations (Moved here for quick summary)
            returns = data['Close'].pct_change().dropna()
            avg_ret, vol = returns.mean(), returns.std()
            num_simulations = 1000
            sim_df = pd.DataFrame()
            for i in range(num_simulations):
                prices = [current_price]
                for _ in range(7): # Use 7 days for quick summary
                    prices.append(prices[-1] * (1 + np.random.normal(avg_ret, vol)))
                sim_df[i] = prices
            median_sim_7d = sim_df.median(axis=1).iloc[-1]
            win_prob_7d = (sim_df.iloc[-1, :] > current_price).mean() * 100

            # AI Smart Entry Calculation
            ai_strat = get_ai_entry_strategy(
                data, 
                float(rsi_series.iloc[-1]), 
                float(ema20.iloc[-1]), 
                float(ema50.iloc[-1]), 
                float(bb_lower.iloc[-1]), 
                win_prob_7d
            )

            rec_col1, rec_col2, rec_col3 = st.columns([1, 1, 1])
            with rec_col1:
                st.markdown(f'''
                    <div style="background: rgba(88, 166, 255, 0.1); padding: 15px; border-radius: 8px; border-left: 5px solid {res['action']['color']}; height: 110px;">
                        <div style="color: #8B949E; font-size: 0.75rem; margin-bottom: 5px;">當前操作建議</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: {res['action']['color']};">{res['action']['status']}</div>
                        <div style="color: #E0E0E0; font-size: 0.75rem; overflow: hidden; white-space: nowrap; text-overflow: ellipsis;">{res['action']['desc']}</div>
                    </div>
                ''', unsafe_allow_html=True)
            
            with rec_col2:
                st.markdown(f'''
                    <div style="background: rgba(63, 185, 80, 0.05); padding: 15px; border-radius: 8px; border-left: 5px solid {ai_strat['color']}; height: 110px; border: 1px solid {ai_strat['color']}33;">
                        <div style="color: #8B949E; font-size: 0.75rem; margin-bottom: 5px;">AI 建議買點 (信心:{ai_strat['confidence']})</div>
                        <div style="display: flex; align-items: baseline; gap: 5px;">
                            <span style="font-size: 1.4rem; font-weight: 800; color: {ai_strat['color']};">${ai_strat['price']:.1f}</span>
                            <span style="color: #8B949E; font-size: 0.7rem;">{ai_strat['action']}</span>
                        </div>
                        <div style="color: #E0E0E0; font-size: 0.7rem; margin-top: 2px; line-height: 1.2;">{ai_strat['desc']}</div>
                    </div>
                ''', unsafe_allow_html=True)
            
            with rec_col3:
                st.markdown(f'''
                    <div style="background: rgba(188, 140, 242, 0.1); padding: 15px; border-radius: 8px; border-left: 5px solid #BC8CF2; height: 110px;">
                        <div style="color: #8B949E; font-size: 0.75rem; margin-bottom: 5px;">7日趨勢預測</div>
                        <div style="display: flex; align-items: baseline; gap: 5px;">
                            <span style="font-size: 1.4rem; font-weight: 800; color: #BC8CF2;">${median_sim_7d:.1f}</span>
                            <span style="color: {'#26a69a' if win_prob_7d > 50 else '#ef5350'}; font-size: 0.8rem; font-weight: bold;">({win_prob_7d:.1f}%)</span>
                        </div>
                        <div style="color: #8B949E; font-size: 0.7rem;">蒙地卡羅模擬勝率</div>
                    </div>
                ''', unsafe_allow_html=True)

            # Overview Tab: Advanced Plotly Chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.03, subplot_titles=(f'{ticker_input} 價量走勢', '成交量'), 
                               row_width=[0.2, 0.7])

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], 
                name='K線', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
            ), row=1, col=1)
            
            # MA
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(window=20).mean(), name='20日均線', line=dict(color='#2962FF', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(window=50).mean(), name='50日均線', line=dict(color='#FF6D00', width=1.5)), row=1, col=1)

            # Volume
            colors = ['#26a69a' if row['Close'] >= row['Open'] else '#ef5350' for index, row in data.iterrows()]
            fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='成交量', marker_color=colors, opacity=0.5), row=2, col=1)

            fig.update_layout(
                template="plotly_dark",
                height=600,
                xaxis_rangeslider_visible=False,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#30363D', zeroline=False)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#30363D', zeroline=False)
            
            st.plotly_chart(fig, use_container_width=True)

            # 2. Financial Health & Stats Section
            f_data = get_financial_data(ticker_input)
            perf_html = ""
            if f_data is not None and len(f_data) >= 2:
                latest_year = f_data.index[-1]
                prev_year = f_data.index[-2]
                rev_latest = f_data['Total Revenue'].iloc[-1]
                rev_prev = f_data['Total Revenue'].iloc[-2]
                ni_latest = f_data['Net Income'].iloc[-1]
                ni_prev = f_data['Net Income'].iloc[-2]
                
                rev_growth = (rev_latest / rev_prev - 1) * 100 if rev_prev != 0 else 0
                ni_growth = (ni_latest / ni_prev - 1) * 100 if ni_prev != 0 else 0
                
                perf_html = f"""
                <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #30363D; display: flex; align-items: center; gap: 20px;">
                    <div>
                        <p style="color: #8B949E; font-size: 0.75rem; margin-bottom: 4px;">年度營收 ({latest_year})</p>
                        <span style="font-size: 1rem; font-weight: bold; color: #FFFFFF;">${rev_latest/1e6:,.0f}M</span>
                        <span style="font-size: 0.8rem; color: {'#3FB950' if rev_growth > 0 else '#f85149'}; margin-left: 5px;">{rev_growth:+.1f}% YoY</span>
                    </div>
                    <div style="width: 1px; height: 30px; background: #30363D; align-self: flex-end; margin-bottom: 5px;"></div>
                    <div>
                        <p style="color: #8B949E; font-size: 0.75rem; margin-bottom: 4px;">年度淨利 ({latest_year})</p>
                        <span style="font-size: 1rem; font-weight: bold; color: #FFFFFF;">${ni_latest/1e6:,.0f}M</span>
                        <span style="font-size: 0.8rem; color: {'#3FB950' if ni_growth > 0 else '#f85149'}; margin-left: 5px;">{ni_growth:+.1f}% YoY</span>
                    </div>
                </div>
                """

            col_gauge, col_stats_data = st.columns([1, 2])
            
            with col_gauge:
                st.plotly_chart(create_tv_gauge(sig_val, sig_text, sig_color), use_container_width=True)
            
            with col_stats_data:
                # Define dynamic styling based on health scores
                avg_health = (p_score + l_score + c_score) / 3
                health_border = "#3FB950" if avg_health > 7 else "#f85149" if avg_health < 4 else "#30363D"
                health_bg = "rgba(63, 185, 80, 0.02)" if avg_health > 7 else "rgba(248, 81, 73, 0.02)" if avg_health < 4 else "#161B22"
                
                st.markdown(f'''
                <div class="data-card" style="height: 100%; margin-bottom: 0; border-color: {health_border}; background: {health_bg}; transition: all 0.5s ease;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <h3 style="margin: 0; font-size: 1rem; display: flex; align-items: center; gap: 8px;">
                            <span>{t["key_stats"]}</span>
                            <span style="font-size: 0.7rem; background: {health_border}; color: white; padding: 2px 6px; border-radius: 4px; opacity: 0.8;">
                                {'優質' if avg_health > 7 else '警戒' if avg_health < 4 else '穩健'}
                            </span>
                        </h3>
                        <div style="display: flex; gap: 10px;">
                            <span style="font-size: 0.7rem; color: #8B949E; background: #21262d; padding: 2px 6px; border-radius: 3px; border: 1px solid #30363D;">獲利 {p_score}/10</span>
                            <span style="font-size: 0.7rem; color: #8B949E; background: #21262d; padding: 2px 6px; border-radius: 3px; border: 1px solid #30363D;">槓桿 {l_score}/10</span>
                            <span style="font-size: 0.7rem; color: #8B949E; background: #21262d; padding: 2px 6px; border-radius: 3px; border: 1px solid #30363D;">現金 {c_score}/10</span>
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;">
                        <div>
                            <p style="color: #8B949E; font-size: 0.75rem; margin-bottom: 5px;">估值指標</p>
                            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 3px;"><span>{t['trailing_pe']}</span><span style="font-weight: bold;">{info.get('trailingPE', 'N/A')}</span></div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 3px;"><span>{t['forward_pe']}</span><span style="font-weight: bold;">{info.get('forwardPE', 'N/A')}</span></div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.85rem;"><span>{t['div_yield']}</span><span style="font-weight: bold; color: #26a69a;">{info.get('dividendYield', 0)*100:.2f}%</span></div>
                        </div>
                        <div>
                            <p style="color: #8B949E; font-size: 0.75rem; margin-bottom: 5px;">價格區間</p>
                            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 3px;"><span>52週高</span><span style="font-weight: bold;">{info.get('fiftyTwoWeekHigh', 'N/A')}</span></div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 3px;"><span>52週低</span><span style="font-weight: bold;">{info.get('fiftyTwoWeekLow', 'N/A')}</span></div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.85rem;"><span>Beta</span><span style="font-weight: bold;">{info.get('beta', 'N/A')}</span></div>
                        </div>
                        <div>
                            <p style="color: #8B949E; font-size: 0.75rem; margin-bottom: 8px;">法人目標分析</p>
                            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 3px;">
                                <span>目標中位</span>
                                <span style="font-weight: bold; color: #58A6FF;">{info.get('targetMedianPrice', 'N/A')}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 8px;">
                                <span>潛在空間</span>
                                <span style="font-weight: bold; color: #26a69a;">
                                    {f"+{((info.get('targetMedianPrice', current_price) / current_price) - 1) * 100:.1f}%" if info.get('targetMedianPrice') else 'N/A'}
                                </span>
                            </div>
                            <!-- Range Visualizer -->
                            <div style="position: relative; width: 100%; height: 24px; background: #21262d; border-radius: 4px; margin-bottom: 8px; border: 1px solid #30363D; overflow: hidden;">
                                <div style="position: absolute; left: 0; top: 0; height: 100%; width: 100%; background: linear-gradient(90deg, #ef5350 0%, #787b86 50%, #26a69a 100%); opacity: 0.1;"></div>
                                <div style="position: absolute; 
                                            left: {min(100, max(0, ((current_price - info.get('targetLowPrice', current_price*0.8)) / (info.get('targetHighPrice', current_price*1.2) - info.get('targetLowPrice', current_price*0.8)) * 100))) if info.get('targetHighPrice') else 50}%; 
                                            top: 0; width: 3px; height: 100%; background: #FFFFFF; box-shadow: 0 0 8px #FFFFFF; z-index: 2;"></div>
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.65rem; color: #8B949E;">
                                <span>低: {info.get('targetLowPrice', 'N/A')}</span>
                                <span>高: {info.get('targetHighPrice', 'N/A')}</span>
                            </div>
                        </div>
                    </div>
                    {perf_html}
                </div>
                ''', unsafe_allow_html=True)

            # Fixed Expert Insight Card at the bottom
            stock_name = info.get('shortName') or info.get('longName') or ticker_input
            res = get_expert_insight(ticker_input, current_price, float(rsi_series.iloc[-1]), sig_text, float(macd_series.iloc[-1]), float(signal_series.iloc[-1]), buy_signals, sell_signals, data.index[-1])
            
            # Professional Diagnostic Section with Grid Layout
            st.markdown(f'''
<div class="data-card" style="border-top: 2px solid {res['action']['color']}; border-color: {res['action']['color']}44; background: linear-gradient(180deg, {res['action']['color']}0a 0%, #0D1117 100%); transition: all 0.5s ease;">
    <h3 style="margin-top:0; color: {res['action']['color']}; font-size: 1.1rem; display: flex; align-items: center; gap: 8px; margin-bottom: 20px;">
        <span style="font-size: 1.4rem;">⚖️</span> {stock_name} - 專家技術診斷報告
    </h3>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
        <div style="background: rgba(48, 54, 61, 0.2); padding: 15px; border-radius: 6px; border-left: 3px solid {res['rsi']['color']};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="color: #8B949E; font-size: 0.8rem;">RSI 強度</span>
                <span style="background: {res['rsi']['color']}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem; font-weight: bold;">{res['rsi']['status']}</span>
            </div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #FFFFFF; margin-bottom: 5px;">{res['rsi']['val']}</div>
            <div style="color: #E0E0E0; font-size: 0.85rem; line-height: 1.4;">{res['rsi']['desc']}</div>
        </div>
        <div style="background: rgba(48, 54, 61, 0.2); padding: 15px; border-radius: 6px; border-left: 3px solid {res['macd']['color']};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="color: #8B949E; font-size: 0.8rem;">MACD 差值</span>
                <span style="background: {res['macd']['color']}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem; font-weight: bold;">{res['macd']['status']}</span>
            </div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #FFFFFF; margin-bottom: 5px;">{res['macd']['val']}</div>
            <div style="color: #E0E0E0; font-size: 0.85rem; line-height: 1.4;">{res['macd']['desc']}</div>
        </div>
        <div style="background: rgba(88, 166, 255, 0.1); padding: 15px; border-radius: 6px; border: 1px dashed #58A6FF;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="color: #58A6FF; font-size: 0.8rem; font-weight: bold;">AI 建議佈局</span>
                <span style="color: {ai_strat['color']}; font-weight: 800; font-size: 0.9rem;">{ai_strat['action']} (${ai_strat['price']:.1f})</span>
            </div>
            <div style="font-size: 1rem; color: #FFFFFF; font-weight: 500; margin-bottom: 5px;">策略部署 (信心:{ai_strat['confidence']})</div>
            <div style="color: #E0E0E0; font-size: 0.85rem; line-height: 1.4;">{ai_strat['desc']}</div>
        </div>
    </div>
</div>
''', unsafe_allow_html=True)

        with tab2:
            st.markdown(f'''
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px;">
    <div class="data-card" style="margin-bottom: 0; padding: 15px; border-left: 3px solid #58A6FF;">
        <div style="color: #8B949E; font-size: 0.75rem;">EMA 趨勢狀態</div>
        <div style="font-size: 1.1rem; font-weight: bold; margin: 5px 0;">{'多頭排列' if ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1] else '趨勢不明' if ema20.iloc[-1] > ema50.iloc[-1] else '空頭趨勢'}</div>
        <div style="color: #26a69a; font-size: 0.7rem;">EMA200 支撐: {ema200.iloc[-1]:.2f}</div>
    </div>
    <div class="data-card" style="margin-bottom: 0; padding: 15px; border-left: 3px solid #BC8CF2;">
        <div style="color: #8B949E; font-size: 0.75rem;">布林帶寬 (BBW)</div>
        <div style="font-size: 1.1rem; font-weight: bold; margin: 5px 0;">{((bb_upper.iloc[-1] - bb_lower.iloc[-1]) / ma20.iloc[-1] * 100):.2f}%</div>
        <div style="color: #8B949E; font-size: 0.7rem;">{'波動擠壓中' if (bb_upper.iloc[-1] - bb_lower.iloc[-1]) < (bb_upper.rolling(100).mean().iloc[-1] - bb_lower.rolling(100).mean().iloc[-1]) else '波動擴張中'}</div>
    </div>
    <div class="data-card" style="margin-bottom: 0; padding: 15px; border-left: 3px solid #26a69a;">
        <div style="color: #8B949E; font-size: 0.75rem;">RSI 乖離率</div>
        <div style="font-size: 1.1rem; font-weight: bold; margin: 5px 0;">{rsi_series.iloc[-1]:.1f}</div>
        <div style="color: {'#ef5350' if rsi_series.iloc[-1] > 70 else '#26a69a' if rsi_series.iloc[-1] < 30 else '#8B949E'}; font-size: 0.7rem;">
            {'超買區域' if rsi_series.iloc[-1] > 70 else '超賣區域' if rsi_series.iloc[-1] < 30 else '中性區間'}
        </div>
    </div>
</div>
''', unsafe_allow_html=True)

            # Combined Price + EMA + BB Chart
            fig_tech = go.Figure()
            
            # Bollinger Bands Area
            fig_tech.add_trace(go.Scatter(x=data.index, y=bb_upper, line=dict(color='rgba(88, 166, 255, 0.2)', width=0), showlegend=False))
            fig_tech.add_trace(go.Scatter(x=data.index, y=bb_lower, line=dict(color='rgba(88, 166, 255, 0.2)', width=0), fill='tonexty', fillcolor='rgba(88, 166, 255, 0.05)', name='布林通道'))
            
            # Price
            fig_tech.add_trace(go.Scatter(x=data.index, y=data['Close'], name='收盤價', line=dict(color='#FFFFFF', width=2)))
            
            # EMAs
            fig_tech.add_trace(go.Scatter(x=data.index, y=ema20, name='EMA 20', line=dict(color='#58A6FF', width=1, dash='dot')))
            fig_tech.add_trace(go.Scatter(x=data.index, y=ema50, name='EMA 50', line=dict(color='#BC8CF2', width=1, dash='dot')))
            fig_tech.add_trace(go.Scatter(x=data.index, y=ema200, name='EMA 200', line=dict(color='#FF6D00', width=1.5)))

            fig_tech.update_layout(
                title="EMA 趨勢與布林通道分析",
                template="plotly_dark", height=450,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_tech, use_container_width=True)

            col_rsi, col_macd = st.columns(2)
            with col_rsi:
                # RSI Chart
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=data.index, y=rsi_series, name='RSI', line=dict(color='#BC8CF2', width=2)))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ef5350", opacity=0.5)
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="#26a69a", opacity=0.5)
                fig_rsi.update_layout(
                    title="RSI (14)", yaxis_range=[0, 100], height=250, template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            with col_macd:
                # MACD Chart
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=data.index, y=macd_series, name='MACD', line=dict(color='#58A6FF', width=1.5)))
                fig_macd.add_trace(go.Scatter(x=data.index, y=signal_series, name='Signal', line=dict(color='#FF6D00', width=1.5)))
                hist_colors = ['#26a69a' if val >= 0 else '#ef5350' for val in (macd_series - signal_series)]
                fig_macd.add_bar(x=data.index, y=macd_series - signal_series, name='Histogram', marker_color=hist_colors, opacity=0.7)
                fig_macd.update_layout(
                    title="MACD", height=250, template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=10, r=10, t=40, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_macd, use_container_width=True)

        with tab3:
            # Prediction Page: Enhanced Monte Carlo & Forecast
            st.markdown(f'''
<div class="data-card">
    <h3 style="margin-top:0; color: #58A6FF; display: flex; align-items: center; gap: 10px;">
        <span>🔮 未來路徑機率模擬</span>
        <span style="font-size: 0.8rem; background: #238636; color: white; padding: 2px 8px; border-radius: 4px; font-weight: normal;">蒙地卡羅演算法</span>
    </h3>
''', unsafe_allow_html=True)
            
            # Calculations
            returns = data['Close'].pct_change().dropna()
            avg_ret, vol = returns.mean(), returns.std()
            prediction_dates = [data.index[-1] + timedelta(days=i) for i in range(predict_days + 1)]
            
            num_simulations = 1000
            sim_df = pd.DataFrame()
            for i in range(num_simulations):
                prices = [current_price]
                for _ in range(predict_days):
                    prices.append(prices[-1] * (1 + np.random.normal(avg_ret, vol)))
                sim_df[i] = prices
            
            p5 = sim_df.quantile(0.05, axis=1)
            p25 = sim_df.quantile(0.25, axis=1)
            p75 = sim_df.quantile(0.75, axis=1)
            p95 = sim_df.quantile(0.95, axis=1)
            median_sim = sim_df.median(axis=1)
            final_prices = sim_df.iloc[-1, :]
            win_prob = (final_prices > current_price).mean() * 100
            
            # Layout for Analysis
            col_mc_chart, col_prob_text = st.columns([2, 1])
            
            with col_mc_chart:
                fig_mc = go.Figure()
                
                # Confidence intervals (Fan Chart style) - Simplified
                # 95% Band
                fig_mc.add_trace(go.Scatter(x=prediction_dates, y=p95, line=dict(width=0), showlegend=False, hoverinfo='skip'))
                fig_mc.add_trace(go.Scatter(
                    x=prediction_dates, y=p5, fill='tonexty', 
                    fillcolor='rgba(88, 166, 255, 0.05)', line=dict(width=0), name='機率區間 (95%)'
                ))
                
                # 50% Band
                fig_mc.add_trace(go.Scatter(x=prediction_dates, y=p75, line=dict(width=0), showlegend=False, hoverinfo='skip'))
                fig_mc.add_trace(go.Scatter(
                    x=prediction_dates, y=p25, fill='tonexty', 
                    fillcolor='rgba(88, 166, 255, 0.15)', line=dict(width=0), name='核心區間 (50%)'
                ))
                
                # Median path with glow effect
                fig_mc.add_trace(go.Scatter(
                    x=prediction_dates, y=median_sim, name='預期路徑 (中位數)', 
                    line=dict(color='#58A6FF', width=4)
                ))
                
                fig_mc.update_layout(
                    title=dict(text=f"未來 {predict_days} 天股價路徑機率預測", font=dict(size=16, color="#FFFFFF")),
                    template="plotly_dark", height=400,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=10, r=10, t=50, b=10),
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
                    transition={'duration': 800, 'easing': 'cubic-in-out'}
                )
                fig_mc.update_xaxes(showgrid=False, zeroline=False)
                fig_mc.update_yaxes(showgrid=True, gridcolor='rgba(48, 54, 61, 0.5)', zeroline=False)
                st.plotly_chart(fig_mc, use_container_width=True)

            with col_prob_text:
                # Calculate Probabilities for specific price levels
                target_3pct = current_price * 1.03
                target_5pct = current_price * 1.05
                target_minus_5pct = current_price * 0.95
                
                prob_3pct = (final_prices > target_3pct).mean() * 100
                prob_5pct = (final_prices > target_5pct).mean() * 100
                prob_minus_5pct = (final_prices < target_minus_5pct).mean() * 100
                
                # Define dynamic styling based on win probability
                mc_card_border = "#3FB950" if win_prob > 55 else "#f85149" if win_prob < 45 else "#30363D"
                mc_card_shadow = "0 0 20px rgba(63, 185, 80, 0.1)" if win_prob > 55 else "0 0 20px rgba(248, 81, 73, 0.1)" if win_prob < 45 else "none"
                mc_card_bg = "rgba(63, 185, 80, 0.02)" if win_prob > 55 else "rgba(248, 81, 73, 0.02)" if win_prob < 45 else "rgba(255, 255, 255, 0.03)"
                
                st.markdown(f'''
<div style="background: {mc_card_bg}; padding: 20px; border-radius: 12px; border: 1px solid {mc_card_border}; height: 400px; box-shadow: {mc_card_shadow}; transition: all 0.5s ease;">
<h4 style="margin-top: 0; color: #FFFFFF; font-size: 1rem; margin-bottom: 20px; display: flex; align-items: center; gap: 8px;">
<span>🎯 到達特定價格機率</span>
<span style="font-size: 0.7rem; background: {mc_card_border}; color: white; padding: 2px 6px; border-radius: 4px; opacity: 0.8;">
{'看多' if win_prob > 55 else '看空' if win_prob < 45 else '中性'}
</span>
</h4>
<div style="margin-bottom: 25px;">
<div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
<span style="color: #8B949E; font-size: 0.85rem;">上漲超過 +3% (${target_3pct:.1f})</span>
<span style="color: {'#3FB950' if prob_3pct > 30 else '#26a69a'}; font-weight: bold;">{prob_3pct:.1f}%</span>
</div>
<div style="width: 100%; height: 6px; background: #21262d; border-radius: 3px;">
<div class="progress-bar-fill" style="width: {prob_3pct}%; height: 100%; background: {'#3FB950' if prob_3pct > 30 else '#26a69a'}; box-shadow: {'0 0 10px rgba(63, 185, 80, 0.4)' if prob_3pct > 40 else 'none'}; border-radius: 3px; transition: all 0.3s;"></div>
</div>
</div>
<div style="margin-bottom: 25px;">
<div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
<span style="color: #8B949E; font-size: 0.85rem;">上漲超過 +5% (${target_5pct:.1f})</span>
<span style="color: {'#3FB950' if prob_5pct > 20 else '#26a69a'}; font-weight: bold;">{prob_5pct:.1f}%</span>
</div>
<div style="width: 100%; height: 6px; background: #21262d; border-radius: 3px;">
<div class="progress-bar-fill" style="width: {prob_5pct}%; height: 100%; background: {'#3FB950' if prob_5pct > 20 else '#26a69a'}; opacity: {0.4 + (prob_5pct/100)}; box-shadow: {'0 0 10px rgba(63, 185, 80, 0.5)' if prob_5pct > 30 else 'none'}; border-radius: 3px; transition: all 0.3s;"></div>
</div>
</div>
<div style="margin-bottom: 25px;">
<div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
<span style="color: #8B949E; font-size: 0.85rem;">維持平盤以上 (${current_price:.1f})</span>
<span style="color: {'#58A6FF' if win_prob > 50 else '#8B949E'}; font-weight: bold;">{win_prob:.1f}%</span>
</div>
<div style="width: 100%; height: 6px; background: #21262d; border-radius: 3px;">
<div class="progress-bar-fill" style="width: {win_prob}%; height: 100%; background: {'#58A6FF' if win_prob > 50 else '#30363D'}; box-shadow: {'0 0 10px rgba(88, 166, 255, 0.4)' if win_prob > 60 else 'none'}; border-radius: 3px; transition: all 0.3s;"></div>
</div>
</div>
<div>
<div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
<span style="color: #8B949E; font-size: 0.85rem;">跌破 -5% (${target_minus_5pct:.1f})</span>
<span style="color: {'#f85149' if prob_minus_5pct > 20 else '#ef5350'}; font-weight: bold;">{prob_minus_5pct:.1f}%</span>
</div>
<div style="width: 100%; height: 6px; background: #21262d; border-radius: 3px;">
<div class="progress-bar-fill" style="width: {prob_minus_5pct}%; height: 100%; background: {'#f85149' if prob_minus_5pct > 20 else '#ef5350'}; opacity: {0.5 + (prob_minus_5pct/100)}; box-shadow: {'0 0 10px rgba(248, 81, 73, 0.5)' if prob_minus_5pct > 25 else 'none'}; border-radius: 3px; transition: all 0.3s;"></div>
</div>
</div>
</div>
''', unsafe_allow_html=True)

            # Metrics for Prediction
            st.markdown('<p style="color: #FFFFFF; font-size: 1rem; font-weight: bold; margin-bottom: 15px;">📊 模擬結果深度解析</p>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("最可能成交價", f"${median_sim.iloc[-1]:.2f}", f"{((median_sim.iloc[-1]/current_price)-1)*100:+.2f}%", help="這是 1000 次模擬中的中位數，代表最有可能發生的結果。")
            with m2:
                st.metric("上漲機率", f"{win_prob:.1f}%", delta=f"{win_prob-50:.1f}%", delta_color="normal", help="在 1000 次模擬中，股價高於目前價格的次數佔比。")
            with m3:
                st.metric("樂觀預期 (前 5%)", f"${p95.iloc[-1]:.2f}", help="在極端樂觀的情況下，股價有 5% 的機率會突破此價位。")
            with m4:
                st.metric("悲觀預期 (後 5%)", f"${p5.iloc[-1]:.2f}", help="在極端悲觀的情況下，股價有 95% 的機率會維持在此價位之上。")
            
            # Risk/Reward Analysis
            st.markdown(f'''
<div style="background: rgba(88, 166, 255, 0.05); padding: 15px; border-radius: 6px; border: 1px solid #30363D; margin-top: 15px;">
    <h4 style="margin-top: 0; font-size: 0.9rem; color: #58A6FF; display: flex; align-items: center; gap: 8px;">
        <span>💡 投資策略參考</span>
    </h4>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
        <div>
            <div style="color: #8B949E; font-size: 0.75rem;">潛在獲利空間</div>
            <div style="color: #26a69a; font-weight: bold; font-size: 1.1rem;">+{((p95.iloc[-1]/current_price)-1)*100:.2f}%</div>
            <div style="color: #8B949E; font-size: 0.7rem; margin-top: 4px;">若市場走勢強勁，理論上的獲利上限。</div>
        </div>
        <div>
            <div style="color: #8B949E; font-size: 0.75rem;">潛在回檔風險</div>
            <div style="color: #ef5350; font-weight: bold; font-size: 1.1rem;">{((p5.iloc[-1]/current_price)-1)*100:.2f}%</div>
            <div style="color: #8B949E; font-size: 0.7rem; margin-top: 4px;">若市場轉弱，應注意的可能跌幅空間。</div>
        </div>
        <div>
            <div style="color: #8B949E; font-size: 0.75rem;">風報比 (勝率權衡)</div>
            <div style="color: #FFFFFF; font-weight: bold; font-size: 1.1rem;">{abs(((p95.iloc[-1]/current_price)-1) / ((p5.iloc[-1]/current_price)-1)):.2f}</div>
            <div style="color: #8B949E; font-size: 0.7rem; margin-top: 4px;">每承擔 1 元風險可獲得的預期回報。數值大於 1 代表期望值較佳。</div>
        </div>
    </div>
</div>
''', unsafe_allow_html=True)

            st.markdown("---")
            st.subheader(t["backtest_header"])
            if len(data) > 20:
                train, actual = data.iloc[:-7], data.iloc[-7:]
                bt_avg = float(train['Close'].tail(10).pct_change().mean().iloc[0] if isinstance(train['Close'].tail(10).pct_change().mean(), pd.Series) else train['Close'].tail(10).pct_change().mean())
                bt_start = float(train['Close'].iloc[-1])
                bt_pred_val = bt_start * (1 + bt_avg)**7
                actual_val = float(actual['Close'].iloc[-1])
                acc = 100 - (abs(bt_pred_val - actual_val) / actual_val * 100)
                
                bc1, bc2 = st.columns(2)
                bc1.metric("回測準確度", f"{acc:.1f}%")
                bc2.metric("7天前預測誤差", f"${abs(bt_pred_val - actual_val):.2f}")
                st.caption(t["backtest_desc"])

            st.markdown('</div>', unsafe_allow_html=True)

        with tab4:
            # Enhanced Signal Monitoring Page
            st.markdown(f'<div class="data-card"><h3 style="margin-top:0; color: #FFD700;">🔔 交易信號監測報告</h3>', unsafe_allow_html=True)
            
            if not latest_signals:
                st.info("目前技術指標尚未偵測到明確的買賣信號。")
            else:
                col_sig_list, col_sig_stats = st.columns([2, 1])
                
                with col_sig_list:
                    st.markdown('<p style="color: #8B949E; font-size: 0.9rem; margin-bottom: 20px; font-weight: 500;">📅 近期觸發信號流水線 (最新 10 筆)</p>', unsafe_allow_html=True)
                    
                    # Start Timeline Container
                    timeline_html = '<div style="position: relative; padding-left: 20px; border-left: 2px solid #30363D; margin-left: 10px;">'
                    
                    for i, s in enumerate(latest_signals[:10]):
                        # Calculate relative days
                        days_ago = (datetime.now().date() - s['date'].date()).days
                        time_str = "今天" if days_ago == 0 else f"{days_ago} 天前"
                        
                        # Dot on the timeline
                        timeline_html += f'''
<div style="position: relative; margin-bottom: 25px;">
    <div style="position: absolute; left: -27px; top: 15px; width: 12px; height: 12px; background: {s['color']}; border-radius: 50%; border: 3px solid #0D1117; box-shadow: 0 0 8px {s['color']}66; z-index: 2;"></div>
    <div style="background: linear-gradient(90deg, {s['color']}0d 0%, rgba(255,255,255,0.02) 100%); border: 1px solid #30363D; border-radius: 12px; padding: 15px 20px; transition: all 0.3s ease; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <div style="display: flex; align-items: center; gap: 15px;">
            <div style="background: {s['color']}22; width: 45px; height: 45px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; border: 1px solid {s['color']}44;">
                {s['icon']}
            </div>
            <div>
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                    <span style="color: {s['color']}; font-weight: 800; font-size: 1.1rem; letter-spacing: 0.5px;">{s['type']}建議</span>
                    <span style="background: #21262d; color: #8B949E; font-size: 0.65rem; padding: 2px 6px; border-radius: 4px; border: 1px solid #30363D;">{time_str}</span>
                </div>
                <div style="color: #8B949E; font-size: 0.8rem; display: flex; align-items: center; gap: 5px;">
                    <span style="opacity: 0.7;">🕒</span> {s['date'].strftime('%Y-%m-%d')}
                </div>
            </div>
        </div>
        <div style="text-align: right;">
            <div style="color: #FFFFFF; font-weight: 800; font-size: 1.3rem; margin-bottom: 2px;">${s['price']:.2f}</div>
            <div style="color: #8B949E; font-size: 0.7rem; font-weight: 500;">建議成交參考價</div>
        </div>
    </div>
</div>
'''
                    
                    timeline_html += '</div>'
                    st.markdown(timeline_html, unsafe_allow_html=True)
                
                with col_sig_stats:
                    st.markdown('<p style="color: #8B949E; font-size: 0.9rem; margin-bottom: 15px;">信號統計摘要</p>', unsafe_allow_html=True)
                    buy_count = len([s for s in latest_signals if s['type'] == "買入"])
                    sell_count = len([s for s in latest_signals if s['type'] == "賣出"])
                    
                    st.markdown(f'''
                    <div style="background: #161B22; border: 1px solid #30363D; padding: 20px; border-radius: 8px;">
                        <div style="margin-bottom: 20px;">
                            <div style="color: #8B949E; font-size: 0.8rem; margin-bottom: 5px;">累計買入信號</div>
                            <div style="color: #26a69a; font-size: 1.8rem; font-weight: bold;">{buy_count}</div>
                        </div>
                        <div style="margin-bottom: 20px;">
                            <div style="color: #8B949E; font-size: 0.8rem; margin-bottom: 5px;">累計賣出信號</div>
                            <div style="color: #ef5350; font-size: 1.8rem; font-weight: bold;">{sell_count}</div>
                        </div>
                        <div style="padding-top: 15px; border-top: 1px solid #30363D;">
                            <div style="color: #8B949E; font-size: 0.8rem; margin-bottom: 10px;">當前市場情緒</div>
                            <div style="color: {'#26a69a' if buy_count > sell_count else '#ef5350' if sell_count > buy_count else '#8B949E'}; font-weight: bold;">
                                {'多頭佔優' if buy_count > sell_count else '空頭佔優' if sell_count > buy_count else '勢均力敵'}
                            </div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    st.warning("⚠️ 信號僅供技術參考，不構成投資建議。請務必結合基本面分析並設定停損點。")

            st.markdown('</div>', unsafe_allow_html=True)

        # Footer
        st.markdown("---")
        st.markdown(
            '<div style="text-align: center; color: #8B949E; font-size: 0.7rem; padding: 20px;">'
            'INSTITUTIONAL TERMINAL v2.0 | REAL-TIME DATA VIA YAHOO FINANCE<br>'
            '© 2026 Financial Analytics Group. All rights reserved. For professional use only.'
            '</div>', 
            unsafe_allow_html=True
        )
    else:
        st.warning(t["no_data"])