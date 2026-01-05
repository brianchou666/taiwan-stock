import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(page_title="Taiwan Stock Market Analysis", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #2e3140;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        min-height: 150px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        line-height: 1.2;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem !important;
        margin-top: 5px;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #BDC3C7 !important;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #2e3140;
        background-color: #1e2130;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e2130;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e5d6c;
    }
    h1, h2, h3, h4 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #f0f2f6;
        margin-top: -10px; /* Reduce top margin to move headers up */
    }
    .centered-header {
        text-align: center;
        width: 100%;
        display: block;
        margin: 20px 0;
    }
    .metric-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #2e3140;
        border-left: 5px solid #1ABC9C;
    }
    .ai-card {
        background: linear-gradient(135deg, #2c3e50 0%, #000000 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #3498db;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
        margin-top: 10px;
    }
    .ai-badge {
        background-color: #3498db;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: bold;
        text-transform: uppercase;
        vertical-align: middle;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Translations
translations = {
    "English": {
        "title": "TW Stock Analysis",
        "description": "Real-time TWSE analytics dashboard dashboard.",
        "settings": "Settings",
        "lang_label": "Language",
        "ticker_label": "Stock Ticker",
        "period_label": "History",
        "interval_label": "Interval",
        "current_price": "Price",
        "change": "Change",
        "volume": "Volume",
        "market_cap": "Market Cap",
        "tab_overview": "📈 Overview",
        "tab_tech": "📊 Technicals",
        "tab_predict": "🔮 Prediction",
        "tab_raw": "📋 Data",
        "price_action": "Price Action",
        "key_stats": "Market Stats",
        "raw_data": "Raw Market Data",
        "no_data": "Ticker not found or Rate Limited.",
        "trailing_pe": "PE Ratio",
        "forward_pe": "Fwd PE",
        "div_yield": "Dividend",
        "high_52w": "52W High",
        "low_52w": "52W Low",
        "beta": "Beta",
        "prediction_header": "Future Forecast",
        "prediction_days": "Days",
        "predicted_price": "Forecasted Price",
        "prediction_desc": "Trend-based estimate. Not financial advice.",
        "mc_expected": "Expected",
        "mc_optimistic": "Optimistic (95%)",
        "mc_pessimistic": "Pessimistic (5%)",
        "tech_analysis": "Technical Indicators",
        "rsi_label": "RSI (Momentum)",
        "macd_label": "MACD (Trend)",
        "signal_label": "Signal",
        "overbought": "Overbought",
        "oversold": "Oversold",
        "backtest_header": "Model Accuracy",
        "backtest_desc": "Checking accuracy using the last 7 days.",
        "actual_price": "Actual",
        "predicted_price_bt": "Predicted",
        "accuracy_score": "Score",
        "error_mae": "Error (Avg)",
        "backtest_wait": "Analysing...",
        "analyst_rating": "Analyst Rating",
        "target_price": "Target Price",
        "target_high": "High Target",
        "target_low": "Low Target",
        "target_mean": "Avg Target",
        "recommendation": "Recommendation",
        "bb_label": "Bollinger Bands",
        "atr_label": "Volatility (ATR)",
        "relative_performance": "Performance vs TAIEX",
        "market_sentiment": "Market Sentiment",
        "overweight": "Outperforming Index",
        "underweight": "Underperforming Index",
        "health_score": "Financial Health Score",
        "profitability": "Profitability",
        "solvency": "Solvency",
        "efficiency": "Efficiency",
        "support_level": "Support",
        "resistance_level": "Resistance",
        "buy_suggestion": "Strategy",
        "buy_point": "Suggested Buy",
        "neutral": "Neutral",
        "strategy_desc": "Combined suggestion based on technicals and levels.",
        "tab_strategy": "⚖️ Strategy BT",
        "win_rate": "Win Rate",
        "total_return": "Total Return",
        "max_drawdown": "Max Drawdown",
        "sharpe_ratio": "Sharpe Ratio",
        "equity_curve": "Equity Curve",
        "signal_buy": "Buy Signal",
        "signal_sell": "Sell Signal",
        "entry_price": "Entry Price",
        "stop_loss": "Stop Loss (SL)",
        "take_profit": "Take Profit (TP)",
        "risk_management": "Risk Management",
        "pos_size": "Suggested Size",
        "ai_analysis": "AI Intelligence Analysis",
        "ai_score": "AI Confidence Score",
        "ai_reasoning": "AI Reasoning Path",
        "ai_summary": "Summary & Action",
        "signal_history": "Signal History",
        "risk_calc_desc": "Based on 1M capital, 2% risk per trade.",
        "no_signals": "No recent signals.",
        "bt_title": "7-Day Backtest Accuracy Check",
        "rate_limit_msg": "Too many requests. Please wait a few minutes and try again."
    },
    "繁體中文": {
        "title": "台股分析助手",
        "description": "即時、簡單、專業的股市分析工具。",
        "settings": "基礎設定",
        "lang_label": "語言",
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
        "tab_raw": "📋 原始數據",
        "price_action": "價格走勢",
        "key_stats": "關鍵統計",
        "raw_data": "歷史數據明細",
        "no_data": "查無此代碼或請求過於頻繁。",
        "trailing_pe": "本益比",
        "forward_pe": "預測本益比",
        "div_yield": "殖利率",
        "high_52w": "52週高點",
        "low_52w": "52週低點",
        "beta": "Beta 係數",
        "prediction_header": "AI 趨勢推估",
        "prediction_days": "推估天數",
        "predicted_price": "推估價格",
        "prediction_desc": "基於近期趨勢計算，僅供參考。",
        "mc_expected": "預期走勢",
        "mc_optimistic": "樂觀估計 (95%)",
        "mc_pessimistic": "保守估計 (5%)",
        "tech_analysis": "技術指標詳解",
        "rsi_label": "RSI (強弱指標)",
        "macd_label": "MACD (趨勢指標)",
        "signal_label": "信號線",
        "overbought": "超買區",
        "oversold": "超賣區",
        "backtest_header": "預測準確度回測",
        "backtest_desc": "使用過去 7 天的數據驗證模型準確性。",
        "actual_price": "實際股價",
        "predicted_price_bt": "預測股價",
        "accuracy_score": "預測準確度",
        "error_mae": "平均誤差",
        "backtest_wait": "計算中...",
        "analyst_rating": "機構評價與目標價",
        "target_price": "目標價",
        "target_high": "最高預估",
        "target_low": "最低預估",
        "target_mean": "平均預估",
        "recommendation": "投資建議",
        "bb_label": "布林通道 (Bollinger Bands)",
        "atr_label": "市場波動率 (ATR)",
        "relative_performance": "相較大盤 (加權指數) 表現",
        "market_sentiment": "市場情緒指標",
        "overweight": "強於大盤",
        "underweight": "弱於大盤",
        "health_score": "財務健康評分 (Fundamental Health)",
        "profitability": "獲利能力",
        "solvency": "償債能力",
        "efficiency": "營運效率",
        "support_level": "支撐位",
        "resistance_level": "壓力位",
        "buy_suggestion": "交易策略建議",
        "buy_point": "建議買點",
        "neutral": "觀望",
        "strategy_desc": "基於技術指標與支撐壓力的綜合建議。",
        "tab_strategy": "⚖️ 策略回測",
        "win_rate": "勝率",
        "total_return": "總報酬率",
        "max_drawdown": "最大回撤 (MDD)",
        "sharpe_ratio": "夏普比率 (Sharpe)",
        "equity_curve": "資金曲線",
        "signal_buy": "買進訊號",
        "signal_sell": "賣出訊號",
        "entry_price": "進場價",
        "stop_loss": "止損價 (SL)",
        "take_profit": "止盈價 (TP)",
        "risk_management": "風險管理建議",
        "pos_size": "建議倉位",
        "ai_analysis": "AI 智能投資分析",
        "ai_score": "AI 信心評分",
        "ai_reasoning": "AI 推理路徑",
        "ai_summary": "綜合總結與行動",
        "signal_history": "訊號歷史記錄",
        "risk_calc_desc": "基於 100萬 資金，單筆風險 2% 計算。",
        "no_signals": "近期無交易訊號。",
        "bt_title": "7日預測準確度驗證",
        "rate_limit_msg": "請求過於頻繁。請稍候幾分鐘再試。"
    }
}

# Sidebar settings
st.sidebar.header(t["settings"])
lang = st.sidebar.selectbox("Language / 語言", options=["English", "繁體中文"], index=1)
t = translations[lang]

st.title(t["title"])
st.markdown(t["description"])

ticker_input = st.sidebar.text_input(t["ticker_label"], value="2330.TW")
period = st.sidebar.selectbox(t["period_label"], options=["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
interval = st.sidebar.selectbox(t["interval_label"], options=["1d", "1wk", "1mo"], index=0)

# Prediction settings
st.sidebar.markdown("---")
st.sidebar.subheader(t["prediction_header"])
predict_days = st.sidebar.slider(t["prediction_days"], 1, 30, 7)

# Display settings
st.sidebar.markdown("---")
show_raw = st.sidebar.checkbox(t["tab_raw"], value=False)

def format_large_number(num, lang):
    if not isinstance(num, (int, float)):
        return "N/A"
    
    if lang == "English":
        if num >= 1e12:
            return f"{num/1e12:.2f} T"
        elif num >= 1e9:
            return f"{num/1e9:.2f} B"
        elif num >= 1e6:
            return f"{num/1e6:.2f} M"
        else:
            return f"{num:,.0f}"
    else:  # 繁體中文
        if num >= 1e12:
            return f"{num/1e12:.2f} 兆"
        elif num >= 1e8:
            return f"{num/1e8:.2f} 億"
        elif num >= 1e4:
            return f"{num/1e4:.2f} 萬"
        else:
            return f"{num:,.0f}"

import time

@st.cache_data(ttl=3600)
def get_stock_info(ticker):
    try:
        s = yf.Ticker(ticker)
        return s.info
    except Exception:
        return {}

@st.cache_data(ttl=3600)
def get_stock_data(ticker, period, interval):
    for i in range(3):
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            if data is not None and not data.empty:
                # Handle cases where yfinance returns a MultiIndex for a single ticker
                if isinstance(data.columns, pd.MultiIndex):
                    if 'Close' in data.columns.get_level_values(0):
                        data.columns = data.columns.get_level_values(0)
                    else:
                        data.columns = data.columns.get_level_values(1)
                return data
        except Exception as e:
            if i < 2:
                time.sleep(2 * (i + 1))
                continue
    return None

@st.cache_data(ttl=3600)
def get_benchmark_data(period, interval):
    for i in range(3):
        try:
            data = yf.download("^TWII", period=period, interval=interval, progress=False)
            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                return data
        except Exception as e:
            if i < 2:
                time.sleep(2 * (i + 1))
                continue
    return None

@st.cache_data(ttl=600)
def run_monte_carlo(s_close, predict_days, iterations=500):
    # Using log returns for more robust statistical properties
    log_returns = np.log(s_close / s_close.shift(1)).dropna()
    
    # Parameters for simulation
    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5 * var)
    stdev = log_returns.std()
    
    daily_returns = np.exp(drift + stdev * np.random.standard_normal((predict_days, iterations)))
    
    last_price = float(s_close.iloc[-1])
    price_list = np.zeros_like(daily_returns)
    price_list[0] = last_price * daily_returns[0]
    
    for t_step in range(1, predict_days):
        price_list[t_step] = price_list[t_step - 1] * daily_returns[t_step]
    
    # Calculate percentiles
    expected_path = np.median(price_list, axis=1)
    optimistic_path = np.percentile(price_list, 95, axis=1)
    pessimistic_path = np.percentile(price_list, 5, axis=1)
    
    return expected_path, optimistic_path, pessimistic_path, last_price

if ticker_input:
    data = get_stock_data(ticker_input, period, interval)
    
    if data is not None and not data.empty:
        # Defensive: Ensure all OHLCV are Series
        s_open = data['Open'].iloc[:, 0] if isinstance(data['Open'], pd.DataFrame) else data['Open']
        s_high = data['High'].iloc[:, 0] if isinstance(data['High'], pd.DataFrame) else data['High']
        s_low = data['Low'].iloc[:, 0] if isinstance(data['Low'], pd.DataFrame) else data['Low']
        s_close = data['Close'].iloc[:, 0] if isinstance(data['Close'], pd.DataFrame) else data['Close']
        s_volume = data['Volume'].iloc[:, 0] if isinstance(data['Volume'], pd.DataFrame) else data['Volume']
        
        # --- Pre-calculate Global Variables ---
        current_p = float(s_close.iloc[-1])
        prev_p = float(s_close.iloc[-2])
        price_change = current_p - prev_p
        price_change_pct = (price_change / prev_p) * 100
        
        # Support/Resistance
        support = float(s_low.rolling(window=20).min().iloc[-1])
        resistance = float(s_high.rolling(window=20).max().iloc[-1])
        
        # Technicals
        # RSI
        delta_s = s_close.diff()
        gain_s = (delta_s.where(delta_s > 0, 0)).rolling(window=14).mean()
        loss_s = (-delta_s.where(delta_s < 0, 0)).rolling(window=14).mean()
        rs_s = gain_s / loss_s
        rsi_series = 100 - (100 / (1 + rs_s))
        rsi_val = float(rsi_series.iloc[-1])
        
        # MACD
        exp1 = s_close.ewm(span=12, adjust=False).mean()
        exp2 = s_close.ewm(span=26, adjust=False).mean()
        macd_series = exp1 - exp2
        signal_series = macd_series.ewm(span=9, adjust=False).mean()

        # ATR
        high_low = s_high - s_low
        high_cp = np.abs(s_high - s_close.shift())
        low_cp = np.abs(s_low - s_close.shift())
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        atr_series = tr.rolling(window=14).mean()
        current_atr = float(atr_series.iloc[-1])

        # Bollinger Bands
        ma20_s = s_close.rolling(window=20).mean()
        std20_s = s_close.rolling(window=20).std()
        upper_bb = ma20_s + (std20_s * 2)
        lower_bb = ma20_s - (std20_s * 2)

        # MA60 for cross
        ma60_s = s_close.rolling(window=60).mean()
        
        # Stock Info
        info = get_stock_info(ticker_input)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(t["current_price"], f"{current_p:.2f}")
        with col2:
            st.metric(t["change"], f"{price_change:.2f}", f"{price_change_pct:.2f}%")
        with col3:
            volume = info.get('volume') or s_volume.iloc[-1]
            st.metric(t["volume"], format_large_number(float(volume), lang))
        with col4:
            market_cap = info.get('marketCap', 'N/A')
            st.metric(t["market_cap"], format_large_number(market_cap, lang))

        # --- Main Section Header ---
        stock_name = info.get('shortName') or info.get('longName', '')
        
        # Mapping for common Taiwan stocks to Chinese
        tw_stock_map = {
            "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2308.TW": "台達電",
            "2303.TW": "聯電", "2881.TW": "富邦金", "2882.TW": "國泰金", "2412.TW": "中華電",
            "1301.TW": "台塑", "1303.TW": "南亞", "2603.TW": "長榮", "2002.TW": "中鋼",
            "2382.TW": "廣達", "2357.TW": "華碩", "3008.TW": "大立光", "2886.TW": "兆豐金",
            "2884.TW": "玉山金", "2891.TW": "中信金", "5880.TW": "合庫金", "2892.TW": "第一金",
            "2880.TW": "華南金", "2885.TW": "元大金", "2883.TW": "開發金", "2887.TW": "台新金",
            "2890.TW": "永豐金", "2379.TW": "瑞昱", "3034.TW": "聯詠", "3711.TW": "日月光投控",
            "2327.TW": "國巨", "2345.TW": "智邦", "3231.TW": "緯創", "6669.TW": "緯穎",
            "2301.TW": "光寶科", "2377.TW": "微星", "2376.TW": "技嘉", "1101.TW": "台泥",
            "1216.TW": "統一", "2105.TW": "正新", "2201.TW": "裕隆", "2609.TW": "陽明",
            "2615.TW": "萬海", "2610.TW": "華航", "2618.TW": "長榮航", "2912.TW": "統一超",
            "9904.TW": "寶成", "9910.TW": "豐泰", "9921.TW": "巨大", "9914.TW": "美利達",
            "5434.TW": "崇越", "6239.TW": "力成", "2337.TW": "旺宏", "2408.TW": "南亞科",
            "2409.TW": "友達", "3481.TW": "群創", "2344.TW": "華邦電", "2313.TW": "華通",
            "2368.TW": "金像電", "2383.TW": "台光電", "6213.TW": "聯茂", "3037.TW": "欣興",
            "8046.TW": "南電", "3189.TW": "景碩", "2474.TW": "可成", "2354.TW": "鴻準",
            "2353.TW": "宏碁", "2324.TW": "仁寶", "2356.TW": "英業達", "4938.TW": "和碩",
            "2395.TW": "研華", "6414.TW": "樺漢", "6415.TW": "矽力-KY", "3661.TW": "世芯-KY",
            "3443.TW": "創意", "3035.TW": "智原", "3532.TW": "台勝科", "6488.TW": "環球晶",
            "5483.TW": "中美晶", "8069.TW": "元太", "1605.TW": "華新", "2606.TW": "裕民",
            "2637.TW": "慧洋-KY", "2207.TW": "和泰車", "2204.TW": "中華", "1402.TW": "遠東新",
            "1476.TW": "儒星", "1477.TW": "聚陽", "9933.TW": "中鼎", "6505.TW": "台塑化",
            "1326.TW": "台化", "1102.TW": "亞泥", "1504.TW": "東元", "1513.TW": "中興電",
            "1519.TW": "華城", "1503.TW": "士電", "2371.TW": "大同", "2633.TW": "台灣高鐵",
            "2634.TW": "漢翔", "2727.TW": "王品", "2707.TW": "晶華", "1210.TW": "大成",
            "1722.TW": "台肥", "1717.TW": "長興", "1710.TW": "東聯", "1704.TW": "長榮鋼",
            "2501.TW": "國建", "2542.TW": "興富發", "2548.TW": "聖暉", "5534.TW": "聖暉*",
            "6205.TW": "詮欣", "6214.TW": "精誠", "6271.TW": "同欣電", "6285.TW": "啟碁",
            "8016.TW": "矽創", "8150.TW": "南茂", "8299.TW": "群聯", "8436.TW": "大江",
            "9939.TW": "宏全", "9941.TW": "裕融", "9945.TW": "潤泰新", "9958.TW": "世紀鋼",
            "0050.TW": "元大台灣50", "0056.TW": "元大高股息", "00878.TW": "國泰永續高股息",
            "00919.TW": "群益台灣精選高息", "00929.TW": "復華台灣科技優息", "2312.TW": "金寶",
            "2352.TW": "佳世達", "2323.TW": "中環", "6116.TW": "彩晶", "2401.TW": "凌陽",
            "3006.TW": "晶豪科", "3044.TW": "健鼎", "2449.TW": "京元電子", "2451.TW": "創見",
            "2338.TW": "光罩", "2367.TW": "燿華", "2498.TW": "宏達電", "2388.TW": "威盛"
        }
        
        if lang == "繁體中文":
            if ticker_input.upper() in tw_stock_map:
                stock_name = tw_stock_map[ticker_input.upper()]
            elif "Taiwan Semiconductor Manufacturing" in stock_name:
                stock_name = "台積電"
        
        st.markdown(f"<h2 class='centered-header'>📈 {ticker_input} - {stock_name}</h2>", unsafe_allow_html=True)

        # Prepare Main Chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=s_open,
            high=s_high,
            low=s_low,
            close=s_close,
            name=ticker_input
        ))
        
        fig.add_trace(go.Scatter(x=data.index, y=ma20_s, name="MA20", line=dict(color='#3498DB', width=1.5)))
        fig.add_trace(go.Scatter(x=data.index, y=ma60_s, name="MA60", line=dict(color='#E67E22', width=1.5)))
        
        fig.update_layout(
            height=500,
            margin=dict(l=0, r=0, t=30, b=0),
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )

        # Tabs for better organization
        tab_list = [t["tab_overview"], t["tab_tech"], t["tab_predict"], t["tab_strategy"]]
        if show_raw:
            tab_list.append(t["tab_raw"])
            
        tabs = st.tabs(tab_list)
        tab1, tab2, tab3, tab_strat = tabs[0], tabs[1], tabs[2], tabs[3]
        if show_raw:
            tab4 = tabs[4]

        with tab_strat:
            st.subheader(f"⚖️ {t['tab_strategy']}")
            
            # Define Backtest Strategy: MA Crossover + RSI
            bt_df = pd.DataFrame(index=data.index)
            bt_df['Close'] = s_close
            bt_df['MA20'] = ma20_s
            bt_df['MA60'] = s_close.rolling(window=60).mean()
            bt_df['RSI'] = rsi_series
            
            # Signals
            bt_df['Signal'] = 0
            # Buy: MA20 > MA60 AND RSI < 50 (Pullback in uptrend) OR RSI < 30
            buy_cond = ((bt_df['MA20'] > bt_df['MA60']) & (bt_df['RSI'] < 50)) | (bt_df['RSI'] < 30)
            # Sell: MA20 < MA60 OR RSI > 70
            sell_cond = (bt_df['MA20'] < bt_df['MA60']) | (bt_df['RSI'] > 70)
            
            bt_df.loc[buy_cond, 'Signal'] = 1
            bt_df.loc[sell_cond, 'Signal'] = -1
            
            # Backtest Simulation
            bt_df['Position'] = bt_df['Signal'].replace(0, method='ffill').shift(1).fillna(0)
            bt_df['Position'] = bt_df['Position'].apply(lambda x: x if x > 0 else 0) # Only Long for simplicity
            
            bt_df['Daily_Return'] = bt_df['Close'].pct_change()
            bt_df['Strategy_Return'] = bt_df['Position'] * bt_df['Daily_Return']
            bt_df['Equity_Curve'] = (1 + bt_df['Strategy_Return']).cumprod()
            bt_df['Drawdown'] = bt_df['Equity_Curve'] / bt_df['Equity_Curve'].cummax() - 1
            
            # Metrics Calculation
            total_ret = (bt_df['Equity_Curve'].iloc[-1] - 1) * 100
            mdd = bt_df['Drawdown'].min() * 100
            
            # Win Rate (based on closed trades)
            bt_df['Trade_Signal'] = bt_df['Position'].diff()
            trades = []
            entry_p = 0
            for i in range(len(bt_df)):
                if bt_df['Trade_Signal'].iloc[i] == 1: # Entry
                    entry_p = bt_df['Close'].iloc[i]
                elif bt_df['Trade_Signal'].iloc[i] == -1 and entry_p != 0: # Exit
                    exit_p = bt_df['Close'].iloc[i]
                    trades.append(exit_p > entry_p)
                    entry_p = 0
            
            win_rate = (sum(trades) / len(trades) * 100) if trades else 0
            sharpe = (bt_df['Strategy_Return'].mean() / bt_df['Strategy_Return'].std() * np.sqrt(252)) if bt_df['Strategy_Return'].std() != 0 else 0
            
            # Display Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(t["total_return"], f"{total_ret:.2f}%")
            m2.metric(t["win_rate"], f"{win_rate:.1f}%")
            m3.metric(t["max_drawdown"], f"{mdd:.2f}%")
            m4.metric(t["sharpe_ratio"], f"{sharpe:.2f}")
            
            # Visualization
            st.markdown(f"#### 📈 {t['equity_curve']}")
            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Equity_Curve'], name=t["equity_curve"], fill='tozeroy', line=dict(color='#1ABC9C')))
            fig_equity.update_layout(height=400, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_equity, use_container_width=True)
            
            # --- Risk Management Section ---
            st.markdown("---")
            st.markdown(f"#### 🛡️ {t['risk_management']}")
            r1, r2 = st.columns(2)
            
            # ATR-based Stop Loss
            sl_dist = current_atr * 2
            tp_dist = current_atr * 3
            
            with r1:
                st.write(f"**{t['stop_loss']}**: `{current_p - sl_dist:.2f}` (ATR x 2)")
                st.write(f"**{t['take_profit']}**: `{current_p + tp_dist:.2f}` (ATR x 3)")
            
            with r2:
                # Simple position sizing (e.g. risk 2% of capital)
                risk_per_share = sl_dist
                capital = 1000000 # Default 1M TWD
                suggested_shares = (capital * 0.02) / risk_per_share
                st.write(f"**{t['pos_size']}**: `{int(suggested_shares)}` 股")
                st.caption(t["risk_calc_desc"])
            
            # --- Recent Signals Table ---
            st.markdown("---")
            st.markdown(f"#### {t['signal_history']}")
            signals_df = bt_df[bt_df['Signal'] != 0].tail(10).copy()
            if not signals_df.empty:
                signals_df['Type'] = signals_df['Signal'].map({1: t["signal_buy"], -1: t["signal_sell"]})
                st.table(signals_df[['Close', 'Type']].sort_index(ascending=False))
            else:
                st.write(t["no_signals"])

        with tab1:
            # Main Price Chart with MA and BB
            st.plotly_chart(fig, use_container_width=True)

            # --- Dashboard Row 1: Key Stats & Performance ---
            st.markdown("---")
            d_col1, d_col2 = st.columns([1, 1.5])
            
            with d_col1:
                st.markdown(f"<h4 class='centered-header'>📊 {t['key_stats']}</h4>", unsafe_allow_html=True)
                stats_container = st.container()
                with stats_container:
                    s1, s2 = st.columns(2)
                    s1.write(f"**{t['trailing_pe']}**")
                    s1.write(f"{info.get('trailingPE', 'N/A')}")
                    s1.write(f"**{t['div_yield']}**")
                    s1.write(f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "N/A")
                    
                    s2.write(f"**{t['high_52w']}**")
                    s2.write(f"{info.get('fiftyTwoWeekHigh', 'N/A')}")
                    s2.write(f"**{t['beta']}**")
                    s2.write(f"{info.get('beta', 'N/A')}")

            with d_col2:
                benchmark_data = get_benchmark_data(period, interval)
                if benchmark_data is not None and not benchmark_data.empty:
                    b_close = benchmark_data['Close'].iloc[:, 0] if isinstance(benchmark_data['Close'], pd.DataFrame) else benchmark_data['Close']
                    stock_norm = (s_close / s_close.iloc[0]) * 100
                    bench_norm = (b_close / b_close.iloc[0]) * 100
                    
                    fig_rel = go.Figure()
                    fig_rel.add_trace(go.Scatter(x=data.index, y=stock_norm, name=ticker_input, line=dict(color='#FF4B4B', width=2)))
                    fig_rel.add_trace(go.Scatter(x=benchmark_data.index, y=bench_norm, name='TAIEX (^TWII)', line=dict(color='#7F8C8D', dash='dash')))
                    
                    current_alpha = float(stock_norm.iloc[-1] - bench_norm.iloc[-1])
                    alpha_status = t["overweight"] if current_alpha > 0 else t["underweight"]
                    
                    fig_rel.update_layout(
                        title=f"<b>{t['relative_performance']}</b> ({alpha_status}: {current_alpha:+.2f}%)",
                        height=350,
                        margin=dict(l=0, r=0, t=40, b=0),
                        template="plotly_dark",
                        hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_rel, use_container_width=True)

            # --- Dashboard Row 2: Analyst & Health ---
            st.markdown("---")
            d_col3, d_col4 = st.columns([1.5, 1])
            
            with d_col3:
                st.markdown(f"<h4 class='centered-header'>🎯 {t['analyst_rating']}</h4>", unsafe_allow_html=True)
                # Combine Rating info and Gauge
                a1, a2 = st.columns([1, 2])
                with a1:
                    rec = info.get('recommendationKey', 'N/A').replace('_', ' ').title()
                    st.write(f"**{t['recommendation']}**")
                    st.info(f"### {rec}")
                    st.write(f"**{t['target_mean']}**")
                    st.success(f"### {info.get('targetMeanPrice', 'N/A')} TWD")
                
                with a2:
                    current_p = float(s_close.iloc[-1])
                    target_p = info.get('targetMeanPrice')
                    if target_p and isinstance(target_p, (int, float)):
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = target_p,
                            delta = {'reference': current_p, 'relative': True, 'valueformat': ".2%"},
                            gauge = {
                                'axis': {'range': [None, max(target_p, current_p) * 1.2]},
                                'bar': {'color': "#1ABC9C"},
                                'steps' : [
                                    {'range': [0, current_p], 'color': "#34495E"},
                                ],
                                'threshold' : {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': target_p
                                }
                            }
                        ))
                        fig_gauge.update_layout(height=250, margin=dict(t=30, b=0, l=10, r=10), template="plotly_dark")
                        st.plotly_chart(fig_gauge, use_container_width=True)

            with d_col4:
                st.markdown(f"<h4 class='centered-header'>🏥 {t['health_score']}</h4>", unsafe_allow_html=True)
                score = 0
                metrics_list = []
                
                # Logic (Re-using existing logic)
                roe = info.get('returnOnEquity')
                if roe:
                    score += min(max(roe * 100, 0), 40)
                    metrics_list.append((t["profitability"], f"{roe*100:.1f}%"))
                
                d2e = info.get('debtToEquity')
                if d2e:
                    score += 30 if d2e < 100 else (15 if d2e < 200 else 0)
                    metrics_list.append((t["solvency"], f"{d2e:.1f}%"))
                
                margin = info.get('profitMargins')
                if margin:
                    score += min(max(margin * 100, 0), 30)
                    metrics_list.append((t["efficiency"], f"{margin*100:.1f}%"))
                
                st.metric(t["health_score"], f"{score:.2f}/100")
                st.progress(score / 100)
                for m_name, m_val in metrics_list:
                    st.write(f"**{m_name}:** {m_val}")
                
            # --- AI Intelligence Analysis Section (Moved out for Centering) ---
            st.markdown("---")
            
            # Use columns to center the AI content across the whole tab width
            ai_col_left, ai_col_mid, ai_col_right = st.columns([1, 2, 1])
            
            with ai_col_mid:
                st.markdown(f"<h4 class='centered-header'>🧠 {t['ai_analysis']}</h4>", unsafe_allow_html=True)
                
                # Logic for AI-like Score and Reasoning
                ai_score = 50 # Base score
                reasoning = []
                
                # 1. Trend Factor
                ma20_last = ma20_s.iloc[-1]
                ma60_last = ma60_s.iloc[-1]
                if ma20_last > ma60_last:
                    ai_score += 15
                    reasoning.append("📈 **趨勢看漲**: 短期均線高於長期均線 (Golden Cross)")
                else:
                    ai_score -= 15
                    reasoning.append("📉 **趨勢看跌**: 均線呈現空頭排列 (Death Cross)")
                
                # 2. Momentum Factor (RSI)
                if rsi_val < 30:
                    ai_score += 20
                    reasoning.append("🔥 **超賣反彈**: RSI 處於低檔區，具備強烈反彈潛力")
                elif rsi_val > 70:
                    ai_score -= 20
                    reasoning.append("🧊 **超買修正**: RSI 處於高檔，短期內面臨拉回壓力")
                else:
                    reasoning.append("⚖️ **動能中性**: RSI 處於平衡區，暫無極端訊號")

                # 3. Support/Resistance Factor
                if current_p <= support * 1.015:
                    ai_score += 10
                    reasoning.append(f"🛡️ **支撐力道**: 價格接近支撐位 ({support:.2f})")
                elif current_p >= resistance * 0.985:
                    ai_score -= 10
                    reasoning.append(f"🚧 **壓力挑戰**: 價格接近壓力位 ({resistance:.2f})")

                # 4. Fundamental Health
                if score > 70:
                    ai_score += 10
                    reasoning.append("💎 **體質優異**: 財務健康評分高於 70，適合中長線持有")
                
                # Clamp score
                ai_score = max(0, min(100, ai_score))
                
                # Display AI Card
                st.markdown(f"""
                <div class="ai-card">
                    <span class="ai-badge">Deep Analysis</span>
                    <h3 style="margin-top:10px; color:#3498db; text-align:center;">{t['ai_score']}: {ai_score}/100</h3>
                    <div style="margin-bottom:15px;">
                        <div style="background-color: #34495e; height: 8px; border-radius: 4px;">
                            <div style="background-color: #3498db; width: {ai_score}%; height: 8px; border-radius: 4px;"></div>
                        </div>
                    </div>
                    <p style="font-size: 0.9rem; color: #BDC3C7;">{t['ai_reasoning']}:</p>
                    <ul style="font-size: 0.85rem; color: #f0f2f6; padding-left: 20px;">
                        {"".join([f"<li>{r}</li>" for r in reasoning])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Action Suggestion
                st.markdown("<br>", unsafe_allow_html=True)
                if ai_score >= 70:
                    st.success(f"🚀 **{t['ai_summary']}**: 強力建議分批進場。目前價格具備技術面與基本面雙重支撐。")
                    st.write(f"建議進場位: `{max(support, current_p * 0.98):.2f}`")
                elif ai_score <= 30:
                    st.warning(f"⚠️ **{t['ai_summary']}**: 建議暫時避開或減碼。市場信號顯示當前下行風險較高。")
                    st.write(f"壓力參考位: `{resistance:.2f}`")
                else:
                    st.info(f"⚖️ **{t['ai_summary']}**: {t['neutral']}。建議耐心等待關鍵位置（支撐位或趨勢反轉信號）再行操作。")
                
                st.caption(t["strategy_desc"])

        with tab2:
            st.subheader(t["tech_analysis"])
            
            # RSI Chart
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=data.index, y=rsi_series, name='RSI', line=dict(color='purple')))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text=t["overbought"])
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text=t["oversold"])
            fig_rsi.update_layout(title=t["rsi_label"], yaxis_range=[0, 100], height=300, template="plotly_dark")
            st.plotly_chart(fig_rsi, use_container_width=True)
            
            # ATR Chart
            fig_atr = go.Figure()
            fig_atr.add_trace(go.Scatter(x=data.index, y=atr_series, name='ATR', line=dict(color='orange')))
            fig_atr.update_layout(title=t["atr_label"], height=300, template="plotly_dark")
            st.plotly_chart(fig_atr, use_container_width=True)
            
            # MACD Chart
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=data.index, y=macd_series, name='MACD', line=dict(color='cyan')))
            fig_macd.add_trace(go.Scatter(x=data.index, y=signal_series, name=t["signal_label"], line=dict(color='orange')))
            fig_macd.add_bar(x=data.index, y=macd_series - signal_series, name='Histogram')
            fig_macd.update_layout(title=t["macd_label"], height=300, template="plotly_dark")
            st.plotly_chart(fig_macd, use_container_width=True)

        with tab3:
            # Prediction Section with better visual hierarchy
            st.markdown(f"### {t['prediction_header']}")
            st.info(t["prediction_desc"])
            
            # --- Monte Carlo Simulation ---
            with st.spinner(t.get("backtest_wait", "Calculating...")):
                expected_path, optimistic_path, pessimistic_path, last_price_mc = run_monte_carlo(s_close, predict_days)
            
            prediction_dates = [data.index[-1] + timedelta(days=i) for i in range(1, predict_days + 1)]
            
            # Forecast Metric
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                st.metric(t["mc_expected"], f"{expected_path[-1]:.2f}", f"{(expected_path[-1]-last_price_mc):+.2f}")
            with f_col2:
                st.metric(t["mc_optimistic"], f"{optimistic_path[-1]:.2f}", f"{(optimistic_path[-1]-last_price_mc):+.2f}", delta_color="normal")
            with f_col3:
                st.metric(t["mc_pessimistic"], f"{pessimistic_path[-1]:.2f}", f"{(pessimistic_path[-1]-last_price_mc):+.2f}", delta_color="inverse")
            
            # Forecast Chart
            fig_pred = go.Figure()
            
            # Historical part
            fig_pred.add_trace(go.Scatter(
                x=data.index[-60:], 
                y=s_close[-60:].values.flatten(), 
                name="Historical", 
                line=dict(color='#BDC3C7', width=2)
            ))
            
            # Confidence Interval Shading
            fig_pred.add_trace(go.Scatter(
                x=prediction_dates + prediction_dates[::-1],
                y=list(optimistic_path) + list(pessimistic_path)[::-1],
                fill='toself',
                fillcolor='rgba(26, 188, 156, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name="90% Confidence Interval"
            ))
            
            # Prediction paths
            fig_pred.add_trace(go.Scatter(
                x=prediction_dates, 
                y=expected_path, 
                name=t["mc_expected"], 
                line=dict(color='#1ABC9C', width=3)
            ))
            
            fig_pred.add_trace(go.Scatter(
                x=prediction_dates, 
                y=optimistic_path, 
                name=t["mc_optimistic"], 
                line=dict(color='#2ECC71', width=1, dash='dot')
            ))
            
            fig_pred.add_trace(go.Scatter(
                x=prediction_dates, 
                y=pessimistic_path, 
                name=t["mc_pessimistic"], 
                line=dict(color='#E74C3C', width=1, dash='dot')
            ))
            
            fig_pred.update_layout(
                title=dict(text=f"<b>{ticker_input}</b> - Monte Carlo {predict_days}D Forecast", font=dict(size=20)),
                yaxis_title="Price (TWD)",
                template="plotly_dark",
                height=500,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            # Backtesting section
            st.markdown("---")
            st.markdown(f"### {t['backtest_header']}")
            st.caption(t["backtest_desc"])
            
            if len(s_close) > 20:
                train_s_close = s_close.iloc[:-7]
                actual_s_close = s_close.iloc[-7:]
                
                # Monte Carlo for Backtest
                bt_log_returns = np.log(train_s_close / train_s_close.shift(1)).dropna()
                bt_u = bt_log_returns.mean()
                bt_var = bt_log_returns.var()
                bt_drift = bt_u - (0.5 * bt_var)
                bt_stdev = bt_log_returns.std()
                
                bt_iterations = 500
                bt_daily_returns = np.exp(bt_drift + bt_stdev * np.random.standard_normal((7, bt_iterations)))
                
                bt_start_price = float(train_s_close.iloc[-1])
                bt_price_list = np.zeros_like(bt_daily_returns)
                bt_price_list[0] = bt_start_price * bt_daily_returns[0]
                
                for t_step in range(1, 7):
                    bt_price_list[t_step] = bt_price_list[t_step - 1] * bt_daily_returns[t_step]
                
                bt_expected_prices = np.median(bt_price_list, axis=1)
                
                actual_prices = actual_s_close.values.flatten()
                mae = float(np.mean(np.abs(actual_prices - bt_expected_prices)))
                actual_mean = float(np.mean(actual_prices))
                accuracy = float(100 - (mae / actual_mean * 100))
                
                # Visual Backtest metrics
                b_col1, b_col2, b_col3 = st.columns(3)
                with b_col1:
                    st.metric(t["accuracy_score"], f"{accuracy:.2f}%", delta=None)
                with b_col2:
                    st.metric(t["error_mae"], f"{mae:.2f}", delta="TWD", delta_color="off")
                with b_col3:
                    st.metric(t["prediction_days"], "7", delta="Days", delta_color="off")
                
                # Backtest visualization
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=actual_s_close.index, y=actual_prices, name=t["actual_price"], line=dict(color='#F1C40F', width=3)))
                fig_bt.add_trace(go.Scatter(x=actual_s_close.index, y=bt_expected_prices, name=t["predicted_price_bt"], line=dict(color='#3498DB', width=2, dash='dash')))
                
                fig_bt.update_layout(
                    title=t["bt_title"],
                    yaxis_title="Price (TWD)",
                    template="plotly_dark",
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20),
                    hovermode="x unified"
                )
                st.plotly_chart(fig_bt, use_container_width=True)

        if show_raw:
            with tab4:
                st.subheader(t["raw_data"])
                st.dataframe(data.tail(50), use_container_width=True)
    else:
        st.error(t["no_data"])
        st.warning(f"⚠️ {t['rate_limit_msg']}")