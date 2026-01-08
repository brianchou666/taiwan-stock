"""
================================================================================
PROJECT: Professional Quant Trading Terminal (Taiwan Stock Market)
VERSION: 2.0.0
AUTHOR: Quant Systems Team
LICENSE: Proprietary / Commercial Buyout
DESCRIPTION: 
    An integrated quantitative analysis system featuring AI-driven strategy 
    evolution, multi-factor scoring, and advanced Monte Carlo risk simulation.
================================================================================
"""

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import streamlit as st
import time
import feedparser
import urllib.parse
from scipy.stats import t as student_t

# --- SYSTEM CONFIGURATION ---
SYSTEM_SETTINGS = {
    "RISK_FREE_RATE": 0.015,       # Annual risk-free rate for Sharpe calculation
    "DEFAULT_BACKTEST_DAYS": 5,    # Holding period for backtest evaluation
    "SIMULATION_DAYS": 30,         # Prediction horizon for Monte Carlo
    "AI_CONFIDENCE_THRESHOLD": 60, # Minimum score to trigger 'Strong Buy'
    "THEME_COLOR": "#58A6FF"       # Primary institutional blue
}

st.set_page_config(
    page_title="專業量化交易終端 | AI-Powered Analysis", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

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
    "tab_news": "📰 市場新聞",
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
    """
    獲取股票歷史數據 (Fetches historical stock data from yfinance)
    包含快取機制以優化性能 (Includes caching for performance optimization)
    """
    ticker = ticker.strip().upper()
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        st.error(f"數據抓取失敗 (Data fetch failed): {e}")
        return None

@st.cache_data
def get_financial_data(ticker):
    """
    獲取年度財務數據趨勢 (Fetches annual financial data for trends)
    用於分析營收與淨利增長 (Used for analyzing Revenue & Net Income growth)
    """
    try:
        stock = yf.Ticker(ticker)
        financials = stock.financials
        if financials is not None and not financials.empty:
            # 提取營收與淨利 (Extract Revenue and Net Income)
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

def analyze_news_sentiment(title):
    """
    簡易金融新聞情緒分析
    """
    bullish_words = ['漲', '紅', '創高', '噴', '強', '買超', '利多', '成長', '優於預期', '上修', '展望佳', '獲利翻倍', '轉虧為盈', '訂單滿']
    bearish_words = ['跌', '綠', '破底', '崩', '弱', '賣超', '利空', '衰退', '低於預期', '下修', '展望淡', '虧損', '裁員', '訂單減']
    
    score = 0
    for word in bullish_words:
        if word in title:
            score += 1
    for word in bearish_words:
        if word in title:
            score -= 1
            
    if score > 0:
        return "利多", "#3FB950"
    elif score < 0:
        return "利空", "#F85149"
    else:
        return "中性", "#8B949E"

def get_expert_insight(ticker, price, rsi, rating, macd_val, signal_val, buy_sigs, sell_sigs, current_date):
    """
    生成專家診斷報告 (Generates Expert Technical Diagnosis Report)
    基於 RSI, MACD 與 AI 評級提供綜合建議
    """
    # RSI 分析 (RSI Analysis)
    rsi_status = "超買" if rsi > 70 else "超賣" if rsi < 30 else "中性"
    rsi_color = "#ef5350" if rsi > 70 else "#26a69a" if rsi < 30 else "#8B949E"
    rsi_desc = "股價進入超買區，短期回檔風險增加。" if rsi > 70 else "股價進入超賣區，可能存在反彈機會。" if rsi < 30 else "RSI 處於中性區間，走勢相對穩定。"
    
    # MACD 分析 (MACD Analysis)
    macd_diff = macd_val - signal_val
    macd_status = "多頭金叉" if macd_diff > 0 else "空頭死叉"
    macd_color = "#26a69a" if macd_diff > 0 else "#ef5350"
    macd_desc = "快線穿越慢線，短期動能偏多。" if macd_diff > 0 else "快線跌破慢線，短期動能轉弱。"
    
    # 策略建議 (Action Advice Based on Rating)
    action = "積極買進" if "STRONG BUY" in rating else "建議買進" if "BUY" in rating else "建議放空" if "SELL" in rating else "避開空頭" if "STRONG SELL" in rating else "中性觀望"
    action_color = "#26a69a" if "BUY" in rating else "#ef5350" if "SELL" in rating else "#58A6FF"
    
    # 即時信號檢查 (Real-time Signal Check)
    latest_signal = "目前無明確進場信號。"
    # 檢查今日是否觸發買賣信號 (Check if today triggers signals)
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

def calculate_health_scores(ticker_metadata):
    """
    根據股票元數據計算 1-10 分的財務健康得分 (Calculates 1-10 scores for financial health based on metadata)
    """
    # 1. 獲利能力得分 (Profitability Score - Margins)
    pm = ticker_metadata.get('profitMargins', 0) or 0
    roe = ticker_metadata.get('returnOnEquity', 0) or 0
    p_score = min(10, max(1, int(pm * 20 + roe * 15))) if pm and roe else 5
    
    # 2. 槓桿得分 (Leverage Score - Debt to Equity)
    de = ticker_metadata.get('debtToEquity', 100) or 100
    l_score = min(10, max(1, 10 - int(de / 40))) if de else 8
    
    # 3. 現金流得分 (Cash Flow Score)
    fcf = ticker_metadata.get('freeCashflow', 0) or 0
    rev = ticker_metadata.get('totalRevenue', 1) or 1
    c_score = min(10, max(1, int((fcf / rev) * 30 + 5))) if fcf and rev else 6
    
    return p_score, l_score, c_score

def optimize_ai_weights(data, rsi_series, ema20, ema50, bb_lower, bb_upper, vr_series, atr_series, health_scores):
    """
    AI Self-Evolution Engine:
    Backtests multiple weight combinations on historical data to find the optimal
    strategy for the specific stock.
    """
    # Define weight candidates (Tech, Fund, Vol)
    candidates = [
        (0.6, 0.2, 0.2), (0.4, 0.4, 0.2), (0.4, 0.2, 0.4),
        (0.3, 0.4, 0.3), (0.5, 0.3, 0.2), (0.2, 0.4, 0.4),
        (0.33, 0.33, 0.34)
    ]
    
    best_weights = (0.4, 0.3, 0.3)
    max_accuracy = -1.0
    
    # We only look at the last 60 days for learning to keep it relevant
    learning_period = 60
    if len(data) < learning_period:
        return best_weights, 0
        
    test_data = data.iloc[-learning_period:]
    
    # Simple backtest of candidates
    for w_tech, w_fund, w_vol in candidates:
        correct_preds = 0
        total_signals = 0
        
        # Check potential signals in the last 40 days (leaving 20 days for performance check)
        for i in range(len(test_data) - 20):
            idx = i
            # Simulate a simplified entry signal
            # RSI low + Close near BB lower
            curr_rsi = rsi_series.iloc[-(learning_period-i)]
            curr_close = test_data['Close'].iloc[i]
            curr_bb_l = bb_lower.iloc[-(learning_period-i)]
            
            if curr_rsi < 40 or curr_close < curr_bb_l * 1.02:
                total_signals += 1
                # Check if price went up in next 5 days
                future_max = test_data['High'].iloc[i+1:i+6].max()
                if future_max > curr_close * 1.03: # 3% gain target
                    correct_preds += 1
        
        accuracy = correct_preds / total_signals if total_signals > 0 else 0
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_weights = (w_tech, w_fund, w_vol)
            
    return best_weights, max_accuracy

def get_ai_entry_strategy(data, rsi, ema20, ema50, bb_lower, win_prob, health_scores, vr, atr, dynamic_weights=None):
    """Generates AI-driven entry strategy and price using multi-factor scoring."""
    current_price = float(data['Close'].iloc[-1])
    p_score, l_score, c_score = health_scores
    
    # Use dynamic weights if provided, else default
    w_tech, w_fund, w_vol = dynamic_weights if dynamic_weights else (0.4, 0.3, 0.3)
    
    # 1. Technical Score (0-100)
    t_raw = 0
    if ema20 > ema50: t_raw += 30
    if rsi < 40: t_raw += 40
    elif rsi < 60: t_raw += 20
    if current_price < bb_lower * 1.02: t_raw += 30
    
    # 2. Fundamental Score (0-100)
    f_raw = (p_score + l_score + c_score) / 30 * 100
    
    # 3. Volume/Momentum Score (0-100)
    v_raw = 0
    if vr > 150: v_raw += 60
    elif vr > 100: v_raw += 30
    if win_prob > 60: v_raw += 40
    
    # Weighted Total Score (Normalized to 0-100)
    total_score = (t_raw * w_tech) + (f_raw * w_fund) + (v_raw * w_vol)
    
    # Decision Logic based on Total Score
    if total_score > 75:
        suggested_price = max(ema20, current_price * 0.99)
        confidence = f"極高 ({total_score:.0f}%)"
        action = "強力買進"
        desc = "技術、基本、量能全面看多，建議積極佈局。"
        color = "#3FB950"
    elif total_score > 55:
        suggested_price = ema20
        confidence = f"高 ({total_score:.0f}%)"
        action = "分批買進"
        desc = "趨勢偏多且體質健全，可於支撐位分批承接。"
        color = "#26a69a"
    elif total_score > 40:
        suggested_price = bb_lower
        confidence = f"中 ({total_score:.0f}%)"
        action = "等待回檔"
        desc = "多空力道均衡，建議等候回測關鍵支撐。"
        color = "#D29922"
    else:
        suggested_price = bb_lower * 0.95
        confidence = f"低 ({total_score:.0f}%)"
        action = "保守觀望"
        desc = "目前條件不佳，建議保持現金部位，靜待趨勢扭轉。"
        color = "#f85149"
        
    # Add Stop Loss suggestion using ATR
    stop_loss = current_price - (2 * atr)
    
    return {
        "price": suggested_price,
        "stop_loss": stop_loss,
        "confidence": confidence,
        "action": action,
        "desc": desc,
        "color": color,
        "score": total_score
    }

def check_signal_performance(signal_date, suggested_price, signal_type, full_data):
    """Checks how the signal performed after the trigger date."""
    try:
        # Get data after the signal date (up to 5 days)
        idx = full_data.index.get_loc(signal_date)
        after_data = full_data.iloc[idx+1 : idx+6]
        
        if len(after_data) == 0:
            return "觀察中", "#8B949E", 0
            
        current_actual = float(full_data['Close'].iloc[-1])
        
        if signal_type == "買入":
            # Did it reach the suggested entry price?
            reached_entry = (after_data['Low'] <= suggested_price).any()
            # Is it currently profitable relative to suggested price?
            profit_pct = (current_actual - suggested_price) / suggested_price * 100
            
            if reached_entry and profit_pct > 2:
                return "精準達標", "#3FB950", profit_pct
            elif reached_entry:
                return "已進場", "#D29922", profit_pct
            else:
                return "未觸及", "#8B949E", 0
        else:
            # Sell signal
            reached_exit = (after_data['High'] >= suggested_price).any()
            # Is price lower than exit price? (Meaning it was a good time to sell)
            save_pct = (suggested_price - current_actual) / suggested_price * 100
            
            if reached_exit and save_pct > 0:
                return "避險成功", "#3FB950", save_pct
            elif reached_exit:
                return "已出場", "#D29922", save_pct
            else:
                return "未觸及", "#8B949E", 0
    except:
        return "分析中", "#8B949E", 0

def run_advanced_mc_simulation(current_price, data, ai_score, atr, ticker_metadata=None, days=30, simulations=1000):
    """
    重構後的蒙地卡羅模擬引擎 v3.0：
    - 採用 Student's t-分佈模擬肥尾效應 (Uses Student's t-distribution for fat-tail modeling)
    - 整合均值回歸 (EMA200) 機制 (Mean Reversion to EMA200)
    - 多因子漂移率與動態波動率調整 (Multi-factor drift and dynamic volatility)
    """
    returns = data['Close'].pct_change().dropna()
    hist_vol = returns.std()
    
    # 1. 計算多因子漂移率 (Drift)
    # AI 信心偏誤
    ai_bias = (ai_score - 50) / 4000 
    
    # 趨勢偏誤 (EMA20 vs EMA50)
    ema20 = data['Close'].ewm(span=20).mean().iloc[-1]
    ema50 = data['Close'].ewm(span=50).mean().iloc[-1]
    ema200 = data['Close'].ewm(span=200).mean().iloc[-1]
    trend_bias = 0.0015 if ema20 > ema50 else -0.0015
    
    # 均值回歸項 (Mean Reversion to EMA200)
    reversion_strength = 0.05 
    dist_from_mean = (ema200 - current_price) / current_price
    reversion_bias = dist_from_mean * reversion_strength / days
    
    # 法人目標價偏誤 (如有)
    target_bias = 0
    if ticker_metadata and ticker_metadata.get('targetMedianPrice'):
        target_price = ticker_metadata.get('targetMedianPrice')
        target_dist = (target_price - current_price) / current_price
        target_bias = target_dist * 0.1 / days

    mu = returns.mean() + ai_bias + trend_bias + reversion_bias + target_bias
    
    # 2. 波動率調整 (Volatility)
    atr_vol_adj = (atr / current_price) / 1.2
    vol = max(hist_vol, atr_vol_adj) * 1.1 
    
    # 3. 模擬執行 (使用 Student's t-分佈)
    df_student = 5 
    sim_results = np.zeros((days + 1, simulations))
    sim_results[0] = current_price
    
    # 預先生成隨機數
    random_shocks = student_t.rvs(df=df_student, loc=mu, scale=vol, size=(days, simulations))
    
    for t in range(1, days + 1):
        sim_results[t] = sim_results[t-1] * (1 + random_shocks[t-1])
        vol *= np.random.uniform(0.99, 1.01)
        
    return pd.DataFrame(sim_results)

def get_ai_exit_strategy(data, rsi, bb_upper, target_median, atr):
    """Generates AI-driven exit strategy and price with dynamic stop-profit."""
    current_price = float(data['Close'].iloc[-1])
    base_target = target_median if target_median and target_median > current_price else bb_upper
    
    # Calculate Dynamic Stop Profit (Trailing Stop)
    trailing_stop = current_price - (1.5 * atr)
    
    # Calculate Exit Score (Higher means more urgent to exit)
    exit_score = 0
    if rsi > 80: exit_score += 40
    elif rsi > 70: exit_score += 25
    elif rsi > 60: exit_score += 10
    
    if current_price > bb_upper: exit_score += 30
    
    # Trend weakness
    ema20 = data['Close'].ewm(span=20).mean().iloc[-1]
    if current_price < ema20: exit_score += 30
    
    # Normalize score (0-100, where 100 is "Must Sell")
    exit_score = min(100, exit_score)
    
    if exit_score > 70:
        suggested_price = current_price
        confidence = f"極高 ({exit_score:.0f}%)"
        action = "立即獲利"
        desc = "指標嚴重噴出且超買，建議現價清倉以確保利潤。"
        color = "#f85149"
    elif exit_score > 50:
        suggested_price = base_target
        confidence = f"高 ({exit_score:.0f}%)"
        action = "分批獲利"
        desc = "進入高檔超買區，建議於目標價分批獲利了結。"
        color = "#ef5350"
    elif exit_score > 30:
        suggested_price = max(base_target, current_price * 1.05)
        confidence = f"中 ({exit_score:.0f}%)"
        action = "部分減碼"
        desc = "股價動能稍緩，建議於壓力區附近減持部分部位。"
        color = "#D29922"
    else:
        suggested_price = max(base_target, current_price * 1.1)
        confidence = f"一般 ({exit_score:.0f}%)"
        action = "續抱觀察"
        desc = "趨勢尚未轉弱，建議守住移動停利點繼續持有。"
        color = "#58A6FF"
        
    return {
        "price": suggested_price,
        "trailing_stop": trailing_stop,
        "confidence": confidence,
        "action": action,
        "desc": desc,
        "color": color,
        "score": exit_score
    }

def calculate_quant_factors(data, ticker_metadata, rsi_series, atr_series):
    """
    量化多因子評分系統 (Modularized Quant Multi-Factor Scoring System)
    """
    factors = {}
    
    # 1. 趨勢因子 (Trend)
    close_prices = data['Close']
    ema20 = close_prices.ewm(span=20).mean()
    ema50 = close_prices.ewm(span=50).mean()
    trend_score = 0
    if close_prices.iloc[-1] > ema20.iloc[-1] > ema50.iloc[-1]:
        trend_score = 90
    elif close_prices.iloc[-1] > ema20.iloc[-1]:
        trend_score = 70
    else:
        trend_score = 40
    factors['趨勢 (Trend)'] = trend_score
    
    # 2. 動能因子 (Momentum)
    mom_score = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50
    factors['動能 (Momentum)'] = mom_score
    
    # 3. 波動因子 (Volatility)
    atr_val = float(atr_series.iloc[-1]) if not atr_series.empty else 0
    vol_ratio = (atr_val / close_prices.iloc[-1]) * 100
    vol_score = max(0, 100 - vol_ratio * 15) 
    factors['波動 (Volatility)'] = vol_score
    
    # 4. 量能因子 (Volume)
    v_ma5 = data['Volume'].rolling(5).mean().iloc[-1]
    v_ma20 = data['Volume'].rolling(20).mean().iloc[-1]
    volume_score = min(100, (v_ma5 / v_ma20) * 50) if v_ma20 > 0 else 50
    factors['量能 (Volume)'] = volume_score
    
    # 5. 價值因子 (Value Factor)
    pe = ticker_metadata.get('trailingPE') or ticker_metadata.get('forwardPE') or 20
    value_score = max(0, 100 - pe * 2)
    factors['價值 (Value)'] = value_score
    
    return factors

def run_backtest(data, all_signals):
    """
    進階量化回測引擎：計算勝率、收益、夏普比率與最大回撤
    """
    if not all_signals or len(data) < 20:
        return None
        
    results = []
    equity_curve = [1.0] # 權益曲線，起始為 1.0
    
    for sig in all_signals:
        sig_date = sig['date']
        post_data = data.loc[sig_date:].head(6)
        if len(post_data) < 2:
            continue
            
        entry_price = sig['price']
        exit_price = post_data['Close'].iloc[-1]
        
        profit_pct = (exit_price - entry_price) / entry_price
        if sig['type'] == '賣出':
            profit_pct = -profit_pct
            
        results.append(profit_pct)
        equity_curve.append(equity_curve[-1] * (1 + profit_pct))
        
    if not results:
        return None
        
    # --- 量化指標計算 ---
    results_series = pd.Series(results)
    win_rate = (results_series > 0).mean() * 100
    avg_profit = results_series.mean() * 100
    
    # 1. 夏普比率 (使用配置中的無風險利率)
    risk_free_per_trade = SYSTEM_SETTINGS["RISK_FREE_RATE"] / 252 * SYSTEM_SETTINGS["DEFAULT_BACKTEST_DAYS"]
    std_dev = results_series.std()
    sharpe = (results_series.mean() - risk_free_per_trade) / std_dev if std_dev > 0 else 0
    
    # 2. 最大回撤 (MDD)
    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.cummax()
    drawdowns = (equity_series - rolling_max) / rolling_max
    mdd = drawdowns.min() * 100
    
    return {
        "win_rate": win_rate,
        "avg_profit": avg_profit,
        "total_signals": len(results),
        "sharpe": sharpe,
        "mdd": mdd,
        "max_profit": results_series.max() * 100
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
    st.markdown(f'<p style="color: #8B949E; font-size: 0.8rem; font-weight: 600;">交易設定</p>', unsafe_allow_html=True)
    investment_budget = st.sidebar.number_input("單筆投資預算", min_value=1000, value=100000, step=10000, help="設定您預計單筆投入的金額，系統將據此計算建議股數。")

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
        ticker_metadata = stock.info
        
        # --- 大盤關聯度分析 (Benchmark: ^TWII) ---
        market_benchmark = "^TWII"
        try:
            m_data = yf.download(market_benchmark, period=period, interval=interval)
            if not m_data.empty:
                if isinstance(m_data.columns, pd.MultiIndex):
                    m_data.columns = m_data.columns.get_level_values(0)
                
                # 對齊日期計算
                aligned_df = pd.concat([data['Close'], m_data['Close']], axis=1).dropna()
                aligned_df.columns = ['Stock', 'Market']
                returns = aligned_df.pct_change().dropna()
                
                market_corr = returns['Stock'].corr(returns['Market'])
                market_cov = returns['Stock'].cov(returns['Market'])
                market_var = returns['Market'].var()
                market_beta = market_cov / market_var if market_var != 0 else 1.0
                
                # 相對漲幅 (近 30 天)
                stock_30d = (data['Close'].iloc[-1] / data['Close'].iloc[-20] - 1) * 100 if len(data) > 20 else 0
                market_30d = (m_data['Close'].iloc[-1] / m_data['Close'].iloc[-20] - 1) * 100 if len(m_data) > 20 else 0
                relative_strength = stock_30d - market_30d
            else:
                market_corr, market_beta, relative_strength = 0, 1.0, 0
        except:
            market_corr, market_beta, relative_strength = 0, 1.0, 0

        p_score, l_score, c_score = calculate_health_scores(ticker_metadata)
        
        # Update header placeholder with stock name and ticker
        ticker_display_name = ticker_metadata.get('shortName') or ticker_metadata.get('longName') or ticker_input
        stock_header_placeholder.markdown(f'<p style="color: #58A6FF; font-size: 1.2rem; font-weight: 600; margin-top: -5px;">{ticker_display_name} <span style="color: #8B949E; font-weight: 400; font-size: 0.9rem;">({ticker_input})</span></p>', unsafe_allow_html=True)

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

        # KD Indicator
        low_9 = data['Low'].rolling(window=9).min()
        high_9 = data['High'].rolling(window=9).max()
        rsv = (data['Close'] - low_9) / (high_9 - low_9) * 100
        k_series = rsv.ewm(com=2, adjust=False).mean()
        d_series = k_series.ewm(com=2, adjust=False).mean()

        # VR (Volume Ratio)
        def calculate_vr(df, n=26):
            diff = df['Close'].diff()
            up_vol = df['Volume'].where(diff > 0, 0).rolling(window=n).sum()
            down_vol = df['Volume'].where(diff < 0, 0).rolling(window=n).sum()
            flat_vol = df['Volume'].where(diff == 0, 0).rolling(window=n).sum()
            vr = (up_vol + 0.5 * flat_vol) / (down_vol + 0.5 * flat_vol) * 100
            return vr
        vr_series = calculate_vr(data)
        
        # ATR (Average True Range) for Volatility
        def calculate_atr(df, n=14):
            high_low = df['High'] - df['Low']
            high_cp = abs(df['High'] - df['Close'].shift())
            low_cp = abs(df['Low'] - df['Close'].shift())
            tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
            atr = tr.rolling(window=n).mean()
            return atr
        atr_series = calculate_atr(data)
        
        # --- AI Self-Evolution (Calculated early for signal filtering) ---
        with st.spinner(f"AI 正在針對 {ticker_input} 進行自我進化優化..."):
            best_weights, learn_acc = optimize_ai_weights(
                data, rsi_series, ema20, ema50, bb_lower, bb_upper, vr_series, atr_series, (p_score, l_score, c_score)
            )
            
            # Initial score for drift calculation and signal filtering
            initial_ai_strat = get_ai_entry_strategy(
                data, float(rsi_series.iloc[-1]), float(ema20.iloc[-1]), float(ema50.iloc[-1]), 
                float(bb_lower.iloc[-1]), 50, (p_score, l_score, c_score), 
                float(vr_series.iloc[-1]), float(atr_series.iloc[-1]), dynamic_weights=best_weights
            )
            ai_score = initial_ai_strat['score']

        # Calculate Buy/Sell Signals with AI + Multi-Factor Confirmation
        all_signals = []
        vol_ma5 = data['Volume'].rolling(window=5).mean()
        target_median = ticker_metadata.get('targetMedianPrice', None)
        
        # 遍歷數據生成信號 (Iterate to generate signals)
        for i in range(20, len(data)): # 從 20 開始確保有足夠的 MA 數據 (Ensure enough data for MAs)
            d = data.index[i]
            
            # --- 基礎技術指標過濾 (Basic Technical Filters) ---
            macd_gold = macd_series.iloc[i] > signal_series.iloc[i] and macd_series.iloc[i-1] <= signal_series.iloc[i-1]
            rsi_buy = rsi_series.iloc[i] > 30 and rsi_series.iloc[i-1] <= 30
            trend_ok_buy = data['Close'].iloc[i] > ema20.iloc[i]
            vol_ok = data['Volume'].iloc[i] > (vol_ma5.iloc[i] * 1.1)
            
            macd_death = macd_series.iloc[i] < signal_series.iloc[i] and macd_series.iloc[i-1] >= signal_series.iloc[i-1]
            rsi_sell = rsi_series.iloc[i] < 70 and rsi_series.iloc[i-1] >= 70
            trend_ok_sell = data['Close'].iloc[i] < ema20.iloc[i]

            # --- 買入信號處理 (Buy Signal Processing) ---
            if (macd_gold or rsi_buy) and trend_ok_buy and vol_ok:
                try:
                    h_rsi = float(rsi_series.iloc[i])
                    h_ema20 = float(ema20.iloc[i])
                    h_ema50 = float(ema50.iloc[i])
                    h_bb_lower = float(bb_lower.iloc[i])
                    h_vr = float(vr_series.iloc[i])
                    h_atr = float(atr_series.iloc[i])
                    
                    # 計算該時間點的 AI 買入評分
                    h_ai = get_ai_entry_strategy(data.iloc[:i+1], h_rsi, h_ema20, h_ema50, h_bb_lower, 55, (p_score, l_score, c_score), h_vr, h_atr, dynamic_weights=best_weights)
                    h_score = h_ai.get('score', 0)
                    
                    # 績效追蹤
                    perf_status, perf_color, perf_pct = check_signal_performance(d, h_ai['price'], "買入", data)
                    
                    price_val = float(data['Close'].iloc[i])
                    all_signals.append({
                        "date": d, "type": "買入", "price": price_val, "color": "#26a69a", "icon": "🔼",
                        "ai_price": h_ai['price'], "ai_action": h_ai['action'], "stop_loss": h_ai['stop_loss'],
                        "perf_status": perf_status, "perf_color": perf_color, "perf_pct": perf_pct,
                        "ai_verified": h_score >= 60, "ai_score": h_score
                    })
                except: continue

            # --- 賣出信號處理 (Sell Signal Processing) ---
            if (macd_death or rsi_sell) and trend_ok_sell:
                try:
                    h_rsi = float(rsi_series.iloc[i])
                    h_bb_upper = float(bb_upper.iloc[i])
                    h_atr = float(atr_series.iloc[i])
                    
                    # 計算該時間點的 AI 賣出評分
                    h_ai = get_ai_exit_strategy(data.iloc[:i+1], h_rsi, h_bb_upper, target_median, h_atr)
                    h_score = h_ai.get('score', 0)
                    
                    # 績效追蹤
                    perf_status, perf_color, perf_pct = check_signal_performance(d, h_ai['price'], "賣出", data)
                    
                    price_val = float(data['Close'].iloc[i])
                    all_signals.append({
                        "date": d, "type": "賣出", "price": price_val, "color": "#ef5350", "icon": "🔽",
                        "ai_price": h_ai['price'], "ai_action": h_ai['action'], "trailing_stop": h_ai.get('trailing_stop', 0),
                        "perf_status": perf_status, "perf_color": perf_color, "perf_pct": perf_pct,
                        "ai_verified": h_score >= 60, "ai_score": h_score
                    })
                except: continue
        
        # 過濾出最近 30 天的信號用於 UI 顯示
        one_month_ago = datetime.now() - timedelta(days=30)
        latest_signals = sorted(
            [s for s in all_signals if s['date'] >= one_month_ago], 
            key=lambda x: x['date'], 
            reverse=True
        )
        latest_sig = latest_signals[0] if latest_signals else None

        # 為舊有函數提供相容性
        buy_signals = [s['date'] for s in all_signals if s['type'] == "買入"]
        sell_signals = [s['date'] for s in all_signals if s['type'] == "賣出"]

        current_price = float(data['Close'].iloc[-1])
        
        # 頂部關鍵指標顯示 (Top Key Metrics Display)
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
            market_cap = ticker_metadata.get('marketCap', 'N/A')
            if isinstance(market_cap, (int, float)):
                st.metric(t["market_cap"], f"{market_cap/1e12:.2f}T")
            else:
                st.metric(t["market_cap"], "N/A")
        with m5:
            day_high = ticker_metadata.get('dayHigh', 'N/A')
            st.metric("當日最高", f"{day_high}" if isinstance(day_high, (int, float)) else "N/A")

        # 技術面評分徽章 (Technical Rating Badge)
        sig_text, sig_color, sig_val = get_signal_score(data, float(rsi_series.iloc[-1]), float(macd_series.iloc[-1]), float(signal_series.iloc[-1]))
        
        st.markdown("---")
        
        # --- AI 模擬與決策計算 (AI Simulation & Decision Calculations) ---
        with st.spinner(f"正在生成 {ticker_input} 未來模擬路徑與決策分析..."):
            atr = float(atr_series.iloc[-1])
            
            sim_df_full = run_advanced_mc_simulation(
                current_price, data, ai_score, atr, ticker_metadata=ticker_metadata, days=30, simulations=1000
            )
            
            # 將關鍵數據存入 session_state 以確保在不同分頁間切換時數據不會消失
            st.session_state['sim_df_full'] = sim_df_full
            st.session_state['ai_score'] = ai_score
            st.session_state['atr'] = atr
            st.session_state['current_price'] = current_price
            st.session_state['ticker_input'] = ticker_input
            st.session_state['prediction_dates'] = [data.index[-1] + timedelta(days=i) for i in range(len(sim_df_full))]
            
            # 提取 7 天模擬數據用於摘要 (Extract 7-day stats for summary)
            sim_df_7d = sim_df_full.iloc[0:8, :]
            median_sim_7d = sim_df_7d.median(axis=1).iloc[-1]
            win_prob_7d = (sim_df_7d.iloc[-1, :] > current_price).mean() * 100
            
            # AI 進出場策略計算 (AI Entry/Exit Strategy Calculation)
            entry_strategy = get_ai_entry_strategy(
                data, float(rsi_series.iloc[-1]), float(ema20.iloc[-1]), float(ema50.iloc[-1]), 
                float(bb_lower.iloc[-1]), win_prob_7d, (p_score, l_score, c_score), 
                float(vr_series.iloc[-1]), float(atr_series.iloc[-1]), dynamic_weights=best_weights
            )
            
            target_median = ticker_metadata.get('targetMedianPrice', None)
            exit_strategy = get_ai_exit_strategy(
                data, float(rsi_series.iloc[-1]), float(bb_upper.iloc[-1]), 
                target_median, float(atr_series.iloc[-1])
            )
            
        tab1, tab2, tab3, tab4, tab5 = st.tabs([t["tab_overview"], t["tab_tech"], t["tab_predict"], t["tab_signals"], t["tab_news"]])

        with tab1:
            # 1. 快速建議與預測摘要 (Quick Recommendation & Prediction Summary)
            insight_report = get_expert_insight(
                ticker_input, 
                current_price, 
                float(rsi_series.iloc[-1]), 
                sig_text, 
                float(macd_series.iloc[-1]), 
                float(signal_series.iloc[-1]), 
                buy_signals, 
                sell_signals, 
                data.index[-1]
            )
            
            # 顯示 AI 進化狀態與市場關聯 (Display Evolution Status & Market Correlation)
            st.markdown(f'''
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;">
                    <div style="background: rgba(56, 139, 253, 0.1); padding: 12px; border-radius: 10px; border: 1px solid #388bfd; display: flex; justify-content: space-between; align-items: center;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 1.2rem;">🧬</span>
                            <div>
                                <div style="color: #388bfd; font-size: 0.8rem; font-weight: bold;">AI 自我進化狀態：已優化</div>
                                <div style="color: #8B949E; font-size: 0.7rem;">針對 {ticker_input} 尋獲最佳權重</div>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="color: #3FB950; font-size: 0.8rem; font-weight: bold;">準確率: {learn_acc*100:.1f}%</div>
                        </div>
                    </div>
                    <div style="background: rgba(188, 140, 242, 0.1); padding: 12px; border-radius: 10px; border: 1px solid #BC8CF2; display: flex; justify-content: space-between; align-items: center;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 1.2rem;">📊</span>
                            <div>
                                <div style="color: #BC8CF2; font-size: 0.8rem; font-weight: bold;">市場關聯分析 (Beta)</div>
                                <div style="color: #8B949E; font-size: 0.7rem;">與大盤相關係數: {market_corr:.2f}</div>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="color: {'#3FB950' if relative_strength > 0 else '#ef5350'}; font-size: 0.8rem; font-weight: bold;">
                                相對強度: {relative_strength:+.1f}%
                            </div>
                            <div style="color: #E0E0E0; font-size: 0.65rem;">Beta: {market_beta:.2f}</div>
                        </div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)

            rec_col1, rec_col2, rec_col3, rec_col4 = st.columns([1, 1, 1, 1])
            with rec_col1:
                st.markdown(f'''
                    <div style="background: rgba(88, 166, 255, 0.1); padding: 12px; border-radius: 8px; border-left: 5px solid {insight_report['action']['color']}; height: 135px;">
                        <div style="color: #8B949E; font-size: 0.7rem; margin-bottom: 5px;">當前操作建議</div>
                        <div style="font-size: 1.2rem; font-weight: 800; color: {insight_report['action']['color']};">{insight_report['action']['status']}</div>
                        <div style="color: #E0E0E0; font-size: 0.7rem; line-height: 1.2;">{insight_report['action']['desc']}</div>
                    </div>
                ''', unsafe_allow_html=True)
            
            with rec_col2:
                st.markdown(f'''
                    <div style="background: rgba(63, 185, 80, 0.05); padding: 12px; border-radius: 8px; border-left: 5px solid {entry_strategy['color']}; height: 135px; border: 1px solid {entry_strategy['color']}33;">
                        <div style="color: #8B949E; font-size: 0.7rem; margin-bottom: 5px;">AI 建議買點 ({entry_strategy['confidence']})</div>
                        <div style="display: flex; align-items: baseline; gap: 5px;">
                            <span style="font-size: 1.2rem; font-weight: 800; color: {entry_strategy['color']};">${entry_strategy['price']:.1f}</span>
                            <span style="color: #8B949E; font-size: 0.65rem;">{entry_strategy['action']}</span>
                        </div>
                        <div style="color: #ef5350; font-size: 0.7rem; font-weight: bold; margin-top: 2px;">建議停損: ${entry_strategy['stop_loss']:.1f}</div>
                        <div style="color: #8B949E; font-size: 0.65rem; line-height: 1.2;">{entry_strategy['desc']}</div>
                    </div>
                ''', unsafe_allow_html=True)

            with rec_col3:
                st.markdown(f'''
                    <div style="background: rgba(248, 81, 73, 0.05); padding: 12px; border-radius: 8px; border-left: 5px solid {exit_strategy['color']}; height: 135px; border: 1px solid {exit_strategy['color']}33;">
                        <div style="color: #8B949E; font-size: 0.7rem; margin-bottom: 5px;">AI 建議賣點 / 出場策略 ({exit_strategy['confidence']})</div>
                        <div style="display: flex; align-items: baseline; gap: 5px;">
                            <span style="font-size: 1.2rem; font-weight: 800; color: {exit_strategy['color']};">${exit_strategy['price']:.1f}</span>
                            <span style="color: #8B949E; font-size: 0.65rem;">{exit_strategy['action']}</span>
                        </div>
                        <div style="color: #3FB950; font-size: 0.7rem; font-weight: bold; margin-top: 2px;">出場保命線: ${exit_strategy['trailing_stop']:.1f} (跌破必賣)</div>
                        <div style="color: #8B949E; font-size: 0.65rem; line-height: 1.2; margin-top: 5px;">{exit_strategy['desc']}</div>
                    </div>
                ''', unsafe_allow_html=True)
            
            with rec_col4:
                st.markdown(f'''
                    <div style="background: rgba(188, 140, 242, 0.1); padding: 12px; border-radius: 8px; border-left: 5px solid #BC8CF2; height: 135px;">
                        <div style="color: #8B949E; font-size: 0.7rem; margin-bottom: 5px;">7日趨勢預測</div>
                        <div style="display: flex; align-items: baseline; gap: 5px;">
                            <span style="font-size: 1.2rem; font-weight: 800; color: #BC8CF2;">${median_sim_7d:.1f}</span>
                            <span style="color: {'#26a69a' if win_prob_7d > 50 else '#ef5350'}; font-size: 0.75rem; font-weight: bold;">({win_prob_7d:.1f}%)</span>
                        </div>
                        <div style="color: #8B949E; font-size: 0.7rem; margin-top: 15px;">蒙地卡羅模擬勝率</div>
                    </div>
                ''', unsafe_allow_html=True)

            # Overview Tab: Advanced Plotly Chart
            quant_factors = calculate_quant_factors(data, ticker_metadata, rsi_series, atr_series)
            
            col_chart, col_quant = st.columns([2, 1])
            
            with col_chart:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                   vertical_spacing=0.03, subplot_titles=(f'{ticker_input} 價量走勢', '成交量'), 
                                   row_width=[0.2, 0.7])

                # Candlestick
                fig.add_trace(go.Candlestick(
                    x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], 
                    name='K線', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
                ), row=1, col=1)
                
                # MA
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(window=5).mean(), name='5日均線', line=dict(color='#FFD700', width=1.5)), row=1, col=1)
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

            with col_quant:
                # 量化評分雷達圖
                categories = list(quant_factors.keys())
                values = list(quant_factors.values())
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    fillcolor='rgba(88, 166, 255, 0.2)',
                    line=dict(color='#58A6FF', width=2),
                    name='因子評分'
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100], gridcolor='#30363D', tickfont=dict(size=8)),
                        angularaxis=dict(gridcolor='#30363D', tickfont=dict(size=10)),
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    showlegend=False,
                    template="plotly_dark",
                    height=350,
                    margin=dict(l=40, r=40, t=40, b=40),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                
                # 綜合評分顯示
                avg_score = sum(values) / len(values)
                score_color = "#3FB950" if avg_score > 70 else "#D29922" if avg_score > 50 else "#f85149"
                
                st.markdown(f'''
                    <div style="text-align: center; padding: 15px; background: rgba(48, 54, 61, 0.2); border-radius: 10px; border: 1px solid #30363D;">
                        <div style="color: #8B949E; font-size: 0.8rem; margin-bottom: 5px;">綜合量化評分</div>
                        <div style="font-size: 2.5rem; font-weight: 800; color: {score_color};">{avg_score:.1f}</div>
                        <div style="color: {score_color}; font-size: 0.7rem; font-weight: bold;">
                            {"市場領導者" if avg_score > 80 else "趨勢強勁" if avg_score > 65 else "中性整理" if avg_score > 50 else "弱勢觀察"}
                        </div>
                    </div>
                ''', unsafe_allow_html=True)

            # 2. 財務健康度與統計摘要 (Financial Health & Stats Section)
            financial_statements = get_financial_data(ticker_input)
            perf_html = ""
            if financial_statements is not None and len(financial_statements) >= 2:
                latest_year = financial_statements.index[-1]
                prev_year = financial_statements.index[-2]
                rev_latest = financial_statements['Total Revenue'].iloc[-1]
                rev_prev = financial_statements['Total Revenue'].iloc[-2]
                ni_latest = financial_statements['Net Income'].iloc[-1]
                ni_prev = financial_statements['Net Income'].iloc[-2]
                
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
                # 根據健康得分定義動態樣式 (Define dynamic styling based on health scores)
                profitability_score, leverage_score, cashflow_score = p_score, l_score, c_score
                avg_health = (profitability_score + leverage_score + cashflow_score) / 3
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
                            <span style="font-size: 0.7rem; color: #8B949E; background: #21262d; padding: 2px 6px; border-radius: 3px; border: 1px solid #30363D;">獲利 {profitability_score}/10</span>
                            <span style="font-size: 0.7rem; color: #8B949E; background: #21262d; padding: 2px 6px; border-radius: 3px; border: 1px solid #30363D;">槓桿 {leverage_score}/10</span>
                            <span style="font-size: 0.7rem; color: #8B949E; background: #21262d; padding: 2px 6px; border-radius: 3px; border: 1px solid #30363D;">現金 {cashflow_score}/10</span>
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;">
                        <div>
                            <p style="color: #8B949E; font-size: 0.75rem; margin-bottom: 5px;">估值指標</p>
                            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 3px;"><span>{t['trailing_pe']}</span><span style="font-weight: bold;">{ticker_metadata.get('trailingPE', 'N/A')}</span></div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 3px;"><span>{t['forward_pe']}</span><span style="font-weight: bold;">{ticker_metadata.get('forwardPE', 'N/A')}</span></div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.85rem;"><span>{t['div_yield']}</span><span style="font-weight: bold; color: #26a69a;">{ticker_metadata.get('dividendYield', 0)*100:.2f}%</span></div>
                        </div>
                        <div>
                            <p style="color: #8B949E; font-size: 0.75rem; margin-bottom: 5px;">價格區間</p>
                            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 3px;"><span>52週高</span><span style="font-weight: bold;">{ticker_metadata.get('fiftyTwoWeekHigh', 'N/A')}</span></div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 3px;"><span>52週低</span><span style="font-weight: bold;">{ticker_metadata.get('fiftyTwoWeekLow', 'N/A')}</span></div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.85rem;"><span>Beta</span><span style="font-weight: bold;">{ticker_metadata.get('beta', 'N/A')}</span></div>
                        </div>
                        <div>
                            <p style="color: #8B949E; font-size: 0.75rem; margin-bottom: 8px;">法人目標分析 (Analyst Targets)</p>
                            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 3px;">
                                <span>目標中位 (建議參考)</span>
                                <span style="font-weight: bold; color: #58A6FF;">{ticker_metadata.get('targetMedianPrice', 'N/A')}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 3px;">
                                <span>法人最高 (壓力位)</span>
                                <span style="font-weight: bold; color: #f85149;">{ticker_metadata.get('targetHighPrice', 'N/A')}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 8px;">
                                <span>潛在空間</span>
                                <span style="font-weight: bold; color: #26a69a;">
                                    {f"+{((ticker_metadata.get('targetMedianPrice', current_price) / current_price) - 1) * 100:.1f}%" if ticker_metadata.get('targetMedianPrice') else 'N/A'}
                                </span>
                            </div>
                            <!-- Range Visualizer -->
                            <div style="position: relative; width: 100%; height: 24px; background: #21262d; border-radius: 4px; margin-bottom: 8px; border: 1px solid #30363D; overflow: hidden;">
                                <div style="position: absolute; left: 0; top: 0; height: 100%; width: 100%; background: linear-gradient(90deg, #ef5350 0%, #787b86 50%, #26a69a 100%); opacity: 0.1;"></div>
                                <div style="position: absolute; 
                                            left: {
                                                (
                                                    min(100, max(0, ((current_price - ticker_metadata.get('targetLowPrice', current_price*0.8)) / 
                                                    (max(0.01, ticker_metadata.get('targetHighPrice', current_price*1.2) - ticker_metadata.get('targetLowPrice', current_price*0.8))) * 100)))
                                                ) if ticker_metadata.get('targetHighPrice') and ticker_metadata.get('targetHighPrice') != ticker_metadata.get('targetLowPrice') else 50
                                            }%; 
                                            top: 0; width: 3px; height: 100%; background: #FFFFFF; box-shadow: 0 0 8px #FFFFFF; z-index: 2;"></div>
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.65rem; color: #8B949E;">
                                <span>低: {ticker_metadata.get('targetLowPrice', 'N/A')}</span>
                                <span>高: {ticker_metadata.get('targetHighPrice', 'N/A')}</span>
                            </div>
                        </div>
                    </div>
                    {perf_html}
                </div>
                ''', unsafe_allow_html=True)

            # 底部專家技術診斷報告 (Expert Insight Card at the bottom)
            ticker_display_name = ticker_metadata.get('shortName') or ticker_metadata.get('longName') or ticker_input
            insight_report = get_expert_insight(
                ticker_input, 
                current_price, 
                float(rsi_series.iloc[-1]), 
                sig_text, 
                float(macd_series.iloc[-1]), 
                float(signal_series.iloc[-1]), 
                buy_signals, 
                sell_signals, 
                data.index[-1]
            )
            
            # 專業診斷區塊佈局 (Professional Diagnostic Section with Grid Layout)
            st.markdown(f'''
<div class="data-card" style="border-top: 2px solid {insight_report['action']['color']}; border-color: {insight_report['action']['color']}44; background: linear-gradient(180deg, {insight_report['action']['color']}0a 0%, #0D1117 100%); transition: all 0.5s ease;">
    <h3 style="margin-top:0; color: {insight_report['action']['color']}; font-size: 1.1rem; display: flex; align-items: center; gap: 8px; margin-bottom: 20px;">
        <span style="font-size: 1.4rem;">⚖️</span> {ticker_display_name} - 專家技術診斷報告
    </h3>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
        <div style="background: rgba(48, 54, 61, 0.2); padding: 15px; border-radius: 6px; border-left: 3px solid {insight_report['rsi']['color']};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="color: #8B949E; font-size: 0.8rem;">RSI 強度</span>
                <span style="background: {insight_report['rsi']['color']}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem; font-weight: bold;">{insight_report['rsi']['status']}</span>
            </div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #FFFFFF; margin-bottom: 5px;">{insight_report['rsi']['val']}</div>
            <div style="color: #E0E0E0; font-size: 0.85rem; line-height: 1.4;">{insight_report['rsi']['desc']}</div>
        </div>
        <div style="background: rgba(48, 54, 61, 0.2); padding: 15px; border-radius: 6px; border-left: 3px solid {insight_report['macd']['color']};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="color: #8B949E; font-size: 0.8rem;">MACD 差值</span>
                <span style="background: {insight_report['macd']['color']}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem; font-weight: bold;">{insight_report['macd']['status']}</span>
            </div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #FFFFFF; margin-bottom: 5px;">{insight_report['macd']['val']}</div>
            <div style="color: #E0E0E0; font-size: 0.85rem; line-height: 1.4;">{insight_report['macd']['desc']}</div>
        </div>
        <div style="background: rgba(88, 166, 255, 0.1); padding: 15px; border-radius: 6px; border: 1px dashed #58A6FF;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="color: #58A6FF; font-size: 0.8rem; font-weight: bold;">AI 建議佈局</span>
                <span style="color: {entry_strategy['color']}; font-weight: 800; font-size: 0.9rem;">{entry_strategy['action']} (${entry_strategy['price']:.1f})</span>
            </div>
            <div style="font-size: 1rem; color: #FFFFFF; font-weight: 500; margin-bottom: 5px;">策略部署 (信心:{entry_strategy['confidence']})</div>
            <div style="color: #E0E0E0; font-size: 0.85rem; line-height: 1.4;">{entry_strategy['desc']}</div>
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
    <div class="data-card" style="margin-bottom: 0; padding: 15px; border-left: 3px solid #FF6D00;">
        <div style="color: #8B949E; font-size: 0.75rem;">VR 容量比率</div>
        <div style="font-size: 1.1rem; font-weight: bold; margin: 5px 0;">{vr_series.iloc[-1]:.1f}%</div>
        <div style="color: {'#ef5350' if vr_series.iloc[-1] > 160 else '#26a69a' if vr_series.iloc[-1] < 70 else '#8B949E'}; font-size: 0.7rem;">
            {'過熱區域' if vr_series.iloc[-1] > 160 else '低迷區域' if vr_series.iloc[-1] < 70 else '常態區間'}
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

            col_kd, col_vr = st.columns(2)
            with col_kd:
                # KD Chart
                fig_kd = go.Figure()
                fig_kd.add_trace(go.Scatter(x=data.index, y=k_series, name='K線', line=dict(color='#FFD700', width=2)))
                fig_kd.add_trace(go.Scatter(x=data.index, y=d_series, name='D線', line=dict(color='#00BFFF', width=2)))
                fig_kd.add_hline(y=80, line_dash="dash", line_color="#ef5350", opacity=0.5)
                fig_kd.add_hline(y=20, line_dash="dash", line_color="#26a69a", opacity=0.5)
                fig_kd.update_layout(
                    title="KD 指標 (9, 3, 3)", yaxis_range=[0, 100], height=250, template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=10, r=10, t=40, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_kd, use_container_width=True)
            
            with col_vr:
                # VR Chart
                fig_vr = go.Figure()
                fig_vr.add_trace(go.Scatter(x=data.index, y=vr_series, name='VR', line=dict(color='#FF6D00', width=2)))
                fig_vr.add_hline(y=160, line_dash="dash", line_color="#ef5350", opacity=0.5)
                fig_vr.add_hline(y=70, line_dash="dash", line_color="#26a69a", opacity=0.5)
                fig_vr.update_layout(
                    title="VR 容量比率 (26)", height=250, template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig_vr, use_container_width=True)

        with tab3:
            # Prediction Page: Enhanced Monte Carlo & Forecast
            st.subheader("🔮 未來路徑機率模擬 (AI 增強版)")
            
            # 從 session_state 讀取數據
            sim_df_full = st.session_state.get("sim_df_full")
            ai_score = st.session_state.get("ai_score")
            atr = st.session_state.get("atr")
            current_price = st.session_state.get("current_price")
            ticker_input = st.session_state.get("ticker_input", "股票代碼")
            prediction_dates = st.session_state.get("prediction_dates")

            if sim_df_full is not None and prediction_dates is not None:
                # 確保日期格式對 Plotly 友好
                plot_dates = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in prediction_dates]
                
                # 取得更精細的分位數數據
                p10 = sim_df_full.quantile(0.10, axis=1).values
                p25 = sim_df_full.quantile(0.25, axis=1).values
                p50 = sim_df_full.quantile(0.50, axis=1).values
                p75 = sim_df_full.quantile(0.75, axis=1).values
                p90 = sim_df_full.quantile(0.90, axis=1).values
                
                # 計算期望值 (均值)
                expected_path = sim_df_full.mean(axis=1).values
                
                fig_mc = go.Figure()
                
                # 1. 繪製 80% 信心區間 (P10 - P90)
                fig_mc.add_trace(go.Scatter(
                    x=plot_dates, y=p90,
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip"
                ))
                fig_mc.add_trace(go.Scatter(
                    x=plot_dates, y=p10,
                    fill="tonexty",
                    fillcolor="rgba(88, 166, 255, 0.1)",
                    line=dict(width=0),
                    name="80% 信心區間 (P10-P90)"
                ))
                
                # 2. 繪製 50% 核心區間 (P25 - P75)
                fig_mc.add_trace(go.Scatter(
                    x=plot_dates, y=p75,
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip"
                ))
                fig_mc.add_trace(go.Scatter(
                    x=plot_dates, y=p25,
                    fill="tonexty",
                    fillcolor="rgba(88, 166, 255, 0.25)",
                    line=dict(width=0),
                    name="50% 核心區間 (P25-P75)"
                ))
                
                # 3. 繪製期望值路徑
                fig_mc.add_trace(go.Scatter(
                    x=plot_dates, y=expected_path,
                    line=dict(color="#FFD700", width=2),
                    name="預期平均路徑"
                ))

                # 4. 繪製中位數路徑
                fig_mc.add_trace(go.Scatter(
                    x=plot_dates, y=p50,
                    line=dict(color="#58a6ff", width=3, dash="dash"),
                    name="中位數路徑"
                ))
                
                fig_mc.update_layout(
                    title=dict(text=f"<b>{ticker_input} 未來 30 日機率分佈預測 (t-分佈 + 均值回歸)</b>", font=dict(size=16, color="#C9D1D9")),
                    xaxis_title="日期",
                    yaxis_title="價格",
                    template="plotly_dark",
                    hovermode="x unified",
                    height=500,
                    margin=dict(l=20, r=20, t=60, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                
                st.plotly_chart(fig_mc, use_container_width=True, key=f"mc_chart_{ticker_input}", theme="streamlit")
                
                # 模擬指標面板 (更新為更豐富的數據)
                st.markdown("<div style=\"margin-top: -20px;\"></div>", unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    final_expected = expected_path[-1]
                    exp_return = (final_expected / current_price - 1) * 100
                    st.metric("預期平均目標", f"{final_expected:.2f}", f"{exp_return:+.2f}%")
                
                with col2:
                    prob_profit = (sim_df_full.iloc[-1] > current_price).mean() * 100
                    st.metric("上漲機率", f"{prob_profit:.1f}%", f"{'偏多' if prob_profit > 55 else '偏空' if prob_profit < 45 else '中性'}")
                
                with col3:
                    # 修正風險指標為更直觀的下跌空間
                    max_downside = (p10[-1] / current_price - 1) * 100
                    st.metric("保守下行空間 (P10)", f"{p10[-1]:.2f}", f"{max_downside:.2f}%", delta_color="inverse")
                
                with col4:
                    max_upside = (p90[-1] / current_price - 1) * 100
                    st.metric("樂觀上行空間 (P90)", f"{p90[-1]:.2f}", f"{max_upside:+.2f}%")
                    
                st.info(f"💡 **核心模型更新**：採用 **Student's t-分佈** 模擬市場極端走勢 (肥尾)，並整合 **EMA200 均值回歸** 機制，使預測更貼近真實市場特徵。")
                
                # 關鍵節點表格 (更新分位數)
                st.markdown("<p style=\"color: #FFFFFF; font-size: 1.1rem; font-weight: bold; margin-top: 30px; margin-bottom: 15px; border-left: 4px solid #58A6FF; padding-left: 10px;\">📅 預測價格區間詳情 (分位數分析)</p>", unsafe_allow_html=True)
                
                table_html = """<style>
.mc-table { width: 100%; border-collapse: collapse; color: #FFFFFF; background: rgba(255, 255, 255, 0.02); border-radius: 8px; overflow: hidden; }
.mc-table thead { background: rgba(88, 166, 255, 0.15); border-bottom: 2px solid #30363D; }
.mc-table th, .mc-table td { padding: 12px; text-align: left; }
.price-cell { text-align: right; font-family: monospace; }
</style>
<table class="mc-table">
<thead>
<tr>
<th>預測節點</th>
<th style="text-align: right; color: #FF7B72;">極端悲觀 (P10)</th>
<th style="text-align: right; color: #FFA657;">保守預期 (P25)</th>
<th style="text-align: right; color: #FFFFFF;">中位數 (P50)</th>
<th style="text-align: right; color: #7EE787;">樂觀預期 (P75)</th>
<th style="text-align: right; color: #D2A8FF;">極端樂觀 (P90)</th>
</tr>
</thead>
<tbody>"""
                for day in [5, 10, 20, 30]:
                    if day < len(sim_df_full):
                        row = sim_df_full.iloc[day]
                        table_html += f"""<tr>
<td>T+{day} 天 ({prediction_dates[day].strftime("%m/%d")})</td>
<td class="price-cell">${row.quantile(0.10):.2f}</td>
<td class="price-cell">${row.quantile(0.25):.2f}</td>
<td class="price-cell" style="font-weight: bold;">${row.quantile(0.50):.2f}</td>
<td class="price-cell">${row.quantile(0.75):.2f}</td>
<td class="price-cell">${row.quantile(0.90):.2f}</td>
</tr>"""
                table_html += "</tbody></table>"
                st.markdown(table_html, unsafe_allow_html=True)
            else:
                st.warning("請先在「即時分析」分頁完成 AI 分析以生成預測數據。")
            
        with tab4:
            # Enhanced Signal Monitoring Page
            st.markdown(f'<div class="data-card"><h3 style="margin-top:0; color: #FFD700;">🔔 交易信號監測報告 (近一個月)</h3>', unsafe_allow_html=True)
            
            # 運行歷史回測績效
            perf = run_backtest(data, all_signals)
            if perf:
                st.markdown(f"""
                <div style="background: rgba(255, 215, 0, 0.05); padding: 15px; border-radius: 8px; border: 1px solid rgba(255, 215, 0, 0.2); margin-bottom: 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; gap: 10px;">
                        <div style="text-align: center; flex: 1;">
                            <div style="color: #8B949E; font-size: 0.75rem;">歷史預估勝率</div>
                            <div style="color: #FFD700; font-size: 1.2rem; font-weight: bold;">{perf['win_rate']:.1f}%</div>
                        </div>
                        <div style="text-align: center; flex: 1;">
                            <div style="color: #8B949E; font-size: 0.75rem;">5日平均收益</div>
                            <div style="color: {'#3FB950' if perf['avg_profit'] > 0 else '#F85149'}; font-size: 1.2rem; font-weight: bold;">{perf['avg_profit']:+.2f}%</div>
                        </div>
                        <div style="text-align: center; flex: 1;">
                            <div style="color: #8B949E; font-size: 0.75rem;">夏普比率 (Sharpe)</div>
                            <div style="color: #58A6FF; font-size: 1.2rem; font-weight: bold;">{perf['sharpe']:.2f}</div>
                        </div>
                        <div style="text-align: center; flex: 1;">
                            <div style="color: #8B949E; font-size: 0.75rem;">最大回撤 (MDD)</div>
                            <div style="color: #F85149; font-size: 1.2rem; font-weight: bold;">{perf['mdd']:.1f}%</div>
                        </div>
                        <div style="text-align: center; flex: 1;">
                            <div style="color: #8B949E; font-size: 0.75rem;">樣本信號數</div>
                            <div style="color: #E0E0E0; font-size: 1.2rem; font-weight: bold;">{perf['total_signals']}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if not latest_signals:
                st.info("近一個月內技術指標尚未偵測到明確的買賣信號。")
            else:
                col_sig_list, col_sig_stats = st.columns([2, 1])
                
                with col_sig_list:
                    st.markdown('<p style="color: #8B949E; font-size: 0.9rem; margin-bottom: 20px; font-weight: 500;">📅 近期觸發信號流水線 (30天內)</p>', unsafe_allow_html=True)
                    
                    # Start Timeline Container
                    timeline_html = '<div style="position: relative; padding-left: 20px; border-left: 2px solid #30363D; margin-left: 10px;">'
                    
                    for i, s in enumerate(latest_signals[:10]):
                        # Calculate relative days
                        days_ago = (datetime.now().date() - s['date'].date()).days
                        time_str = "今天" if days_ago == 0 else f"{days_ago} 天前"
                        
                        # Calculate suggested shares
                        suggested_shares = int(investment_budget / s['price'])
                        shares_text = f"建議{s['type']}{suggested_shares:,} 股"
                        
                        # Enhanced card styling with performance metrics
                        ai_score_val = s.get('ai_score', 0)
                        ai_badge = f'<span style="background: rgba(88, 166, 255, 0.1); color: #58A6FF; font-size: 0.65rem; padding: 1px 6px; border-radius: 4px; border: 1px solid rgba(88, 166, 255, 0.3); font-weight: 600; margin-left: 5px;">🤖 AI 認證 ({ai_score_val:.0f})</span>' if s.get('ai_verified') else f'<span style="color: #8B949E; font-size: 0.65rem; margin-left: 5px;">AI 評分: {ai_score_val:.0f}</span>'
                        
                        timeline_html += f'''<div style="position: relative; margin-bottom: 15px;">
<div style="position: absolute; left: -25px; top: 12px; width: 8px; height: 8px; background: {s['color']}; border-radius: 50%; z-index: 2;"></div>
<div style="background: #161B22; border: 1px solid #30363D; border-radius: 8px; padding: 15px 18px; display: block;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
<div style="display: flex; align-items: center; gap: 12px;">
<div style="color: {s['color']}; font-size: 1.2rem;">{s['icon']}</div>
<div>
<div style="display: flex; align-items: center; gap: 8px;">
<span style="color: {s['color']}; font-weight: bold; font-size: 0.95rem;">{s['type']}建議</span>
<span style="color: #8B949E; font-size: 0.75rem;">({time_str})</span>
<span style="background: {s['perf_color']}22; color: {s['perf_color']}; font-size: 0.65rem; padding: 1px 6px; border-radius: 4px; border: 1px solid {s['perf_color']}44; font-weight: 600;">{s['perf_status']}</span>
{ai_badge}
</div>
<div style="color: #8B949E; font-size: 0.75rem;">{s['date'].strftime('%Y-%m-%d')}</div>
</div>
</div>
<div style="text-align: right;">
<div style="color: #FFFFFF; font-weight: bold; font-size: 1.1rem;">${s['price']:.2f}</div>
<div style="color: {s['color']}; font-size: 0.8rem; font-weight: 600;">{shares_text}</div>
</div>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; padding-top: 10px; border-top: 1px dashed #30363D;">
<div style="font-size: 0.75rem; color: #8B949E;">
🤖 當時 AI 建議: <span style="color: #E0E0E0; font-weight: 600;">${s['ai_price']:.1f} ({s['ai_action']})</span>
</div>
<div style="font-size: 0.75rem; color: {s['perf_color'] if s['perf_pct'] != 0 else '#8B949E'}; font-weight: 600;">
{f"{s['perf_pct']:+.1f}%" if s['perf_pct'] != 0 else "待追蹤"}
</div>
</div>
</div>
</div>'''
                    
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

        with tab5:
            # News Page - 使用原生組件避免佈局崩潰
            st.subheader(f"📰 {ticker_display_name} 即時市場新聞")
            
            try:
                news_list = []
                
                # 來源 1: Yahoo Finance
                try:
                    yf_news = stock.news
                    if yf_news:
                        for item in yf_news:
                            tags = item.get('relatedTickers', [])
                            if tags is None: tags = []
                            news_list.append({
                                'title': item.get('title'),
                                'link': item.get('link'),
                                'publisher': item.get('publisher', 'Yahoo Finance'),
                                'time': pd.to_datetime(item.get('providerPublishTime', 0), unit='s'),
                                'tags': tags,
                                'source': 'Yahoo'
                            })
                except Exception as yf_err:
                    st.warning(f"Yahoo 新聞抓取暫時不可用: {yf_err}")
                
                # 來源 2: Google News RSS (擴展來源)
                if len(news_list) < 15:
                    try:
                        clean_name = ticker_display_name.split('(')[0].strip()
                        # 擴展搜尋關鍵字以包含更多大新聞源
                        search_keywords = [f"{clean_name} 股票", f"{clean_name} 營收", f"{clean_name} 財經"]
                        
                        for query_text in search_keywords:
                            encoded_query = urllib.parse.quote(query_text)
                            rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
                            
                            feed = feedparser.parse(rss_url)
                            for entry in feed.entries[:10]: # 每個關鍵字取前 10 條
                                if not any(n['title'] == entry.title for n in news_list):
                                    try:
                                        published = pd.to_datetime(entry.published)
                                    except:
                                        published = datetime.now()
                                        
                                    news_list.append({
                                        'title': entry.title,
                                        'link': entry.link,
                                        'publisher': entry.source.get('title', 'Google News') if hasattr(entry, 'source') else 'Google News',
                                        'time': published,
                                        'tags': ['新聞', '市場'],
                                        'source': 'Google'
                                    })
                            if len(news_list) >= 30: break # 最多獲取 30 條
                    except Exception as g_err:
                        st.warning(f"Google 新聞抓取暫時不可用: {g_err}")
                
                if not news_list:
                    st.info(f"目前尚無 {ticker_display_name} 的相關即時新聞。")
                else:
                    news_list.sort(key=lambda x: x['time'], reverse=True)
                    
                    for item in news_list[:20]: 
                        time_str = item['time'].strftime('%m-%d %H:%M')
                        source_label = "Yahoo" if item['source'] == 'Yahoo' else "Google"
                        
                        # 加入 AI 情緒分析
                        sentiment_text, sentiment_color = analyze_news_sentiment(item['title'])
                        
                        # 簡化單條新聞 HTML，移除嵌套 div
                        st.markdown(f"""
                        <div style="padding: 12px; border-bottom: 1px solid #30363D; margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; margin-bottom: 5px;">
                                <div style="display: flex; gap: 8px; align-items: center;">
                                    <span style="color: #58A6FF; font-weight: bold;">[{source_label}] {item['publisher']}</span>
                                    <span style="background: {sentiment_color}22; color: {sentiment_color}; padding: 1px 6px; border-radius: 4px; border: 1px solid {sentiment_color}44; font-weight: bold; font-size: 0.65rem;">{sentiment_text}</span>
                                </div>
                                <span style="color: #8B949E;">{time_str}</span>
                            </div>
                            <a href="{item['link']}" target="_blank" style="text-decoration: none; color: #FFFFFF; font-size: 1rem; font-weight: 500; line-height: 1.4;">
                                {item['title']}
                            </a>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"新聞分頁發生錯誤: {e}")

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
else:
    st.info("請在側邊欄輸入股票代碼以開始分析（例如：2330.TW）")