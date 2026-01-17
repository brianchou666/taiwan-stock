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
import re
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
    
    /* Force Dark Theme for Streamlit Elements */
    [data-testid="stHeader"], [data-testid="stSidebar"], .stApp {
        background-color: #0E1117 !important;
        color: #E0E0E0 !important;
    }
    
    /* Target Settings Dialog and Menu */
    div[role="dialog"], div[data-testid="stMenu"] {
        background-color: #161B22 !important;
        color: #E0E0E0 !important;
        border: 1px solid #30363D !important;
    }
    
    button {
        color: #E0E0E0 !important;
    }
    
    /* Fix for SVG icons in menu */
    svg {
        fill: #E0E0E0 !important;
    }
    
    /* Style Streamlit Tooltip Icons to blend with Dark Theme */
    .stTooltipIcon {
        color: #58A6FF !important;
        background: rgba(88, 166, 255, 0.1);
        border-radius: 50%;
        padding: 2px;
        opacity: 0.6;
        transition: all 0.3s ease;
        transform: scale(1.1);
    }
    .stTooltipIcon:hover {
        opacity: 1;
        background: rgba(88, 166, 255, 0.2);
        transform: scale(1.25);
        box-shadow: 0 0 12px rgba(88, 166, 255, 0.4);
    }
    .stTooltipIcon svg {
        stroke: #58A6FF !important;
        width: 18px !important;
        height: 18px !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161B22 !important;
        border-right: 1px solid #30363D;
    }
    
    /* Hide Sidebar Scrollbar */
    section[data-testid="stSidebar"] > div {
        overflow: hidden !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-top: 2rem !important;
        overflow: hidden !important;
    }
    
    /* Card/Container Styling */
    .data-card {
        background: #161B22;
        border: 1px solid #30363D;
        padding: 15px;
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

    /* News Item Styling */
    .news-item {
        padding: 16px;
        border-bottom: 1px solid #30363D;
        transition: all 0.2s ease;
    }
    .news-item:hover {
        background: rgba(88, 166, 255, 0.05);
        border-left: 3px solid #58A6FF;
        padding-left: 13px; /* Adjust for border-left */
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
    "predicted_price": "推估價格",
    "tech_analysis": "技術指標詳解",
    "signal_label": "信號線",
    "backtest_header": "預測準確度回測",
    "backtest_desc": "使用過去 7 天的數據驗證模型準確性。",
    "target_price": "目標價參考資訊"
}

def resolve_ticker(ticker):
    """
    智能解析股票代碼 (Intelligently resolves ticker symbols)
    自動處理台股後綴 (.TW, .TWO) 並校正格式，支援美股與台股
    """
    if not ticker:
        return "2330.TW"
        
    ticker = ticker.strip().upper()
    
    # 如果已經有正確後綴，直接返回
    if ticker.endswith(".TW") or ticker.endswith(".TWO"):
        return ticker
        
    # 處理帶點但後綴不正確的情況 (例如 2330.T)
    if "." in ticker:
        base = ticker.split(".")[0]
        if re.match(r'^\d{4,6}$', base):
            return f"{base}.TW"
        return ticker
        
    # 如果全是數字 (台股代碼)
    if re.match(r'^\d{4,6}$', ticker):
        # 預設優先匹配上市 (.TW)，get_stock_data 會在失敗時嘗試 .TWO
        return f"{ticker}.TW"
            
    # 如果是常見的美股或國際代碼 (例如 AAPL, TSLA)
    return ticker

def get_market_colors(ticker):
    """
    根據市場獲取顏色方案 (Get color scheme based on market)
    台股: 漲紅跌綠 (Taiwan: Up Red, Down Green)
    美股/國際: 漲綠跌紅 (US/Intl: Up Green, Down Red)
    """
    ticker_upper = ticker.upper()
    is_taiwan = ".TW" in ticker_upper or ".TWO" in ticker_upper or re.match(r'^\d{4,6}$', ticker)
    if is_taiwan:
        return {
            "up": "#ef5350",     # 紅色
            "down": "#26a69a",   # 綠色
            "up_bg": "rgba(239, 83, 80, 0.1)",
            "down_bg": "rgba(38, 166, 154, 0.1)",
            "buy": "#ef5350",    # 台股買入通常標示紅色
            "sell": "#26a69a"    # 台股賣出通常標示綠色
        }
    else:
        return {
            "up": "#26a69a",     # 綠色
            "down": "#ef5350",   # 紅色
            "up_bg": "rgba(38, 166, 154, 0.1)",
            "down_bg": "rgba(239, 83, 80, 0.1)",
            "buy": "#26a69a",    # 美股買入綠色
            "sell": "#ef5350"    # 美股賣出紅色
        }

@st.cache_data(ttl=3600) # 增加快取過期時間
def get_stock_data(ticker, period, interval):
    """
    獲取股票歷史數據 (Fetches historical stock data from yfinance)
    增強容錯處理，自動切換上市/上櫃後綴
    """
    resolved_ticker = resolve_ticker(ticker)
    
    def download_with_retry(t_code):
        try:
            d = yf.download(t_code, period=period, interval=interval, progress=False, auto_adjust=True)
            if d is None or d.empty:
                return None
            
            # 處理 yfinance v0.2.x+ 的 MultiIndex 問題
            if isinstance(d.columns, pd.MultiIndex):
                # 尋找 Close 所在的層級
                if t_code in d.columns.get_level_values(1):
                    d = d.xs(t_code, level=1, axis=1)
                else:
                    d.columns = d.columns.get_level_values(0)
            
            # 確保必要的列存在
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            if all(col in d.columns for col in required):
                return d
            return None
        except Exception:
            return None

    # 第一輪嘗試
    data = download_with_retry(resolved_ticker)
    
    # 如果失敗且是台股，嘗試切換後綴
    if (data is None or data.empty) and (".TW" in resolved_ticker or ".TWO" in resolved_ticker):
        alt_ticker = resolved_ticker.replace(".TW", ".TWO") if ".TW" in resolved_ticker else resolved_ticker.replace(".TWO", ".TW")
        data = download_with_retry(alt_ticker)
        if data is not None and not data.empty:
            resolved_ticker = alt_ticker
            
    # 如果還是失敗，嘗試原始輸入 (針對美股或特殊代號)
    if (data is None or data.empty) and resolved_ticker != ticker.upper():
        data = download_with_retry(ticker.upper())
        if data is not None and not data.empty:
            resolved_ticker = ticker.upper()
            
    return data, resolved_ticker

@st.cache_data(ttl=86400) # 財務數據快取 24 小時
def get_financial_data(ticker):
    """
    獲取財務數據趨勢 (Fetches financial data for trends)
    支援年度與季度數據備援，並增強列名解析
    """
    resolved_ticker = resolve_ticker(ticker)
    try:
        stock = yf.Ticker(resolved_ticker)
        
        # 嘗試獲取年度數據，若無則嘗試季度數據
        financials = stock.financials
        if financials is None or financials.empty:
            financials = stock.quarterly_financials
            
        # 再次檢查切換後綴
        if (financials is None or financials.empty) and ".TW" in resolved_ticker:
            stock = yf.Ticker(resolved_ticker.replace(".TW", ".TWO"))
            financials = stock.financials
            if financials is None or financials.empty:
                financials = stock.quarterly_financials
        
        if financials is not None and not financials.empty:
            available_cols = [str(c).strip() for c in financials.index]
            
            # 定義可能的列名映射
            rev_names = ['Total Revenue', 'Total Operating Revenue', 'Operating Revenue', 'Revenue']
            ni_names = ['Net Income', 'Net Income Common Stockholders', 'Net Income From Continuing Ops', 'Net Profit']
            
            target_rev = next((name for name in rev_names if name in available_cols), None)
            target_ni = next((name for name in ni_names if name in available_cols), None)
            
            if not target_rev or not target_ni:
                return None
                
            data = financials.T[[target_rev, target_ni]].copy()
            data.columns = ['Total Revenue', 'Net Income']
            
            # 處理日期索引，轉換為年份或 YYYY-QQ
            new_index = []
            for idx in data.index:
                try:
                    new_index.append(idx.strftime('%Y-%m-%d'))
                except:
                    new_index.append(str(idx))
            data.index = new_index
            
            return data.sort_index()
        return None
    except Exception:
        return None

def get_signal_score(data, rsi_val, macd_val, signal_val, supertrend_dir=None, m_colors=None):
    """
    綜合信號評分系統 v3.5 (波段交易優化版)
    優先考慮中長期趨勢共振與量價結構，過濾短期雜訊
    """
    # 預設顏色方案 (Default colors if m_colors not provided)
    if m_colors is None:
        m_colors = {
            "up": "#26a69a", "down": "#ef5350", "buy": "#26a69a", "sell": "#ef5350"
        }
    
    score_pts = 0
    current_price = float(data['Close'].iloc[-1])
    
    # 1. 中長期趨勢共振 (Trend Resonance) - 權重高
    ema20 = float(data['Close'].ewm(span=20).mean().iloc[-1])
    ema50 = float(data['Close'].ewm(span=50).mean().iloc[-1])
    ema200 = float(data['Close'].ewm(span=200).mean().iloc[-1])
    
    if ema20 > ema50 > ema200: score_pts += 2 # 多頭完美排列
    elif current_price > ema50: score_pts += 1 # 站上關鍵生命線
    
    # 2. SuperTrend (波段核心指標)
    if supertrend_dir is not None and supertrend_dir == 1: score_pts += 1
    
    # 3. MACD 動能 (MACD Momentum)
    if macd_val > signal_val: score_pts += 1
    
    # 4. RSI 強勢區間 (RSI Strength)
    # 波段交易看重強勢而非超賣
    if 50 < rsi_val < 75: score_pts += 1 
    elif rsi_val < 35: score_pts += 1 # 極端超跌機會
    
    # 5. 量能增溫 (Volume Support)
    v_ma20 = data['Volume'].rolling(20).mean().iloc[-1]
    if data['Volume'].iloc[-1] > v_ma20 * 1.3:
        if data['Close'].iloc[-1] > data['Open'].iloc[-1]: score_pts += 1
        
    # Mapping score to Rating
    if score_pts >= 5: return "波段強勢", m_colors["buy"], 90
    elif score_pts >= 3: return "趨勢偏多", m_colors["buy"], 75
    elif score_pts >= 1: return "震盪整理", "#787b86", 55
    elif score_pts <= -2: return "波段轉弱", m_colors["sell"], 20
    return "趨勢不明", "#787b86", 45

def analyze_news_sentiment(title, m_colors=None):
    """
    簡易金融新聞情緒分析
    """
    # 預設顏色方案 (Default colors if m_colors not provided)
    if m_colors is None:
        m_colors = {
            "up": "#26a69a", "down": "#ef5350", "buy": "#26a69a", "sell": "#ef5350"
        }
        
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
        return "利多", "#26a69a"
    elif score < 0:
        return "利空", "#ef5350"
    else:
        return "中性", "#8B949E"

def get_expert_insight(ticker, price, rsi, rating, macd_val, signal_val, buy_sigs, sell_sigs, current_date, supertrend_dir=None, m_colors=None):
    """
    生成專家診斷報告 (Generates Expert Technical Diagnosis Report) - 波段交易優化版
    基於 RSI, MACD, SuperTrend 與 AI 評級提供綜合建議
    """
    # RSI 分析 (RSI Analysis)
    rsi_status = "強勢區" if rsi > 60 else "超賣" if rsi < 30 else "中性"
    rsi_color = "#26a69a" if rsi > 60 else "#26a69a" if rsi < 30 else "#8B949E"
    rsi_desc = "RSI 進入 60 以上強勢區，波段動能正在釋放。" if rsi > 60 else "股價進入超賣區，可能存在波段築底機會。" if rsi < 30 else "RSI 處於中性區間，適合波段佈局。"
    
    # MACD 分析 (MACD Analysis)
    macd_diff = macd_val - signal_val
    macd_status = "趨勢確認" if macd_diff > 0 else "趨勢轉弱"
    macd_color = "#26a69a" if macd_diff > 0 else "#ef5350"
    macd_desc = "MACD 柱狀體翻紅，波段多頭趨勢獲得確認。" if macd_diff > 0 else "MACD 動能放緩，波段可能進入整理期。"
    
    # --- NEW: SuperTrend Analysis ---
    st_status = "波段看多" if supertrend_dir == 1 else "波段看空" if supertrend_dir == -1 else "中性"
    st_color = "#26a69a" if supertrend_dir == 1 else "#ef5350" if supertrend_dir == -1 else "#8B949E"
    st_desc = "SuperTrend 通道向上，波段操作者建議持股續抱。" if supertrend_dir == 1 else "SuperTrend 轉向，波段趨勢有反轉風險。" if supertrend_dir == -1 else "趨勢尚不明確。"

    # 策略建議 (Action Advice Based on Rating)
    action = "波段進場" if "波段強勢" in rating else "趨勢偏多" if "趨勢偏多" in rating else "減碼觀望" if "波段轉弱" in rating else "中性佈局"
    action_color = "#26a69a" if "趨勢" in action or "波段" in action else "#ef5350" if "轉弱" in action else "#58A6FF"
    
    # 即時信號檢查 (Real-time Signal Check)
    latest_signal = "目前多指標共振，波段結構穩定。"
    if supertrend_dir == 1:
        latest_signal = "🔥 波段共振：趨勢通道看多，波段發動中！"
    elif supertrend_dir == -1:
        latest_signal = "⚠️ 波段轉弱：趨勢跌破支撐，建議執行移動停損。"

    # 檢查今日是否觸發買賣信號 (Check if today triggers signals)
    if current_date in buy_sigs:
        latest_signal = "🚀 今日觸發【波段買點】，技術面全面轉強！"
        action = "波段進場"
        action_color = "#26a69a"
    elif current_date in sell_sigs:
        latest_signal = "🔻 今日觸發【波段賣點】，建議獲利了結或停損。"
        action = "獲利減碼"
        action_color = "#ef5350"
    
    # 波段具體建議 (Swing Specific Advice)
    swing_advice = "建議觀察 EMA50 支撐，若不跌破則波段持有。"
    if supertrend_dir == 1:
        swing_advice = "多頭波段中，建議以 SuperTrend 線作為移動停利點。"
    elif rsi > 70:
        swing_advice = "短線過熱，波段持有者可部分獲利了結，等回測再加碼。"
    elif rsi < 30:
        swing_advice = "進入超賣區，波段可開始分批佈局底倉。"

    return {
        "rsi": {"val": f"{rsi:.1f}", "status": rsi_status, "color": rsi_color, "desc": rsi_desc},
        "macd": {"val": f"{macd_diff:+.2f}", "status": macd_status, "color": macd_color, "desc": macd_desc},
        "supertrend": {"status": st_status, "color": st_color, "desc": st_desc},
        "action": {"status": action, "color": action_color, "desc": latest_signal},
        "swing_advice": swing_advice
    }

def create_tv_gauge(score_val, label, color, m_colors=None):
    """Creates a TradingView-style gauge chart."""
    # 預設顏色方案 (Default colors if m_colors not provided)
    if m_colors is None:
        m_colors = {
            "up": "#26a69a", "down": "#ef5350", "buy": "#26a69a", "sell": "#ef5350"
        }
    
    # Helper to convert hex to rgba for Plotly gauge compatibility
    def get_rgba(hex_color, alpha):
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({r}, {g}, {b}, {alpha})'
    
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
                {'range': [0, 20], 'color': m_colors["down"]},
                {'range': [20, 40], 'color': get_rgba(m_colors['down'], 0.6)},
                {'range': [40, 60], 'color': '#787b86'},
                {'range': [60, 80], 'color': get_rgba(m_colors['up'], 0.6)},
                {'range': [80, 100], 'color': m_colors["up"]}
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
    根據股票元數據計算 1-10 分的五大因子得分 (Calculates 1-10 scores for 5 major factors)
    """
    # 1. 獲利能力得分 (Profitability - Margins & ROE)
    pm = ticker_metadata.get('profitMargins', 0) or 0
    roe = ticker_metadata.get('returnOnEquity', 0) or 0
    p_score = min(10, max(1, int(pm * 25 + roe * 20))) if pm or roe else 5
    
    # 2. 安全性/槓桿得分 (Safety/Leverage - Debt to Equity)
    de = ticker_metadata.get('debtToEquity', 100) or 100
    current_ratio = ticker_metadata.get('currentRatio', 1) or 1
    l_score = min(10, max(1, 10 - int(de / 40) + int(current_ratio * 2))) if de else 8
    
    # 3. 現金流得分 (Cash Flow - FCF/Revenue)
    fcf = ticker_metadata.get('freeCashflow', 0) or 0
    rev = ticker_metadata.get('totalRevenue', 1) or 1
    c_score = min(10, max(1, int((fcf / rev) * 40 + 4))) if fcf and rev else 6
    
    # 4. 成長因子 (Growth - Revenue & Earnings Growth)
    rev_growth = ticker_metadata.get('revenueGrowth', 0) or 0
    earn_growth = ticker_metadata.get('earningsGrowth', 0) or 0
    g_score = min(10, max(1, int(rev_growth * 20 + earn_growth * 15 + 5))) if rev_growth or earn_growth else 5
    
    # 5. 品質因子 (Quality - Operating Margins & ROIC)
    op_margin = ticker_metadata.get('operatingMargins', 0) or 0
    roic = ticker_metadata.get('returnOnAssets', 0) * 2 or 0 # 簡化 ROIC
    q_score = min(10, max(1, int(op_margin * 30 + roic * 20 + 3))) if op_margin or roic else 6
    
    return {
        "profitability": p_score,
        "safety": l_score,
        "cashflow": c_score,
        "growth": g_score,
        "quality": q_score
    }

def optimize_ai_weights(data, rsi_series, ema20, ema50, bb_lower, bb_upper, vr_series, atr_series, health_scores):
    """
    AI 進化引擎 v2.0 (AI Self-Evolution Engine):
    - 採用網格搜尋 (Grid Search) 優化技術、基本面、量能權重
    - 多維度評估：整合準確率 (Accuracy) 與 獲益率 (Profit Factor)
    - 適應性學習：自動根據個股歷史表現調整預測參數
    """
    # 擴展權重組合 (Tech, Fund, Vol)
    candidates = []
    for w_t in np.arange(0.2, 0.7, 0.1):
        for w_f in np.arange(0.1, 0.6, 0.1):
            w_v = round(1.0 - w_t - w_f, 2)
            if w_v >= 0.1:
                candidates.append((round(w_t, 2), round(w_f, 2), w_v))
    
    best_weights = (0.4, 0.3, 0.3)
    max_score = -1.0
    
    # 學習週期增加至 120 天，以獲取更穩定的特徵
    learning_period = 120
    if len(data) < learning_period:
        learning_period = len(data)
        
    test_data = data.iloc[-learning_period:]
    
    for w_tech, w_fund, w_vol in candidates:
        gains = []
        correct_preds = 0
        total_signals = 0
        
        # 使用滑動窗口進行回測
        for i in range(len(test_data) - 10):
            # 模擬進場訊號判定 (結合多個技術指標)
            curr_rsi = rsi_series.iloc[-(learning_period-i)] if not rsi_series.empty else 50
            curr_close = test_data['Close'].iloc[i]
            curr_bb_l = bb_lower.iloc[-(learning_period-i)] if not bb_lower.empty else curr_close * 0.95
            curr_ema20 = ema20.iloc[-(learning_period-i)] if not ema20.empty else curr_close
            
            # 觸發條件：RSI 超賣 或 觸及布林下軌 或 股價回測 20MA
            if curr_rsi < 45 or curr_close < curr_bb_l * 1.01 or curr_close < curr_ema20 * 1.01:
                total_signals += 1
                
                # 檢查未來 5 天表現
                future_prices = test_data['Close'].iloc[i+1:i+6]
                if not future_prices.empty:
                    max_future = future_prices.max()
                    profit = (max_future - curr_close) / curr_close
                    gains.append(profit)
                    
                    if profit > 0.025: # 2.5% 獲利目標
                        correct_preds += 1
        
        if total_signals > 0:
            accuracy = correct_preds / total_signals
            avg_gain = np.mean(gains) if gains else 0
            # 綜合評分：準確率 (60%) + 平均收益 (40%)
            score = (accuracy * 0.6) + (avg_gain * 40) 
            
            if score > max_score:
                max_score = score
                best_weights = (w_tech, w_fund, w_vol)
            
    return best_weights, max_score

def get_ai_entry_strategy(data, rsi, ema20, ema50, bb_lower, win_prob, health_scores, vr, atr, supertrend_dir=None, dynamic_weights=None, m_colors=None):
    """
    Generates AI-driven entry strategy (Swing Trading Optimized v3.5)
    """
    if m_colors is None:
        m_colors = {"up": "#26a69a", "down": "#ef5350", "buy": "#26a69a", "sell": "#ef5350"}
    
    current_price = float(data['Close'].iloc[-1])
    
    # 因子提取 (Factor Extraction)
    if isinstance(health_scores, dict):
        p_score, l_score, c_score = health_scores.get("profitability", 5), health_scores.get("safety", 5), health_scores.get("cashflow", 5)
        g_score, q_score = health_scores.get("growth", 5), health_scores.get("quality", 5)
    else:
        p_score, l_score, c_score = health_scores
        g_score, q_score = 5, 5
    
    # 權重分配：技術面(40%)、基本面(35%)、動能/量能(25%)
    w_tech, w_fund, w_vol = dynamic_weights if dynamic_weights else (0.40, 0.35, 0.25)
    
    # 1. Technical Score (0-100)
    t_raw = 0
    ema200 = float(data['Close'].ewm(span=200).mean().iloc[-1])
    if current_price > ema50 > ema200: t_raw += 40 
    if 40 < rsi < 65: t_raw += 25
    if supertrend_dir == 1: t_raw += 35
    
    # 2. Fundamental Score (0-100)
    f_raw = (p_score + l_score + c_score + g_score + q_score) * 2 # 轉為百分制
    
    # 3. Volume/Momentum Score (0-100)
    v_raw = 0
    if vr > 130: v_raw += 30
    elif vr > 100: v_raw += 15
    
    vol_ma20 = data['Volume'].rolling(20).mean().iloc[-1]
    if data['Volume'].iloc[-1] > vol_ma20 * 1.5: v_raw += 25
    
    price_change_3d = (current_price - data['Close'].iloc[-4]) / data['Close'].iloc[-4]
    if price_change_3d > 0.03: v_raw += 25 
    if win_prob > 55: v_raw += 20
    
    # 綜合評分
    total_score = (min(100, t_raw) * w_tech) + (min(100, f_raw) * w_fund) + (min(100, v_raw) * w_vol)
    
    # 策略生成
    if total_score > 65:
        action, color = "積極買入", "#26a69a"
        desc = "多頭趨勢確立且量價配合完美，具備極高爆發潛力。"
        suggested_price = current_price
        target_price = current_price + (3.5 * atr)
    elif total_score > 45:
        action, color = "建議試探", "#26a69a"
        desc = "趨勢偏多但波動較大，建議分批佈局，守穩支撐。"
        suggested_price = ema20
        target_price = current_price + (2.5 * atr)
    elif total_score > 25:
        action, color = "少量參與", "#D29922"
        desc = "目前信號中性，僅適合以極小部位參與短線反彈。"
        suggested_price = ema20 * 0.98
        target_price = current_price + (1.5 * atr)
    else:
        action, color = "保守觀望", "#8B949E"
        desc = "信號疲弱且趨勢不明，建議空手等待更佳入場時機。"
        suggested_price = ema50 * 0.95
        target_price = current_price * 1.05

    return {
        "price": suggested_price,
        "target_price": target_price,
        "stop_loss": current_price - (2.5 * atr),
        "confidence": f"{total_score:.1f}%",
        "action": action,
        "desc": desc,
        "color": color,
        "score": total_score
    }

def check_signal_performance(signal_date, suggested_price, signal_type, full_data, m_colors=None):
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
                return "精準達標", "#26a69a", profit_pct
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
                return "避險成功", "#26a69a", save_pct
            elif reached_exit:
                return "已出場", "#D29922", save_pct
            else:
                return "未觸及", "#8B949E", 0
    except:
        return "分析中", "#8B949E", 0

def get_institutional_strategy(data, i, health_scores=None, ema_data=None, vol_ma50=None):
    """
    識別知名基金與大師策略 (Identifies famous fund & master strategies)
    """
    strategies = []
    close_prices = data['Close']
    volumes = data['Volume']
    
    # 1. 海龜交易 (Turtle Trading) - 20日突破
    high_20 = close_prices.iloc[max(0, i-20):i].max()
    if close_prices.iloc[i] > high_20:
        strategies.append({"name": "海龜突破", "desc": "Turtle Trading: 20日價格高點突破，趨勢啟動。"})
        
    # 2. 米奈爾維尼 (Minervini Trend Template)
    # 結合基本面 (Quality/Growth) 與技術面 (均線)
    if ema_data is not None:
        e50, e150, e200 = ema_data
        is_trend_template = (close_prices.iloc[i] > e50.iloc[i] > e150.iloc[i] > e200.iloc[i] and 
                            e200.iloc[i] > e200.iloc[max(0, i-20)])
        
        fund_ok = True
        if health_scores and isinstance(health_scores, dict):
            # Minervini 偏好高品質成長股
            if health_scores.get("quality", 5) < 6 or health_scores.get("growth", 5) < 6:
                fund_ok = False
        
        if is_trend_template and fund_ok:
            strategies.append({"name": "米奈爾維尼", "desc": "Minervini: 均線多頭排列且財務品質優秀，進入第二階段增長。"})
        
    # 3. 歐尼爾 (O'Neil CAN SLIM)
    # C: Current Earnings, A: Annual Earnings (由 health_scores 代表)
    if vol_ma50 is not None:
        vol_spike = volumes.iloc[i] > vol_ma50.iloc[i] * 1.5
        price_up = close_prices.iloc[i] > close_prices.iloc[i-1]
        
        fund_ok = True
        if health_scores and isinstance(health_scores, dict):
            # CAN SLIM 要求強勁的成長 (Growth)
            if health_scores.get("growth", 5) < 7:
                fund_ok = False
        
        if price_up and vol_spike and fund_ok:
            strategies.append({"name": "CAN SLIM", "desc": "O'Neil: 價格帶量突破且具備高成長動能，機構資金介入。"})
        
    return strategies

def run_advanced_mc_simulation(current_price, data, ai_score, atr, ticker_metadata=None, supertrend_dir=None, days=30, simulations=1000):
    """
    重構後的蒙地卡羅模擬引擎 v4.0 (Advanced MC Engine):
    - 採用動態波動率 (Dynamic Volatility) 與 EWMA 模型
    - 整合趨勢加速度 (Trend Acceleration) 與 均值回歸 (EMA200)
    - 使用 Student's t-分佈模擬肥尾風險，並根據近期偏度調整
    """
    returns = data['Close'].pct_change().dropna()
    
    # 1. 動態波動率計算 (EWMA Volatility)
    # 給予近期波動更高權重，以更精確反應當前市場情緒
    recent_returns = returns.tail(20)
    ewma_vol = recent_returns.ewm(span=10).std().iloc[-1]
    hist_vol = returns.std()
    
    # 2. 計算多因子漂移率 (Drift)
    # AI 信心偏誤
    ai_bias = (ai_score - 50) / 3500 
    
    # 趨勢強度與加速度 (Trend Strength & Acceleration)
    ema20_series = data['Close'].ewm(span=20).mean()
    ema20_slope = (ema20_series.iloc[-1] - ema20_series.iloc[-5]) / ema20_series.iloc[-5]
    trend_bias = ema20_slope * 0.15 # 將均線斜率轉換為漂移偏誤
    
    # SuperTrend 趨勢確認
    indicator_bias = 0.0012 * supertrend_dir if supertrend_dir is not None else 0
    
    # 均值回歸 (Mean Reversion) - 修正項
    ema200 = data['Close'].ewm(span=200).mean().iloc[-1]
    reversion_strength = 0.08
    dist_from_mean = (ema200 - current_price) / current_price
    reversion_bias = dist_from_mean * reversion_strength / days
    
    # 法人目標價影響
    target_bias = 0
    if ticker_metadata and ticker_metadata.get('targetMedianPrice'):
        target_price = ticker_metadata.get('targetMedianPrice')
        if target_price > 0:
            target_dist = (target_price - current_price) / current_price
            target_bias = target_dist * 0.15 / days

    # 綜合預期回報率 (Expected Return)
    mu = returns.mean() + ai_bias + trend_bias + indicator_bias + reversion_bias + target_bias
    
    # 綜合波動率 (Adjusted Volatility)
    # 考慮 ATR 與 EWMA 波動率的極大值，並針對弱勢行情增加波動溢價
    atr_vol = (atr / current_price) / np.sqrt(14) # 將 ATR 轉換為每日波動估計
    base_vol = max(ewma_vol, atr_vol, hist_vol * 0.8)
    
    # 若處於空頭市場 (價格 < EMA200)，增加 15% 波動率以反映風險
    if current_price < ema200:
        base_vol *= 1.15
        
    # 3. 模擬執行 (Student's t-分佈)
    df_student = 3.5 # 較低的自由度以模擬更頻繁的極端行情 (肥尾)
    sim_results = np.zeros((days + 1, simulations))
    sim_results[0] = current_price
    
    # 預先生成隨機衝擊
    random_shocks = student_t.rvs(df=df_student, loc=mu, scale=base_vol, size=(days, simulations))
    
    for t in range(1, days + 1):
        # 價格演化：幾何布朗運動變體
        sim_results[t] = sim_results[t-1] * (1 + random_shocks[t-1])
        # 波動率動態調整 (Volatility Clustering)
        # 隨機調整波動率，模擬波動聚集效應
        base_vol *= np.random.normal(1.0, 0.01)
        
    return pd.DataFrame(sim_results)

def get_ai_exit_strategy(data, rsi, bb_upper, target_median, atr, health_scores=None, supertrend_dir=None, m_colors=None):
    """
    Generates AI-driven exit strategy (Dynamic Profit Taking v3.5)
    """
    if m_colors is None:
        m_colors = {"up": "#26a69a", "down": "#ef5350", "buy": "#26a69a", "sell": "#ef5350"}
    
    current_price = float(data['Close'].iloc[-1])
    base_target = target_median if target_median and target_median > current_price else bb_upper
    
    # 1. 退出壓力評分 (Exit Pressure Score)
    exit_score = 0
    if rsi > 80: exit_score += 45
    elif rsi > 70: exit_score += 25
    
    if current_price > bb_upper: exit_score += 25
    if supertrend_dir == -1: exit_score += 20
    
    # 均線背離或破位
    ema20 = data['Close'].ewm(span=20).mean().iloc[-1]
    if current_price < ema20: exit_score += 20
    
    # 2. 基本面調節 (減壓因子)
    f_bonus = 0
    if health_scores and isinstance(health_scores, dict):
        g_score, q_score = health_scores.get("growth", 5), health_scores.get("quality", 5)
        if g_score >= 8 and q_score >= 8: f_bonus = -20
        elif g_score >= 7 or q_score >= 7: f_bonus = -10
    
    exit_score = max(0, min(100, exit_score + f_bonus))
    
    # 策略生成
    if exit_score > 65:
        action, color = "立即獲利", m_colors["sell"]
        desc = "多項指標顯示嚴重超買或趨勢反轉，建議立即結清部位。"
        suggested_price = current_price
    elif exit_score > 45:
        action, color = "分批獲利", m_colors["sell"]
        desc = "股價進入超漲區間，建議於壓力位附近分批逢高減碼。"
        suggested_price = base_target
    elif exit_score > 25:
        action, color = "短線調節", "#D29922"
        desc = "上升動能減弱，建議適度縮減部位，守住獲利。"
        suggested_price = max(base_target, current_price * 1.02)
    else:
        action, color = "續抱觀察", "#58A6FF"
        desc = "趨勢依然穩健，建議守住移動停利位，讓利潤奔跑。"
        suggested_price = max(base_target, current_price * 1.05)
        
    return {
        "price": suggested_price,
        "trailing_stop": current_price - (1.5 * atr),
        "confidence": f"{exit_score:.1f}%",
        "action": action,
        "desc": desc,
        "color": color,
        "score": exit_score
    }

def calculate_quant_factors(data, ticker_metadata, rsi_series, atr_series, supertrend_dir=None, financial_data=None):
    """
    量化多因子評分系統 v3.0 (Quant Multi-Factor Scoring System)
    優化因子權重與計算邏輯，新增波動率調節與趨勢共振因子
    """
    factors = {}
    
    # 1. 趨勢因子 (Trend) - 權重: 25%
    close_prices = data['Close']
    ema20 = close_prices.ewm(span=20).mean()
    ema50 = close_prices.ewm(span=50).mean()
    ema200 = close_prices.ewm(span=200).mean()
    
    trend_score = 0
    # 多頭排列檢查 (Resonance)
    if close_prices.iloc[-1] > ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1]:
        trend_score = 100
    elif close_prices.iloc[-1] > ema20.iloc[-1] > ema50.iloc[-1]:
        trend_score = 85
    elif close_prices.iloc[-1] > ema50.iloc[-1]:
        trend_score = 65
    elif close_prices.iloc[-1] > ema200.iloc[-1]:
        trend_score = 50
    else:
        trend_score = 30
    
    # 結合 SuperTrend 方向調節
    if supertrend_dir is not None:
        if supertrend_dir == 1:
            trend_score = min(100, trend_score + 15)
        else:
            trend_score = max(0, trend_score - 15)
    
    factors['趨勢 (Trend)'] = trend_score
    
    # 2. 動能因子 (Momentum) - 權重: 15%
    # 結合 RSI 與 價格變化率 (ROC)
    rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50
    roc_5 = (close_prices.iloc[-1] / close_prices.iloc[-6] - 1) * 100 if len(close_prices) > 5 else 0
    
    # 強勢股動能評分：RSI 在 50-75 之間且 ROC > 0 最優
    mom_score = 50
    if 50 < rsi_val < 80:
        mom_score = 70 + (rsi_val - 50)
    elif rsi_val >= 80:
        mom_score = 90 # 過熱但動能極強
    elif rsi_val < 30:
        mom_score = 40 # 超賣反彈動能
    
    if roc_5 > 2: mom_score = min(100, mom_score + 10)
    
    factors['動能 (Momentum)'] = mom_score
    
    # 3. 波動因子 (Volatility) - 權重: 10% (低波動得分高)
    atr_val = float(atr_series.iloc[-1]) if not atr_series.empty else 0
    vol_ratio = (atr_val / close_prices.iloc[-1]) * 100
    # 波動率調節：平穩上漲的股票得分高
    vol_score = max(0, min(100, 100 - (vol_ratio * 12))) 
    factors['波動 (Volatility)'] = vol_score
    
    # 4. 量能因子 (Volume) - 權重: 15%
    v_ma5 = data['Volume'].rolling(5).mean().iloc[-1]
    v_ma20 = data['Volume'].rolling(20).mean().iloc[-1]
    v_ma50 = data['Volume'].rolling(50).mean().iloc[-1]
    
    # 量能共振：短期量大於中期量，且中期量大於長期量
    if v_ma5 > v_ma20 > v_ma50:
        volume_score = 95
    elif v_ma5 > v_ma20:
        volume_score = 80
    elif v_ma5 < v_ma20:
        volume_score = 45
    else:
        volume_score = 60
        
    factors['量能 (Volume)'] = volume_score
    
    # 5. 價值因子 (Value) - 權重: 10%
    pe = ticker_metadata.get('trailingPE') or ticker_metadata.get('forwardPE') or 20
    pb = ticker_metadata.get('priceToBook') or 2.0
    # 優化價值評分邏輯 (更符合台股特性)
    pe_score = max(0, min(100, 110 - pe * 2.5)) # PE 15 左右得分 70+
    pb_score = max(0, min(100, 110 - pb * 25))  # PB 1.5 左右得分 70+
    factors['價值 (Value)'] = (pe_score * 0.6 + pb_score * 0.4)
    
    # 6. 品質因子 (Quality) - 權重: 15%
    roe = ticker_metadata.get('returnOnEquity', 0) or 0
    profit_margin = ticker_metadata.get('profitMargins', 0) or 0
    debt_equity = ticker_metadata.get('debtToEquity', 100) or 100
    
    roe_score = min(100, roe * 500) # ROE 20% 為滿分
    margin_score = min(100, profit_margin * 400) # 利潤率 25% 為滿分
    debt_score = max(0, min(100, 120 - debt_equity / 1.5)) # 債務控制評分
    factors['品質 (Quality)'] = (roe_score * 0.4 + margin_score * 0.3 + debt_score * 0.3)
    
    # 7. 成長因子 (Growth) - 權重: 10%
    growth_score = 55
    if financial_data is not None and len(financial_data) >= 2:
        rev_growth = (financial_data['Total Revenue'].iloc[-1] / financial_data['Total Revenue'].iloc[-2] - 1) * 100
        ni_growth = (financial_data['Net Income'].iloc[-1] / financial_data['Net Income'].iloc[-2] - 1) * 100
        # 成長性綜合評分
        growth_score = min(100, max(0, 60 + rev_growth * 0.5 + ni_growth * 0.3))
    factors['成長 (Growth)'] = growth_score
    
    return factors

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

# Header Section
col_title, col_status = st.columns([3, 2])
with col_title:
    st.markdown(f'<h1 class="main-header">{t["title"]}</h1>', unsafe_allow_html=True)
    stock_header_placeholder = st.empty() # For stock name and ticker

if ticker_input:
    data, resolved_ticker = get_stock_data(ticker_input, period, interval)
    
    if data is not None and not data.empty:
        # 根據市場獲取顏色方案 (Get market-specific colors)
        m_colors = get_market_colors(resolved_ticker)
        
        # Update status badge with market colors
        with col_status:
            st.markdown(f'''<div style="text-align: right; padding-top: 10px;">
<div style="font-size: 0.75rem; color: #8B949E; margin-bottom: 6px; font-family: 'Roboto Mono', monospace; opacity: 0.8;">
SYNC_TIME: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
</div>
<div style="display: inline-flex; align-items: center; background: rgba(35, 134, 54, 0.1); color: #3FB950; border: 1px solid rgba(63, 185, 80, 0.2); padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.5px; transition: all 0.3s ease;">
<span style="width: 8px; height: 8px; background: #3FB950; border-radius: 50%; margin-right: 8px; box-shadow: 0 0 8px #3FB950;"></span>
MARKET_CONNECTED
</div>
</div>''', unsafe_allow_html=True)
            
        stock = yf.Ticker(resolved_ticker)
        try:
            ticker_metadata = stock.info
            if not ticker_metadata or not isinstance(ticker_metadata, dict):
                ticker_metadata = {"shortName": resolved_ticker}
        except:
            ticker_metadata = {"shortName": resolved_ticker}
        
        # --- 貨幣符號處理 (Currency Symbol Handling) ---
        currency_code = ticker_metadata.get('currency', 'USD')
        currency_symbols = {
            'TWD': 'NT$',
            'USD': '$',
            'HKD': 'HK$',
            'JPY': '¥',
            'EUR': '€',
            'GBP': '£'
        }
        c_symbol = currency_symbols.get(currency_code, '$')
        
        # --- 動態大盤關聯度分析 (Dynamic Benchmark Analysis) ---
        if resolved_ticker.endswith(".TW") or resolved_ticker.endswith(".TWO") or resolved_ticker.isdigit():
            market_benchmark = "^TWII"
        elif "-USD" in resolved_ticker or resolved_ticker in ["BTC-USD", "ETH-USD"]:
            market_benchmark = "BTC-USD"
        elif "=" in resolved_ticker: # Forex or Commodities
            market_benchmark = "DX-Y.NYB" # Dollar Index
        else:
            market_benchmark = "^GSPC" # S&P 500 for US/Global
            
        market_corr, market_beta, relative_strength = 0, 1.0, 0
        try:
            m_data = yf.download(market_benchmark, period=period, interval=interval, progress=False, auto_adjust=True)
            if not m_data.empty:
                if isinstance(m_data.columns, pd.MultiIndex):
                    m_data.columns = m_data.columns.get_level_values(0)
                
                # 對齊日期計算
                aligned_df = pd.concat([data['Close'], m_data['Close']], axis=1).dropna()
                if not aligned_df.empty and len(aligned_df) > 1:
                    aligned_df.columns = ['Stock', 'Market']
                    returns = aligned_df.pct_change().dropna()
                    
                    if not returns.empty:
                        market_corr = returns['Stock'].corr(returns['Market'])
                        market_cov = returns['Stock'].cov(returns['Market'])
                        market_var = returns['Market'].var()
                        market_beta = market_cov / market_var if market_var != 0 else 1.0
                
                # 相對漲幅 (近 20 天)
                if len(data) > 20 and len(m_data) > 20:
                    stock_30d = (data['Close'].iloc[-1] / data['Close'].iloc[-20] - 1) * 100
                    market_30d = (m_data['Close'].iloc[-1] / m_data['Close'].iloc[-20] - 1) * 100
                    relative_strength = stock_30d - market_30d
        except:
            pass

        health_scores = calculate_health_scores(ticker_metadata)
        p_score = health_scores["profitability"]
        l_score = health_scores["safety"]
        c_score = health_scores["cashflow"]
        
        # Update header placeholder with stock name and ticker
        ticker_display_name = ticker_metadata.get('shortName') or ticker_metadata.get('longName') or resolved_ticker
        stock_header_placeholder.markdown(f'<p style="color: #58A6FF; font-size: 1.2rem; font-weight: 600; margin-top: -5px;">{ticker_display_name} <span style="color: #8B949E; font-weight: 400; font-size: 0.9rem;">({resolved_ticker})</span></p>', unsafe_allow_html=True)

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
            try:
                high_low = df['High'] - df['Low']
                high_cp = abs(df['High'] - df['Close'].shift())
                low_cp = abs(df['Low'] - df['Close'].shift())
                tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
                atr = tr.rolling(window=n).mean()
                return atr
            except Exception as e:
                st.error(f"ATR 計算出錯: {e}")
                return pd.Series(0, index=df.index)

        atr_series = calculate_atr(data)

        # --- NEW: SuperTrend Indicator ---
        def calculate_supertrend(df, atr_s, multiplier=3):
            try:
                hl2 = (df['High'] + df['Low']) / 2
                upperband = hl2 + (multiplier * atr_s)
                lowerband = hl2 - (multiplier * atr_s)
                
                supertrend = pd.Series(index=df.index, dtype=float)
                direction = pd.Series(index=df.index, dtype=int) # 1 for up, -1 for down
                
                # Initialize
                supertrend.iloc[0] = upperband.iloc[0]
                direction.iloc[0] = -1
                
                for i in range(1, len(df)):
                    if direction.iloc[i-1] == 1: # Previous trend was up
                        if df['Close'].iloc[i] < lowerband.iloc[i-1]:
                            direction.iloc[i] = -1
                            supertrend.iloc[i] = upperband.iloc[i]
                        else:
                            direction.iloc[i] = 1
                            supertrend.iloc[i] = max(lowerband.iloc[i], lowerband.iloc[i-1])
                    else: # Previous trend was down
                        if df['Close'].iloc[i] > upperband.iloc[i-1]:
                            direction.iloc[i] = 1
                            supertrend.iloc[i] = lowerband.iloc[i]
                        else:
                            direction.iloc[i] = -1
                            supertrend.iloc[i] = min(upperband.iloc[i], upperband.iloc[i-1])
                return supertrend, direction
            except Exception as e:
                st.error(f"SuperTrend 計算出錯: {e}")
                return pd.Series(0, index=df.index), pd.Series(0, index=df.index)

        supertrend_line, supertrend_dir = calculate_supertrend(data, atr_series)
        
        # --- AI Self-Evolution (Calculated early for signal filtering) ---
        if len(data) > 30:
            with st.spinner(f"AI 正在針對 {resolved_ticker} 進行自我進化優化..."):
                try:
                    best_weights, learn_acc = optimize_ai_weights(
                        data, rsi_series, ema20, ema50, bb_lower, bb_upper, vr_series, atr_series, health_scores
                    )
                except Exception as e:
                    st.warning(f"AI 優化過程出現小問題: {e}，使用預設權重。")
                    best_weights, learn_acc = (0.45, 0.35, 0.20), 0.5
        else:
            best_weights, learn_acc = (0.45, 0.35, 0.20), 0.5
            
        # Initial score for drift calculation and signal filtering
        initial_ai_strat = get_ai_entry_strategy(
            data, 
            float(rsi_series.iloc[-1]) if not rsi_series.empty else 50, 
            float(ema20.iloc[-1]) if not ema20.empty else current_price, 
            float(ema50.iloc[-1]) if not ema50.empty else current_price, 
            float(bb_lower.iloc[-1]) if not bb_lower.empty else current_price * 0.95, 
            50, health_scores, 
            float(vr_series.iloc[-1]) if not vr_series.empty else 100, 
            float(atr_series.iloc[-1]) if not atr_series.empty else 0, 
            supertrend_dir=supertrend_dir.iloc[-1] if not supertrend_dir.empty else 0,
            dynamic_weights=best_weights
        )
        ai_score = initial_ai_strat['score']

        # Calculate Buy/Sell Signals with AI + Multi-Factor Confirmation
        all_signals = []
        vol_ma5 = data['Volume'].rolling(window=5).mean()
        vol_ma50 = data['Volume'].rolling(window=50).mean()
        
        # 預計算機構策略所需的指標
        e50 = data['Close'].ewm(span=50).mean()
        e150 = data['Close'].ewm(span=150).mean()
        e200 = data['Close'].ewm(span=200).mean()
        ema_data = (e50, e150, e200)
        
        target_median = ticker_metadata.get('targetMedianPrice', None)
        
        # 遍歷數據生成信號 (Iterate to generate signals)
        for i in range(20, len(data)): # 從 20 開始確保有足夠의 MA 數據 (Ensure enough data for MAs)
            d = data.index[i]
            
            # 機構策略識別 (Institutional Strategy Identification)
            inst_strats = get_institutional_strategy(data, i, health_scores=health_scores, ema_data=ema_data, vol_ma50=vol_ma50)
            
            # --- 基礎技術指標過濾 (Basic Technical Filters - Loosened for more entries) ---
            macd_gold = macd_series.iloc[i] > signal_series.iloc[i] and macd_series.iloc[i-1] <= signal_series.iloc[i-1]
            rsi_buy = rsi_series.iloc[i] > 30 and rsi_series.iloc[i-1] <= 30
            kd_gold = k_series.iloc[i] > d_series.iloc[i] and k_series.iloc[i-1] <= d_series.iloc[i-1]
            bb_touch_low = data['Low'].iloc[i] <= bb_lower.iloc[i]
            
            # 買入條件放寬：滿足任一技術指標金叉，且價格高於 20MA 或 正在反彈 (KD 金叉)
            trend_ok_buy = data['Close'].iloc[i] > ema20.iloc[i] or kd_gold
            vol_ok = data['Volume'].iloc[i] > vol_ma5.iloc[i] # 降低成交量門檻，從 1.1x 降至 1.0x
            
            macd_death = macd_series.iloc[i] < signal_series.iloc[i] and macd_series.iloc[i-1] >= signal_series.iloc[i-1]
            rsi_sell = rsi_series.iloc[i] < 70 and rsi_series.iloc[i-1] >= 70
            kd_death = k_series.iloc[i] < d_series.iloc[i] and k_series.iloc[i-1] >= d_series.iloc[i-1]
            bb_touch_high = data['High'].iloc[i] >= bb_upper.iloc[i]
            
            # 賣出條件放寬
            trend_ok_sell = data['Close'].iloc[i] < ema20.iloc[i] or kd_death or bb_touch_high

            # --- 買入信號處理 (Buy Signal Processing) ---
            if (macd_gold or rsi_buy or kd_gold or bb_touch_low) and trend_ok_buy and vol_ok:
                try:
                    h_rsi = float(rsi_series.iloc[i])
                    h_ema20 = float(ema20.iloc[i])
                    h_ema50 = float(ema50.iloc[i])
                    h_bb_lower = float(bb_lower.iloc[i])
                    h_vr = float(vr_series.iloc[i])
                    h_atr = float(atr_series.iloc[i])
                    
                    # 計算該時間點的 AI 買入評分
                    h_ai = get_ai_entry_strategy(
                        data.iloc[:i+1], h_rsi, h_ema20, h_ema50, h_bb_lower, 55, health_scores, h_vr, h_atr, 
                        supertrend_dir=supertrend_dir.iloc[i],
                        dynamic_weights=best_weights, 
                        m_colors=m_colors
                    )
                    h_score = h_ai.get('score', 0)
                    
                    # 績效追蹤
                    perf_status, perf_color, perf_pct = check_signal_performance(d, h_ai['price'], "買入", data, m_colors=m_colors)
                    
                    # 只有當 AI 評分高於一定閾值，或者動作不是「觀望」時才加入信號流水線
                    # 降低顯示閾值以增加信號頻率 (從 45 降至 35)
                    if h_score >= 35 or h_ai['action'] != "保守觀望" or len(inst_strats) > 0:
                        price_val = float(data['Close'].iloc[i])
                        # 波段優選邏輯：AI 高分 + 大師策略共振
                        is_swing_prime = h_score >= 60 and len(inst_strats) >= 1
                        
                        all_signals.append({
                            "date": d, "type": "買入", "price": price_val, "color": m_colors["buy"], "icon": "🔼",
                            "ai_price": h_ai['price'], "ai_action": h_ai['action'], "stop_loss": h_ai['stop_loss'],
                            "perf_status": perf_status, "perf_color": perf_color, "perf_pct": perf_pct,
                            "ai_verified": h_score >= 60, "ai_score": h_score,
                            "inst_strategies": inst_strats,
                            "is_swing_prime": is_swing_prime
                        })
                except: continue

            # --- 賣出信號處理 (Sell Signal Processing) ---
            if (macd_death or rsi_sell or kd_death or bb_touch_high) and trend_ok_sell:
                try:
                    h_rsi = float(rsi_series.iloc[i])
                    h_bb_upper = float(bb_upper.iloc[i])
                    h_atr = float(atr_series.iloc[i])
                    
                    # 計算該時間點的 AI 賣出評分
                    h_ai = get_ai_exit_strategy(data.iloc[:i+1], h_rsi, h_bb_upper, target_median, h_atr, health_scores=health_scores, m_colors=m_colors)
                    h_score = h_ai.get('score', 0)
                    
                    # 績效追蹤
                    perf_status, perf_color, perf_pct = check_signal_performance(d, h_ai['price'], "賣出", data, m_colors=m_colors)
                    
                    # 只有當 AI 評分高於一定閾值，或者動作不是「續抱」時才加入信號流水線
                    # 降低顯示閾值 (從 40 降至 30)
                    if h_score >= 30 or h_ai['action'] != "續抱觀察":
                        price_val = float(data['Close'].iloc[i])
                        all_signals.append({
                            "date": d, "type": "賣出", "price": price_val, "color": m_colors["sell"], "icon": "🔽",
                            "ai_price": h_ai['price'], "ai_action": h_ai['action'], "trailing_stop": h_ai.get('trailing_stop', 0),
                            "perf_status": perf_status, "perf_color": perf_color, "perf_pct": perf_pct,
                            "ai_verified": h_score >= 60, "ai_score": h_score,
                            "inst_strategies": inst_strats
                        })
                except: continue
        
        # 過濾出最近 30 天的信號用於 UI 顯示
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
        sig_text, sig_color, sig_val = get_signal_score(
            data, float(rsi_series.iloc[-1]), float(macd_series.iloc[-1]), float(signal_series.iloc[-1]), 
            supertrend_dir=supertrend_dir.iloc[-1], 
            m_colors=m_colors
        )
        
        st.markdown("---")
        
        # --- AI 模擬與決策計算 (AI Simulation & Decision Calculations) ---
        with st.spinner(f"正在生成 {ticker_input} 未來模擬路徑與決策分析..."):
            atr = float(atr_series.iloc[-1])
            
            sim_df_full = run_advanced_mc_simulation(
                current_price, data, ai_score, atr, ticker_metadata=ticker_metadata, 
                supertrend_dir=supertrend_dir.iloc[-1],
                days=60, simulations=1000
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
                float(bb_lower.iloc[-1]), win_prob_7d, health_scores, 
                float(vr_series.iloc[-1]), float(atr_series.iloc[-1]), dynamic_weights=best_weights,
                m_colors=m_colors
            )
            
            target_median = ticker_metadata.get('targetMedianPrice', None)
            exit_strategy = get_ai_exit_strategy(
                data, float(rsi_series.iloc[-1]), float(bb_upper.iloc[-1]), 
                target_median, float(atr_series.iloc[-1]), health_scores=health_scores, m_colors=m_colors
            )
            
        tab1, tab2, tab3, tab4 = st.tabs([t["tab_overview"], t["tab_tech"], t["tab_predict"], t["tab_news"]])

        with tab1:
            # 1. 快速建議與預測摘要 (Quick Recommendation & Prediction Summary)
            insight_report = get_expert_insight(
                resolved_ticker, 
                current_price, 
                float(rsi_series.iloc[-1]), 
                sig_text, 
                float(macd_series.iloc[-1]), 
                float(signal_series.iloc[-1]), 
                buy_signals, 
                sell_signals, 
                data.index[-1],
                supertrend_dir=supertrend_dir.iloc[-1],
                m_colors=m_colors
            )
            
            # 顯示 AI 進化狀態與市場關聯 (Display Evolution Status & Market Correlation)
            st.markdown(f'''<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin-bottom: 24px;">
<!-- AI Engine Status Card -->
<div class="data-card" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0; border-left: 4px solid #58A6FF; background: linear-gradient(145deg, rgba(88, 166, 253, 0.05) 0%, #161B22 100%);">
    <div style="display: flex; align-items: center; gap: 14px;">
        <div style="background: rgba(56, 139, 253, 0.1); width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.4rem; border: 1px solid rgba(56, 139, 253, 0.2);">🧬</div>
        <div>
            <div style="color: #58A6FF; font-size: 0.9rem; font-weight: 800; letter-spacing: 0.8px; text-transform: uppercase;">AI 引擎狀態</div>
            <div style="color: #8B949E; font-size: 0.75rem; margin-top: 2px;">模式：自我優化 (SELF_OPTIMIZED)</div>
        </div>
    </div>
    <div style="text-align: right;">
        <div style="color: #3FB950; font-size: 1.1rem; font-weight: 900; letter-spacing: -0.5px;">{learn_acc*100:.1f}%</div>
        <div style="color: #8B949E; font-size: 0.65rem; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">核心準確率</div>
    </div>
</div>

<!-- Market Correlation Card -->
<div class="data-card" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0; border-left: 4px solid #BC8CF2; background: linear-gradient(145deg, rgba(188, 140, 242, 0.05) 0%, #161B22 100%);">
    <div style="display: flex; align-items: center; gap: 14px;">
        <div style="background: rgba(188, 140, 242, 0.1); width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.4rem; border: 1px solid rgba(188, 140, 242, 0.2);">📊</div>
        <div>
            <div style="color: #BC8CF2; font-size: 0.9rem; font-weight: 800; letter-spacing: 0.8px; text-transform: uppercase;">大盤關聯分析</div>
            <div style="color: #8B949E; font-size: 0.75rem; margin-top: 2px;">相關性: {market_corr:.2f} | 貝塔: {market_beta:.2f}</div>
        </div>
    </div>
    <div style="text-align: right;">
        <div style="color: {"#FF7B72" if relative_strength < 0 else "#3FB950"}; font-size: 1.1rem; font-weight: 900; letter-spacing: -0.5px;">
            {relative_strength:+.1f}%
        </div>
        <div style="color: #8B949E; font-size: 0.65rem; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">相對超額收益</div>
    </div>
</div>
</div>''', unsafe_allow_html=True)

            # 推薦建議區塊 - 採用 Grid 佈局提升響應性 (Recommendation Grid)
            st.markdown(f'''<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 24px;">
<!-- 當前操作建議 -->
<div class="data-card" style="border-left: 5px solid {insight_report['action']['color']}; height: 160px; margin-bottom: 0; background: linear-gradient(180deg, {insight_report['action']['color']}0a 0%, #161B22 100%);">
    <div style="color: #8B949E; font-size: 0.7rem; margin-bottom: 5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">當前操作建議</div>
    <div style="font-size: 1.3rem; font-weight: 800; color: {insight_report['action']['color']}; margin-bottom: 4px;">{insight_report['action']['status']}</div>
    <div style="color: #C9D1D9; font-size: 0.75rem; line-height: 1.4;">{insight_report['action']['desc']}</div>
</div>

<!-- AI 建議買點 -->
<div class="data-card" style="border-left: 5px solid {entry_strategy['color']}; height: 160px; margin-bottom: 0; background: linear-gradient(180deg, {entry_strategy['color']}0a 0%, #161B22 100%);">
    <div style="color: #8B949E; font-size: 0.7rem; margin-bottom: 5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">AI 建議買點 ({entry_strategy['confidence']})</div>
    <div style="display: flex; align-items: baseline; gap: 6px; margin-bottom: 4px;">
        <span style="font-size: 1.3rem; font-weight: 800; color: {entry_strategy['color']};">{c_symbol}{entry_strategy['price']:.1f}</span>
        <span style="color: #8B949E; font-size: 0.7rem;">{entry_strategy['action']}</span>
    </div>
    <div style="color: #FF7B72; font-size: 0.75rem; font-weight: 700; margin-bottom: 4px;">建議停損: {c_symbol}{entry_strategy['stop_loss']:.1f}</div>
    <div style="color: #C9D1D9; font-size: 0.7rem; line-height: 1.3;">{entry_strategy['desc']}</div>
</div>

<!-- AI 建議賣點 -->
<div class="data-card" style="border-left: 5px solid {exit_strategy['color']}; height: 160px; margin-bottom: 0; background: linear-gradient(180deg, {exit_strategy['color']}0a 0%, #161B22 100%);">
    <div style="color: #8B949E; font-size: 0.7rem; margin-bottom: 5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">AI 建議賣點 ({exit_strategy['confidence']})</div>
    <div style="display: flex; align-items: baseline; gap: 6px; margin-bottom: 4px;">
        <span style="font-size: 1.3rem; font-weight: 800; color: {exit_strategy['color']};">{c_symbol}{exit_strategy['price']:.1f}</span>
        <span style="color: #8B949E; font-size: 0.7rem;">{exit_strategy['action']}</span>
    </div>
    <div style="color: #3FB950; font-size: 0.75rem; font-weight: 700; margin-bottom: 4px;">出場保命線: {c_symbol}{exit_strategy['trailing_stop']:.1f}</div>
    <div style="color: #C9D1D9; font-size: 0.7rem; line-height: 1.3;">{exit_strategy['desc']}</div>
</div>

<!-- 7日趨勢預測 -->
<div class="data-card" style="border-left: 5px solid #BC8CF2; height: 160px; margin-bottom: 0; background: linear-gradient(180deg, rgba(188, 140, 242, 0.05) 0%, #161B22 100%);">
    <div style="color: #8B949E; font-size: 0.7rem; margin-bottom: 5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">7日趨勢預測</div>
    <div style="display: flex; align-items: baseline; gap: 6px; margin-8px;">
        <span style="font-size: 1.3rem; font-weight: 800; color: #BC8CF2;">{c_symbol}{median_sim_7d:.1f}</span>
        <span style="color: {"#3FB950" if win_prob_7d > 50 else "#FF7B72"}; font-size: 0.8rem; font-weight: 700;">({win_prob_7d:.1f}%)</span>
    </div>
    <div style="color: #8B949E; font-size: 0.7rem; margin-top: 20px; border-top: 1px solid #30363D; padding-top: 8px;">蒙地卡羅模擬勝率</div>
</div>
</div>''', unsafe_allow_html=True)

            # Overview Tab: Advanced Plotly Chart
            financial_data = get_financial_data(resolved_ticker)
            quant_factors = calculate_quant_factors(
                data, ticker_metadata, rsi_series, atr_series, 
                supertrend_dir=supertrend_dir.iloc[-1], 
                financial_data=financial_data
            )
            
            col_chart, col_quant = st.columns([2, 1])
            
            with col_chart:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                   vertical_spacing=0.05, subplot_titles=(f'<b>{resolved_ticker} 價量走勢</b>', '<b>成交量</b>'), 
                                   row_width=[0.25, 0.75])

                # Candlestick - 專業配色與樣式
                fig.add_trace(go.Candlestick(
                    x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], 
                    name='K線', 
                    increasing_line_color=m_colors["up"], decreasing_line_color=m_colors["down"],
                    increasing_fillcolor=m_colors["up"], decreasing_fillcolor=m_colors["down"],
                    line=dict(width=1)
                ), row=1, col=1)
                
                # MA - 柔和色調與線條優化
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(window=5).mean(), name='5MA', line=dict(color='#FFD700', width=1.2, shape='spline'), opacity=0.8), row=1, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(window=20).mean(), name='20MA', line=dict(color='#2962FF', width=1.5, shape='spline')), row=1, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(window=50).mean(), name='50MA', line=dict(color='#FF6D00', width=1.8, shape='spline')), row=1, col=1)

                # Volume - 漸層與半透明優化
                v_colors = [m_colors["up"] if row['Close'] >= row['Open'] else m_colors["down"] for index, row in data.iterrows()]
                fig.add_trace(go.Bar(
                    x=data.index, y=data['Volume'], 
                    name='成交量', 
                    marker_color=v_colors, 
                    opacity=0.6,
                    marker_line_width=0
                ), row=2, col=1)

                fig.update_layout(
                    template="plotly_dark",
                    height=650,
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=10, r=10, t=50, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(
                        orientation="h", 
                        yanchor="bottom", y=1.02, 
                        xanchor="right", x=1,
                        font=dict(size=10, color="#8B949E"),
                        bgcolor="rgba(0,0,0,0)"
                    ),
                    hovermode="x unified",
                    hoverlabel=dict(bgcolor="#161B22", font_size=12, font_family="monospace")
                )
                
                # 座標軸優化：網格線淡化與標籤美化
                fig.update_xaxes(
                    showgrid=True, gridwidth=1, gridcolor='rgba(48, 54, 61, 0.3)', 
                    zeroline=False, tickfont=dict(color="#8B949E", size=10),
                    spikemode="across", spikesnap="cursor", spikedash="dot", spikethickness=1, spikecolor="#8B949E"
                )
                fig.update_yaxes(
                    showgrid=True, gridwidth=1, gridcolor='rgba(48, 54, 61, 0.3)', 
                    zeroline=False, tickfont=dict(color="#8B949E", size=10),
                    side="right" # 價格放在右側更符合交易習慣
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})

            with col_quant:
                # 數據完整性提示 (Data Integrity Warning)
                if financial_data is None:
                    st.warning("⚠️ 財務數據缺失，部分量化因子採用技術面估算 (Missing financial data, using technical estimates for some factors)")

                # 量化評分雷達圖 - 升級版
                categories = list(quant_factors.keys())
                values = list(quant_factors.values())
                
                # 為了閉合雷達圖，需要將首個點重複
                r_values = values + [values[0]]
                theta_categories = categories + [categories[0]]
                
                # 因子解釋映射
                factor_desc = {
                    '趨勢 (Trend)': '反映股價長期均線排列與波段方向',
                    '動能 (Momentum)': '結合 RSI 強度與價格變化率 (ROC)',
                    '波動 (Volatility)': '衡量股價波動平穩度 (ATR 調節)',
                    '量能 (Volume)': '分析成交量共振與量價背離情況',
                    '價值 (Value)': '估算 PE/PB 處於歷史區間的位置',
                    '品質 (Quality)': '評估 ROE、利潤率與債務結構',
                    '成長 (Growth)': '追蹤營收與淨利的年度/季度增長率'
                }
                
                fig_radar = go.Figure()
                
                # 添加基準參考線 (50分)
                fig_radar.add_trace(go.Scatterpolar(
                    r=[50] * len(theta_categories),
                    theta=theta_categories,
                    fill=None,
                    line=dict(color='rgba(139, 148, 158, 0.3)', width=1, dash='dash'),
                    hoverinfo='none',
                    showlegend=False,
                    name='中性基準'
                ))
                
                # 主因子層
                fig_radar.add_trace(go.Scatterpolar(
                    r=r_values,
                    theta=theta_categories,
                    fill='toself',
                    fillcolor='rgba(88, 166, 255, 0.15)',
                    line=dict(color='#58A6FF', width=2, shape='linear'),
                    marker=dict(
                        color='#58A6FF',
                        size=8,
                        symbol='diamond',
                        line=dict(color='#FFFFFF', width=1)
                    ),
                    name='因子評分',
                    customdata=[factor_desc.get(cat, '') for cat in theta_categories],
                    hovertemplate="<b>%{theta}</b>: %{r:.1f}分<br>%{customdata}<extra></extra>"
                ))
                
                # 添加一層發光效果
                fig_radar.add_trace(go.Scatterpolar(
                    r=r_values,
                    theta=theta_categories,
                    mode='lines',
                    line=dict(color='#58A6FF', width=6, shape='linear'),
                    opacity=0.1,
                    hoverinfo='skip',
                    showlegend=False
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        gridshape='linear',
                        radialaxis=dict(
                            visible=True, 
                            range=[0, 100], 
                            gridcolor='rgba(139, 148, 158, 0.15)', 
                            tickfont=dict(size=8, color="#8B949E", family="Monaco, monospace"),
                            angle=0,
                            tickangle=0,
                            showline=False,
                            gridwidth=1,
                            ticks=''
                        ),
                        angularaxis=dict(
                            gridcolor='rgba(139, 148, 158, 0.15)', 
                            tickfont=dict(size=10, color="#E6EAF1"),
                            rotation=90,
                            direction="clockwise",
                            gridwidth=1
                        ),
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    showlegend=False,
                    template="plotly_dark",
                    height=280,
                    margin=dict(l=40, r=40, t=30, b=30),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    hovermode='closest'
                )
                st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})
                
                # 綜合評分顯示 - 升級版
                avg_score = sum(values) / len(values)
                score_color = "#26a69a" if avg_score > 65 else "#D29922" if avg_score > 50 else "#ef5350"
                
                # 計算與前一交易日的變化 (這裡模擬一個小幅波動，或是實際計算)
                # 為了簡單起見，我們先顯示狀態標籤
                status_text = "市場領導者" if avg_score > 80 else "趨勢強勁" if avg_score > 65 else "中性整理" if avg_score > 50 else "弱勢觀察"
                
                st.markdown(f'''<div class="data-card" style="text-align: center; background: linear-gradient(135deg, #1C2128 0%, #0D1117 100%); border: 1px solid #30363D; border-top: 3px solid {score_color}; padding: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); position: relative; overflow: hidden;">
<div style="position: absolute; top: 0; right: 0; width: 60px; height: 60px; background: radial-gradient(circle at top right, {score_color}10, transparent); pointer-events: none;"></div>
<div style="color: #8B949E; font-size: 0.65rem; font-weight: 800; margin-bottom: 8px; letter-spacing: 1.5px; text-transform: uppercase;">Quantitative Score</div>
<div style="display: flex; justify-content: center; align-items: baseline; gap: 4px;">
    <div style="font-size: 2.8rem; font-weight: 900; color: {score_color}; line-height: 1; text-shadow: 0 0 20px {score_color}33;">{avg_score:.1f}</div>
    <div style="color: #8B949E; font-size: 0.9rem; font-weight: 600; opacity: 0.5;">/ 100</div>
</div>
<div style="margin-top: 12px; padding: 4px 14px; background: {score_color}10; color: {score_color}; border-radius: 8px; font-size: 0.8rem; font-weight: 700; display: inline-block; border: 1px solid {score_color}20;">
{status_text}
</div>
<div style="margin-top: 15px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; text-align: left; border-top: 1px solid #30363D; padding-top: 12px;">
<div style="font-size: 0.7rem; color: #8B949E; display: flex; align-items: center; gap: 5px;">
    <span style="width: 5px; height: 5px; background: {"#26a69a" if values[0]>60 else "#ef5350"}; border-radius: 50%; box-shadow: 0 0 5px {"#26a69a" if values[0]>60 else "#ef5350"};"></span>
    主力: <span style="color: {"#26a69a" if values[0]>60 else "#ef5350"}; font-weight: 700;">{"看多" if values[0]>60 else "保守"}</span>
</div>
<div style="font-size: 0.7rem; color: #8B949E; display: flex; align-items: center; gap: 5px;">
    <span style="width: 5px; height: 5px; background: {"#26a69a" if values[4]>60 else "#ef5350"}; border-radius: 50%; box-shadow: 0 0 5px {"#26a69a" if values[4]>60 else "#ef5350"};"></span>
    價值: <span style="color: {"#26a69a" if values[4]>60 else "#ef5350"}; font-weight: 700;">{"合理" if values[4]>60 else "偏高"}</span>
</div>
</div>
</div>''', unsafe_allow_html=True)

            # 2. 財務健康度與統計摘要 (Financial Health & Stats Section)
            financial_statements = get_financial_data(resolved_ticker)
            perf_html = ""
            if financial_statements is not None and len(financial_statements) >= 2:
                latest_year = financial_statements.index[-1]
                prev_year = financial_statements.index[-2]
                rev_latest = financial_statements['Total Revenue'].iloc[-1]
                rev_prev = financial_statements['Total Revenue'].iloc[-2]
                ni_latest = financial_statements['Net Income'].iloc[-1]
                ni_prev = financial_statements['Net Income'].iloc[-2]
                
                # 計算成長率 (Calculate Growth Rates)
                rev_growth = (rev_latest / rev_prev - 1) * 100 if rev_prev != 0 else 0
                ni_growth = (ni_latest / ni_prev - 1) * 100 if ni_prev != 0 else 0
                
                rev_growth_color = "#26a69a" if rev_growth > 0 else "#ef5350"
                ni_growth_color = "#26a69a" if ni_growth > 0 else "#ef5350"
                
                perf_html = f"""<div class="data-card" style="margin-top: 20px; background: rgba(88, 166, 255, 0.03); border: 1px solid rgba(88, 166, 255, 0.1); display: flex; justify-content: space-around; align-items: center; gap: 20px; padding: 22px; margin-bottom: 0;">
<div style="text-align: center; flex: 1;">
<p style="color: #8B949E; font-size: 0.75rem; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; font-weight: 700;">年度營收 ({latest_year})</p>
<div style="display: flex; flex-direction: column; align-items: center;">
<span style="font-size: 1.3rem; font-weight: 900; color: #F0F6FC; letter-spacing: -0.5px;">{c_symbol}{rev_latest/1e6:,.0f}M</span>
<span style="font-size: 0.75rem; color: {rev_growth_color}; font-weight: 800; background: {rev_growth_color}15; padding: 2px 10px; border-radius: 6px; margin-top: 8px; border: 1px solid {rev_growth_color}25;">
{rev_growth:+.1f}% YoY
</span>
</div>
</div>
<div style="width: 1px; height: 50px; background: rgba(48, 54, 61, 0.8);"></div>
<div style="text-align: center; flex: 1;">
<p style="color: #8B949E; font-size: 0.75rem; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; font-weight: 700;">年度淨利 ({latest_year})</p>
<div style="display: flex; flex-direction: column; align-items: center;">
<span style="font-size: 1.3rem; font-weight: 900; color: #F0F6FC; letter-spacing: -0.5px;">{c_symbol}{ni_latest/1e6:,.0f}M</span>
<span style="font-size: 0.75rem; color: {ni_growth_color}; font-weight: 800; background: {ni_growth_color}15; padding: 2px 10px; border-radius: 6px; margin-top: 8px; border: 1px solid {ni_growth_color}25;">
{ni_growth:+.1f}% YoY
</span>
</div>
</div>
</div>"""


            col_gauge, col_stats_data = st.columns([1, 2])
            
            with col_gauge:
                st.plotly_chart(create_tv_gauge(sig_val, sig_text, sig_color, m_colors=m_colors), width="stretch")
            
            with col_stats_data:
                # 根據健康得分定義動態樣式 (Define dynamic styling based on health scores)
                profitability_score, leverage_score, cashflow_score = p_score, l_score, c_score
                avg_health = (profitability_score + leverage_score + cashflow_score) / 3
                health_border = "#26a69a" if avg_health > 7 else "#ef5350" if avg_health < 4 else "#30363D"
                health_bg = "rgba(38, 166, 154, 0.05)" if avg_health > 7 else "rgba(239, 83, 80, 0.05)" if avg_health < 4 else "#161B22"
                
                st.markdown(f'''<div class="data-card" style="height: 100%; margin-bottom: 0; border: 1px solid {health_border}55; background: linear-gradient(180deg, {health_bg}, rgba(13, 17, 23, 0.98)); border-radius: 18px; padding: 24px; box-shadow: 0 12px 40px rgba(0,0,0,0.3); border-top: 4px solid {health_border};">
<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 24px;">
<div>
<h3 style="margin: 0; font-size: 1.25rem; font-weight: 800; color: #FFFFFF; display: flex; align-items: center; gap: 12px; letter-spacing: -0.2px;">
<span style="background: {health_border}; width: 5px; height: 22px; border-radius: 2.5px;"></span>
{t["key_stats"]}
</h3>
<p style="margin: 6px 0 0 17px; font-size: 0.75rem; color: #8B949E; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 600;">基本面分析 (Fundamental Analysis)</p>
</div>
<div style="text-align: right;">
<div style="font-size: 0.7rem; color: #8B949E; margin-bottom: 6px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">財務健康狀態</div>
<span style="font-size: 0.8rem; background: {health_border}25; color: {health_border}; padding: 5px 12px; border-radius: 8px; border: 1px solid {health_border}50; font-weight: 900; letter-spacing: 0.5px;">
{'優異 (EXCELLENT)' if avg_health > 7 else '警示 (WARNING)' if avg_health < 4 else '穩定 (STABLE)'}
</span>
</div>
</div>
<div style="display: flex; gap: 12px; margin-bottom: 24px; padding: 12px; background: rgba(48, 54, 61, 0.25); border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
<div style="flex: 1; text-align: center; border-right: 1px solid rgba(48, 54, 61, 0.6);">
<div style="font-size: 0.65rem; color: #8B949E; margin-bottom: 4px; font-weight: 600;">獲利能力</div>
<div style="font-size: 1.1rem; font-weight: 900; color: #FFFFFF;">{profitability_score}<span style="font-size: 0.7rem; color: #8B949E; font-weight: 400; margin-left: 2px;">/10</span></div>
</div>
<div style="flex: 1; text-align: center; border-right: 1px solid rgba(48, 54, 61, 0.6);">
<div style="font-size: 0.65rem; color: #8B949E; margin-bottom: 4px; font-weight: 600;">財務槓桿</div>
<div style="font-size: 1.1rem; font-weight: 900; color: #FFFFFF;">{leverage_score}<span style="font-size: 0.7rem; color: #8B949E; font-weight: 400; margin-left: 2px;">/10</span></div>
</div>
<div style="flex: 1; text-align: center;">
<div style="font-size: 0.65rem; color: #8B949E; margin-bottom: 4px; font-weight: 600;">現金流量</div>
<div style="font-size: 1.1rem; font-weight: 900; color: #FFFFFF;">{cashflow_score}<span style="font-size: 0.7rem; color: #8B949E; font-weight: 400; margin-left: 2px;">/10</span></div>
</div>
</div>
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 16px;">
<div class="data-card" style="background: rgba(48, 54, 61, 0.18); padding: 14px; margin-bottom: 0;">
    <p style="color: #58A6FF; font-size: 0.75rem; font-weight: 800; margin-bottom: 10px; text-transform: uppercase; border-bottom: 1px solid rgba(88, 166, 255, 0.25); padding-bottom: 6px; letter-spacing: 0.5px;">估值分析 (Valuation)</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 6px;"><span style="color: #8B949E;">{t['trailing_pe']}</span><span style="font-weight: 700; color: #FFFFFF;">{ticker_metadata.get('trailingPE', 'N/A')}</span></div>
    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 6px;"><span style="color: #8B949E;">{t['forward_pe']}</span><span style="font-weight: 700; color: #FFFFFF;">{ticker_metadata.get('forwardPE', 'N/A')}</span></div>
    <div style="display: flex; justify-content: space-between; font-size: 0.85rem;"><span style="color: #8B949E;">{t['div_yield']}</span><span style="font-weight: 700; color: #3FB950;">{ticker_metadata.get('dividendYield', 0)*100:.2f}%</span></div>
</div>
<div class="data-card" style="background: rgba(48, 54, 61, 0.18); padding: 14px; margin-bottom: 0;">
    <p style="color: #D29922; font-size: 0.75rem; font-weight: 800; margin-bottom: 10px; text-transform: uppercase; border-bottom: 1px solid rgba(210, 153, 34, 0.25); padding-bottom: 6px; letter-spacing: 0.5px;">價格區間 (Price Range)</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 6px;"><span style="color: #8B949E;">52 週最高</span><span style="font-weight: 700; color: #FFFFFF;">{ticker_metadata.get('fiftyTwoWeekHigh', 'N/A')}</span></div>
    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 6px;"><span style="color: #8B949E;">52 週最低</span><span style="font-weight: 700; color: #FFFFFF;">{ticker_metadata.get('fiftyTwoWeekLow', 'N/A')}</span></div>
    <div style="display: flex; justify-content: space-between; font-size: 0.85rem;"><span style="color: #8B949E;">貝塔係數 (Beta)</span><span style="font-weight: 700; color: #FFFFFF;">{ticker_metadata.get('beta', 'N/A')}</span></div>
</div>
<div class="data-card" style="background: rgba(48, 54, 61, 0.18); padding: 14px; margin-bottom: 0;">
    <p style="color: #BC8CFF; font-size: 0.75rem; font-weight: 800; margin-bottom: 10px; text-transform: uppercase; border-bottom: 1px solid rgba(188, 140, 255, 0.25); padding-bottom: 6px; letter-spacing: 0.5px;">分析師預測 (Forecast)</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 6px;">
        <span style="color: #8B949E;">目標中位價</span>
        <span style="font-weight: 700; color: #58A6FF;">{ticker_metadata.get('targetMedianPrice', 'N/A')}</span>
    </div>
    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 8px;">
        <span style="color: #8B949E;">潛在漲幅</span>
        <span style="font-weight: 700; color: {m_colors['up'] if (ticker_metadata.get('targetMedianPrice', current_price) / current_price) - 1 > 0 else m_colors['down']};">
            {f"+{((ticker_metadata.get('targetMedianPrice', current_price) / current_price) - 1) * 100:.1f}%" if ticker_metadata.get('targetMedianPrice') else 'N/A'}
        </span>
    </div>
    <div style="margin-top: 10px;">
        <div style="position: relative; width: 100%; height: 8px; background: #21262d; border-radius: 4px; overflow: hidden; border: 1px solid rgba(255,255,255,0.05);">
            <div style="position: absolute; left: 0; top: 0; height: 100%; width: 100%; background: linear-gradient(90deg, {m_colors['down']} 0%, #787b86 50%, {m_colors['up']} 100%); opacity: 0.25;"></div>
            <div style="position: absolute; 
                left: {
                    (
                        min(100, max(0, ((current_price - ticker_metadata.get('targetLowPrice', current_price*0.8)) / 
                        (max(0.01, ticker_metadata.get('targetHighPrice', current_price*1.2) - ticker_metadata.get('targetLowPrice', current_price*0.8))) * 100)))
                    ) if ticker_metadata.get('targetHighPrice') and ticker_metadata.get('targetHighPrice') != ticker_metadata.get('targetLowPrice') else 50
                }%; 
                top: 0; height: 100%; width: 4px; background: #FFFFFF; box-shadow: 0 0 8px rgba(255,255,255,0.8); border-radius: 2px;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.6rem; color: #8B949E; margin-top: 4px; font-weight: 500;">
            <span>目標低</span>
            <span>目標高</span>
        </div>
    </div>
</div>
</div>
{perf_html}
</div>''', unsafe_allow_html=True)
 
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
                data.index[-1],
                supertrend_dir=supertrend_dir.iloc[-1],
                m_colors=m_colors
            )
            
            # 專業診斷區塊佈局 (Professional Diagnostic Section with Grid Layout)
            st.markdown(f'''<div class="data-card" style="border-top: 4px solid {insight_report['action']['color']}; border-color: {insight_report['action']['color']}77; background: linear-gradient(180deg, {insight_report['action']['color']}0a 0%, #0D1117 100%); transition: all 0.5s ease; border-radius: 18px; padding: 24px; box-shadow: 0 12px 40px rgba(0,0,0,0.3);">
<h3 style="margin-top:0; color: {insight_report['action']['color']}; font-size: 1.25rem; font-weight: 800; display: flex; align-items: center; gap: 12px; margin-bottom: 24px; letter-spacing: -0.2px;">
<span style="font-size: 1.6rem;">⚖️</span> {ticker_display_name} - 專家技術診斷報告
</h3>
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 18px;">
<!-- RSI Diagnostic -->
<div class="data-card" style="background: rgba(48, 54, 61, 0.25); border-left: 4px solid {insight_report['rsi']['color']}; margin-bottom: 0;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
        <span style="color: #8B949E; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">RSI 強度</span>
        <span style="background: {insight_report['rsi']['color']}; color: white; padding: 3px 10px; border-radius: 10px; font-size: 0.7rem; font-weight: 800;">{insight_report['rsi']['status']}</span>
    </div>
    <div style="font-size: 1.75rem; font-weight: 900; color: #FFFFFF; margin-bottom: 6px; letter-spacing: -0.5px;">{insight_report['rsi']['val']}</div>
    <div style="color: #E0E0E0; font-size: 0.85rem; line-height: 1.5; font-weight: 400;">{insight_report['rsi']['desc']}</div>
</div>

<!-- MACD Diagnostic -->
<div class="data-card" style="background: rgba(48, 54, 61, 0.25); border-left: 4px solid {insight_report['macd']['color']}; margin-bottom: 0;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
        <span style="color: #8B949E; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">MACD 差值</span>
        <span style="background: {insight_report['macd']['color']}; color: white; padding: 3px 10px; border-radius: 10px; font-size: 0.7rem; font-weight: 800;">{insight_report['macd']['status']}</span>
    </div>
    <div style="font-size: 1.75rem; font-weight: 900; color: #FFFFFF; margin-bottom: 6px; letter-spacing: -0.5px;">{insight_report['macd']['val']}</div>
    <div style="color: #E0E0E0; font-size: 0.85rem; line-height: 1.5; font-weight: 400;">{insight_report['macd']['desc']}</div>
</div>

<!-- AI Layout Suggestion -->
<div class="data-card" style="background: linear-gradient(180deg, rgba(88, 166, 255, 0.1) 0%, #161B22 100%); border: 1px dashed rgba(88, 166, 255, 0.4); border-left: 4px solid #58A6FF; margin-bottom: 0;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
        <span style="color: #58A6FF; font-size: 0.8rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px;">AI 建議佈局</span>
        <span style="color: {entry_strategy['color']}; font-weight: 900; font-size: 0.95rem;">{entry_strategy['action']} (${entry_strategy['price']:.1f})</span>
    </div>
    <div style="font-size: 1.1rem; color: #FFFFFF; font-weight: 700; margin-bottom: 8px;">策略部署 (信心:{entry_strategy['confidence']})</div>
    <div style="color: #E0E0E0; font-size: 0.85rem; line-height: 1.5;">{entry_strategy['desc']}</div>
</div>

<!-- Swing Advice -->
<div class="data-card" style="background: linear-gradient(180deg, rgba(255, 215, 0, 0.05) 0%, #161B22 100%); border: 1px solid rgba(255, 215, 0, 0.25); border-left: 4px solid #D29922; margin-bottom: 0;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
        <span style="color: #D29922; font-size: 0.8rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px;">波段操作建議</span>
        <span style="font-size: 1.4rem;">💡</span>
    </div>
    <div style="font-size: 1.1rem; color: #FFFFFF; font-weight: 700; margin-bottom: 8px;">操作方針</div>
    <div style="color: #E0E0E0; font-size: 0.85rem; line-height: 1.5;">{insight_report['swing_advice']}</div>
</div>
</div>
</div>''', unsafe_allow_html=True)


        with tab2:
            st.markdown(f'''<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px;">
<!-- EMA Card -->
<div class="data-card" style="margin-bottom: 0; border-top: 4px solid #58A6FF; background: linear-gradient(145deg, rgba(88, 166, 255, 0.05) 0%, #161B22 100%);">
    <div style="color: #8B949E; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px;">EMA 趨勢狀態 (波段)</div>
    <div style="font-size: 1.2rem; font-weight: 800; margin: 12px 0; color: #F0F6FC; letter-spacing: -0.2px;">
        {'🔥 多頭排列 (主升段)' if ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1] else 
         '📈 偏多整理 (初升/回測)' if ema20.iloc[-1] > ema50.iloc[-1] else 
         '❄️ 空頭排列 (主跌段)' if ema20.iloc[-1] < ema50.iloc[-1] < ema200.iloc[-1] else 
         '🔄 趨勢轉折/震盪'}
    </div>
    <div style="color: #58A6FF; font-size: 0.75rem; font-family: 'Roboto Mono', monospace; font-weight: 500;">EMA200 支撐: {ema200.iloc[-1]:.2f}</div>
</div>

<!-- Bollinger Card -->
<div class="data-card" style="margin-bottom: 0; border-top: 4px solid #BC8CF2; background: linear-gradient(145deg, rgba(188, 140, 242, 0.05) 0%, #161B22 100%);">
    <div style="color: #8B949E; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px;">布林帶寬 (波動率)</div>
    <div style="font-size: 1.4rem; font-weight: 900; margin: 12px 0; color: #F0F6FC; letter-spacing: -0.5px;">{((bb_upper.iloc[-1] - bb_lower.iloc[-1]) / ma20.iloc[-1] * 100):.2f}%</div>
    <div style="color: #BC8CF2; font-size: 0.75rem; font-family: 'Roboto Mono', monospace; font-weight: 500;">{'💎 壓縮中 (醞釀變盤)' if (bb_upper.iloc[-1] - bb_lower.iloc[-1]) < (bb_upper.rolling(100).mean().iloc[-1] - bb_lower.rolling(100).mean().iloc[-1]) else '🌊 擴張中 (趨勢進行)'}</div>
</div>

<!-- RSI Card -->
<div class="data-card" style="margin-bottom: 0; border-top: 4px solid {"#3FB950" if rsi_series.iloc[-1] < 35 else "#FF7B72" if rsi_series.iloc[-1] > 65 else '#8B949E'}; background: linear-gradient(145deg, rgba(63, 185, 80, 0.05) 0%, #161B22 100%);">
    <div style="color: #8B949E; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px;">RSI 強度 (波段)</div>
    <div style="font-size: 1.4rem; font-weight: 900; margin: 12px 0; color: #F0F6FC; letter-spacing: -0.5px;">{rsi_series.iloc[-1]:.1f}</div>
    <div style="color: {"#FF7B72" if rsi_series.iloc[-1] > 60 else "#3FB950" if rsi_series.iloc[-1] < 40 else '#8B949E'}; font-size: 0.75rem; font-family: 'Roboto Mono', monospace; font-weight: 500;">
        {'⚡ 強勢區域' if rsi_series.iloc[-1] > 60 else '🛡️ 弱勢區域' if rsi_series.iloc[-1] < 40 else '⚖️ 中性區間'}
    </div>
</div>

<!-- VR Card -->
<div class="data-card" style="margin-bottom: 0; border-top: 4px solid {"#FF7B72" if vr_series.iloc[-1] > 160 else "#3FB950" if vr_series.iloc[-1] < 70 else '#FFD700'}; background: linear-gradient(145deg, rgba(255, 123, 114, 0.05) 0%, #161B22 100%);">
    <div style="color: #8B949E; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px;">VR 成交量比 (波段)</div>
    <div style="font-size: 1.4rem; font-weight: 900; margin: 12px 0; color: #F0F6FC; letter-spacing: -0.5px;">{vr_series.iloc[-1]:.1f}%</div>
    <div style="color: {"#FF7B72" if vr_series.iloc[-1] > 160 else "#3FB950" if vr_series.iloc[-1] < 70 else '#8B949E'}; font-size: 0.75rem; font-family: 'Roboto Mono', monospace; font-weight: 500;">
        {'🔥 量能過熱' if vr_series.iloc[-1] > 160 else '❄️ 量能低迷' if vr_series.iloc[-1] < 70 else '🍃 量能溫和'}
    </div>
</div>

<!-- SuperTrend Card -->
<div class="data-card" style="margin-bottom: 0; border-top: 4px solid {"#3FB950" if supertrend_dir.iloc[-1] == 1 else "#FF7B72"}; background: linear-gradient(145deg, rgba(63, 185, 80, 0.05) 0%, #161B22 100%);">
    <div style="color: #8B949E; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px;">SuperTrend 波段方向</div>
    <div style="font-size: 1.2rem; font-weight: 800; margin: 12px 0; color: #F0F6FC; letter-spacing: -0.2px;">{'🟢 多頭趨勢' if supertrend_dir.iloc[-1] == 1 else '🔴 空頭趨勢'}</div>
    <div style="color: #8B949E; font-size: 0.75rem; font-family: 'Roboto Mono', monospace; font-weight: 500;">波段防守: {supertrend_line.iloc[-1]:.2f}</div>
</div>
</div>''', unsafe_allow_html=True)


            # Combined Price + EMA + BB Chart
            fig_tech = go.Figure()
            
            # Bollinger Bands Area (Smoother Fill)
            fig_tech.add_trace(go.Scatter(x=data.index, y=bb_upper, line=dict(color='rgba(88, 166, 255, 0.05)', width=0), showlegend=False, hoverinfo='skip'))
            fig_tech.add_trace(go.Scatter(x=data.index, y=bb_lower, line=dict(color='rgba(88, 166, 255, 0.05)', width=0), fill='tonexty', fillcolor='rgba(88, 166, 255, 0.08)', name='布林通道'))
            
            # Price (Thicker Line)
            fig_tech.add_trace(go.Scatter(x=data.index, y=data['Close'], name='收盤價', line=dict(color='#FFFFFF', width=2.5)))
            
            # EMAs (Enhanced Visibility)
            fig_tech.add_trace(go.Scatter(x=data.index, y=ema20, name='EMA 20', line=dict(color='#58A6FF', width=1.5, dash='dot', shape='spline')))
            fig_tech.add_trace(go.Scatter(x=data.index, y=ema50, name='EMA 50', line=dict(color='#BC8CF2', width=1.5, dash='dot', shape='spline')))
            fig_tech.add_trace(go.Scatter(x=data.index, y=ema200, name='EMA 200', line=dict(color='#FF6D00', width=2.0, shape='spline')))

            # SuperTrend (Better Contrast)
            st_color = m_colors["up"] if supertrend_dir.iloc[-1] == 1 else m_colors["down"]
            fig_tech.add_trace(go.Scatter(x=data.index, y=supertrend_line, name='SuperTrend', line=dict(color=st_color, width=1.5, dash='dash'), opacity=0.6))

            fig_tech.update_layout(
                title="<b>EMA 趨勢與布林通道分析</b>",
                template="plotly_dark", height=500,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=50, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
                hovermode="x unified",
                xaxis=dict(showgrid=True, gridcolor='rgba(48, 54, 61, 0.2)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(48, 54, 61, 0.2)', side="right")
            )
            st.plotly_chart(fig_tech, use_container_width=True)


            col_rsi, col_macd = st.columns(2)
            with col_rsi:
                # RSI Chart
                fig_rsi = go.Figure()
                # 添加背景區域
                fig_rsi.add_hrect(y0=70, y1=100, fillcolor="rgba(239, 83, 80, 0.1)", line_width=0)
                fig_rsi.add_hrect(y0=0, y1=30, fillcolor="rgba(38, 166, 154, 0.1)", line_width=0)
                
                fig_rsi.add_trace(go.Scatter(x=data.index, y=rsi_series, name='RSI', line=dict(color='#BC8CF2', width=2.5, shape='spline')))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color=m_colors["up"], opacity=0.4)
                fig_rsi.add_hline(y=30, line_dash="dash", line_color=m_colors["down"], opacity=0.4)
                fig_rsi.update_layout(
                    title="<b>RSI (14) 相對強弱指標</b>", yaxis_range=[0, 100], height=300, template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=10, r=10, t=50, b=10),
                    hovermode="x unified",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', side="right")
                )
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            with col_macd:
                # MACD Chart
                fig_macd = go.Figure()
                hist = macd_series - signal_series
                fig_macd.add_trace(go.Scatter(x=data.index, y=macd_series, name='MACD', line=dict(color='#58A6FF', width=1.5)))
                fig_macd.add_trace(go.Scatter(x=data.index, y=signal_series, name='Signal', line=dict(color='#FF6D00', width=1.5)))
                
                hist_colors = [m_colors["up"] if val >= 0 else m_colors["down"] for val in hist]
                fig_macd.add_bar(x=data.index, y=hist, name='Histogram', marker_color=hist_colors, opacity=0.7)
                
                fig_macd.update_layout(
                    title="<b>MACD 指數平滑異同移動平均線</b>", height=300, template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=10, r=10, t=50, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
                    hovermode="x unified",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', side="right")
                )
                st.plotly_chart(fig_macd, use_container_width=True)

            col_kd, col_vr = st.columns(2)
            with col_kd:
                # KD Chart
                fig_kd = go.Figure()
                # 添加背景區域
                fig_kd.add_hrect(y0=80, y1=100, fillcolor="rgba(239, 83, 80, 0.1)", line_width=0)
                fig_kd.add_hrect(y0=0, y1=20, fillcolor="rgba(38, 166, 154, 0.1)", line_width=0)
                
                fig_kd.add_trace(go.Scatter(x=data.index, y=k_series, name='K線', line=dict(color='#FFD700', width=2, shape='spline')))
                fig_kd.add_trace(go.Scatter(x=data.index, y=d_series, name='D線', line=dict(color='#00BFFF', width=2, shape='spline')))
                fig_kd.update_layout(
                    title="<b>KD 指標 (9, 3, 3) 隨機指標</b>", yaxis_range=[0, 100], height=300, template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=10, r=10, t=50, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
                    hovermode="x unified",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', side="right")
                )
                st.plotly_chart(fig_kd, use_container_width=True)
            
            with col_vr:
                # VR Chart
                fig_vr = go.Figure()
                fig_vr.add_trace(go.Scatter(x=data.index, y=vr_series, name='VR', line=dict(color='#FF6D00', width=2.5, shape='spline')))
                fig_vr.add_hline(y=160, line_dash="dash", line_color=m_colors["up"], opacity=0.4)
                fig_vr.add_hline(y=70, line_dash="dash", line_color=m_colors["down"], opacity=0.4)
                fig_vr.update_layout(
                    title="<b>VR 容量比率 (26) 成交量分析</b>", height=300, template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=10, r=10, t=50, b=10),
                    hovermode="x unified",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', side="right")
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
                
                # 1. 繪製 80% 信心區間 (P10 - P90) - 更柔和的漸層
                fig_mc.add_trace(go.Scatter(
                    x=plot_dates, y=p90,
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip"
                ))
                fig_mc.add_trace(go.Scatter(
                    x=plot_dates, y=p10,
                    fill="tonexty",
                    fillcolor="rgba(88, 166, 255, 0.08)",
                    line=dict(width=0),
                    name="80% 信心區間 (P10-P90)"
                ))
                
                # 2. 繪製 50% 核心區間 (P25 - P75) - 較深的層次感
                fig_mc.add_trace(go.Scatter(
                    x=plot_dates, y=p75,
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip"
                ))
                fig_mc.add_trace(go.Scatter(
                    x=plot_dates, y=p25,
                    fill="tonexty",
                    fillcolor="rgba(88, 166, 255, 0.18)",
                    line=dict(width=0),
                    name="50% 核心區間 (P25-P75)"
                ))
                
                # 3. 繪製期望值路徑 - 金色發光感
                fig_mc.add_trace(go.Scatter(
                    x=plot_dates, y=expected_path,
                    line=dict(color="#FFD700", width=2.5, shape='spline'),
                    name="預期平均路徑"
                ))

                # 4. 繪製中位數路徑 - 藍色發光感
                fig_mc.add_trace(go.Scatter(
                    x=plot_dates, y=p50,
                    line=dict(color="#58a6ff", width=2, dash="dash", shape='spline'),
                    name="中位數路徑"
                ))
                
                fig_mc.update_layout(
                    title=dict(
                        text=f"<b>{resolved_ticker} 未來 60 日波段路徑機率分佈預測 (AI 增強版)</b>", 
                        font=dict(size=18, color="#F0F6FC"),
                        x=0.01, y=0.95
                    ),
                    template="plotly_dark",
                    hovermode="x unified",
                    height=550,
                    margin=dict(l=10, r=10, t=80, b=20),
                    legend=dict(
                        orientation="h", 
                        yanchor="bottom", y=1.02, 
                        xanchor="right", x=1,
                        bgcolor="rgba(0,0,0,0)",
                        font=dict(size=11, color="#8B949E")
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    hoverlabel=dict(bgcolor="#161B22", font_size=12, font_family="monospace")
                )

                fig_mc.update_xaxes(
                    showgrid=True, gridwidth=1, gridcolor='rgba(48, 54, 61, 0.2)',
                    tickfont=dict(color="#8B949E", size=10),
                    title_text=""
                )
                fig_mc.update_yaxes(
                    showgrid=True, gridwidth=1, gridcolor='rgba(48, 54, 61, 0.2)',
                    tickfont=dict(color="#8B949E", size=10),
                    side="right",
                    title_text=""
                )
                
                st.plotly_chart(fig_mc, width="stretch", key=f"mc_chart_{resolved_ticker}", theme="streamlit")
                
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
<td class="price-cell">{c_symbol}{row.quantile(0.10):.2f}</td>
<td class="price-cell">{c_symbol}{row.quantile(0.25):.2f}</td>
<td class="price-cell" style="font-weight: bold;">{c_symbol}{row.quantile(0.50):.2f}</td>
<td class="price-cell">{c_symbol}{row.quantile(0.75):.2f}</td>
<td class="price-cell">{c_symbol}{row.quantile(0.90):.2f}</td>
</tr>"""
                table_html += "</tbody></table>"
                st.markdown(table_html, unsafe_allow_html=True)
            else:
                st.warning("請先在「即時分析」分頁完成 AI 分析以生成預測數據。")
            
        with tab4:
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
                        # 根據市場動態調整搜尋關鍵字
                        is_tw = resolved_ticker.endswith(".TW") or resolved_ticker.endswith(".TWO") or resolved_ticker.isdigit()
                        
                        if is_tw:
                            search_keywords = [f"{clean_name} 股票", f"{clean_name} 營收", f"{clean_name} 財經"]
                            lang_param = "&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
                        else:
                            # 對於美股/加密貨幣，同時搜尋中英文新聞
                            search_keywords = [f"{clean_name} stock", f"{clean_name} earnings", f"{clean_name} news"]
                            lang_param = "&hl=en-US&gl=US&ceid=US:en"
                        
                        for query_text in search_keywords:
                            encoded_query = urllib.parse.quote(query_text)
                            rss_url = f"https://news.google.com/rss/search?q={encoded_query}{lang_param}"
                            
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
                        sentiment_text, sentiment_color = analyze_news_sentiment(item['title'], m_colors=m_colors)
                        
                        # 使用 CSS 類別簡化新聞 HTML
                        st.markdown(f"""<div class="news-item">
<div style="display: flex; justify-content: space-between; font-size: 0.75rem; margin-bottom: 8px;">
<div style="display: flex; gap: 8px; align-items: center;">
<span style="color: #58A6FF; font-weight: 700;">[{source_label}] {item['publisher']}</span>
<span style="background: {sentiment_color}20; color: {sentiment_color}; padding: 2px 8px; border-radius: 4px; border: 1px solid {sentiment_color}40; font-weight: 800; font-size: 0.65rem; text-transform: uppercase;">{sentiment_text}</span>
</div>
<span style="color: #8B949E; font-family: 'Roboto Mono', monospace;">{time_str}</span>
</div>
<a href="{item['link']}" target="_blank" style="text-decoration: none; color: #F0F6FC; font-size: 1.05rem; font-weight: 600; line-height: 1.5; display: block;">
{item['title']}
</a>
</div>""", unsafe_allow_html=True)
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