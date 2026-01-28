"""
================================================================================
PROJECT: Professional Quant Trading Terminal (Taiwan Stock Market)
VERSION: 2.0.0
AUTHOR: Quant Systems Team
LICENSE: Proprietary / Commercial Buyout
DESCRIPTION: 
    An integrated quantitative analysis system featuring AI-driven strategy 
    evolution and multi-factor scoring.
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
import requests

# --- SYSTEM CONFIGURATION ---
SYSTEM_SETTINGS = {
    "RISK_FREE_RATE": 0.015,       # Annual risk-free rate for Sharpe calculation
    "DEFAULT_BACKTEST_DAYS": 5,    # Holding period for backtest evaluation
    "AI_CONFIDENCE_THRESHOLD": 60, # Minimum score to trigger 'Strong Buy'
    "THEME_COLOR": "#58A6FF"       # Primary institutional blue
}

st.set_page_config(
    page_title="專業量化交易終端 | AI-Powered Analysis", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- SESSION STATE INITIALIZATION & SYNC ---
if 'ticker_input' not in st.session_state:
    st.session_state.ticker_input = "2330.TW"
if 'main_ticker_input' not in st.session_state:
    st.session_state.main_ticker_input = st.session_state.ticker_input

# Handle pending ticker updates from other pages (e.g., Sniper)
if 'pending_ticker_update' in st.session_state:
    new_ticker = st.session_state.pending_ticker_update
    st.session_state.ticker_input = new_ticker
    st.session_state.main_ticker_input = new_ticker
    del st.session_state.pending_ticker_update

# Custom CSS for Institutional Look
st.markdown("""
    <style>
    /* 現代化機構級深色主題變數 */
    :root {
        --bg-main: #0E1117;
        --bg-secondary: #161B22;
        --accent-blue: #58a6ff;
        --accent-gold: #D29922;
        --accent-up: #26a69a;
        --accent-down: #ef5350;
        --text-primary: #E6EDF3;
        --text-secondary: #8B949E;
        --border-color: rgba(48, 54, 61, 0.8);
        --glass-bg: rgba(22, 27, 34, 0.75);
    }

    /* 全局背景與字體 */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: radial-gradient(circle at top right, #1a1f2e, #0e1117) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', 'PingFang TC', -apple-system, sans-serif !important;
    }

    /* 隱藏預設元素並優化頂部空間 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {
        background-color: transparent !important;
    }
    /* 確保側邊欄切換按鈕可見並美化 */
    button[kind="header"] {
        color: var(--text-secondary) !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 50% !important;
        transition: all 0.3s ease !important;
    }
    button[kind="header"]:hover {
        color: var(--accent-blue) !important;
        background-color: rgba(88, 166, 255, 0.1) !important;
    }

    /* 側邊欄美化 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161B22 0%, #0D1117 100%) !important;
        border-right: 1px solid var(--border-color) !important;
        box-shadow: 5px 0 15px rgba(0,0,0,0.3) !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        background-color: transparent !important;
        padding-top: 2rem !important;
    }

    /* 側邊欄標題 */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: var(--accent-blue) !important;
        font-weight: 800 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        font-size: 1.1rem !important;
        margin-bottom: 1.5rem !important;
        padding-left: 0.5rem !important;
        border-left: 4px solid var(--accent-blue) !important;
    }

    /* 側邊欄按鈕美化 - 統一機構風格 */
    section[data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        border-radius: 10px !important;
        background: rgba(255, 255, 255, 0.02) !important;
        color: var(--text-primary) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        padding: 0.8rem 1.2rem !important;
        transition: all 0.25s ease !important;
        font-weight: 600 !important;
        margin-bottom: 10px !important;
        text-align: left !important;
        display: flex !important;
        align-items: center !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        border-color: var(--accent-blue) !important;
        background: rgba(88, 166, 255, 0.08) !important;
        color: var(--accent-blue) !important;
        transform: translateX(4px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
    }

    /* 股市狙擊手按鈕特殊色 */
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] div.stButton > button {
        border-color: rgba(210, 153, 34, 0.3) !important;
        color: var(--accent-gold) !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] div.stButton > button:hover {
        border-color: var(--accent-gold) !important;
        background: rgba(210, 153, 34, 0.1) !important;
    }

    /* 數據卡片 (Data Card) - 玻璃擬態 */
    .data-card {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        border: 1px solid var(--border-color) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
        margin-bottom: 24px !important;
        transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1) !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .data-card::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(88, 166, 255, 0.3), transparent);
    }

    .data-card:hover {
        border-color: rgba(88, 166, 255, 0.5) !important;
        transform: translateY(-5px) !important;
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.6) !important;
        background: rgba(22, 27, 34, 0.85) !important;
    }

    /* 指標卡片 (Metric Card) - 現代化玻璃質感 */
    .metric-card {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        text-align: center !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        height: 100% !important;
        min-height: 120px !important; /* 確保高度一致 */
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        position: relative !important;
        width: 100% !important; /* 強制填滿容器寬度 */
    }

    .metric-card:hover {
        background: rgba(88, 166, 255, 0.05) !important;
        border-color: var(--accent-blue) !important;
        transform: scale(1.02) !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4) !important;
    }

    .metric-label {
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        margin-bottom: 8px !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
    }

    .metric-value {
        color: var(--text-primary) !important;
        font-size: 2rem !important;
        font-weight: 900 !important;
        font-family: 'JetBrains Mono', 'Inter', monospace !important;
        line-height: 1.1 !important;
        background: linear-gradient(180deg, #FFFFFF 0%, #ADBAC7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* 資訊區塊 (Info Box) - 增強視覺層次 */
    .info-box {
        background: rgba(13, 17, 23, 0.4) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-left: 4px solid var(--accent-blue) !important;
        height: 100% !important;
        transition: all 0.3s ease !important;
    }

    .info-box:hover {
        background: rgba(13, 17, 23, 0.6) !important;
        border-color: rgba(88, 166, 255, 0.2) !important;
    }

    .info-header {
        font-size: 0.95rem !important;
        font-weight: 800 !important;
        color: var(--accent-blue) !important;
        margin-bottom: 12px !important;
        border-bottom: 1px solid rgba(88, 166, 255, 0.15) !important;
        padding-bottom: 8px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    .info-row {
        display: flex !important;
        justify-content: space-between !important;
        margin-bottom: 8px !important;
        font-size: 0.9rem !important;
    }

    .info-label { color: var(--text-secondary) !important; }
    .info-value { color: var(--text-primary) !important; font-weight: 600 !important; }

    /* 分頁 Tabs 美化 - 現代化導航 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px !important;
        background-color: transparent !important;
        border-bottom: 1px solid var(--border-color) !important;
        padding: 0 10px !important;
    }

    .stTabs [data-baseweb="tab"] {
        height: 54px !important;
        background-color: transparent !important;
        padding: 10px 4px !important;
        border: none !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.5px !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
        transform: translateY(-2px) !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--accent-blue) !important;
        background-color: transparent !important;
    }

    /* 自定義 Tab 下底線動畫 */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--accent-blue) !important;
        height: 3px !important;
        border-radius: 3px 3px 0 0 !important;
    }

    /* 新聞卡片清單 - 現代化卡片 */
    .news-item {
        padding: 16px 20px !important;
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        margin-bottom: 16px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 8px !important;
    }

    .news-item:hover {
        background: rgba(88, 166, 255, 0.04) !important;
        border-color: rgba(88, 166, 255, 0.2) !important;
        transform: translateX(6px) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
    }

    .news-item::before {
        content: "" !important;
        position: absolute !important;
        left: 0 !important;
        top: 0 !important;
        height: 100% !important;
        width: 0 !important;
        background: var(--accent-blue) !important;
        transition: width 0.3s ease !important;
        opacity: 0.8 !important;
    }

    .news-item:hover::before {
        width: 4px !important;
    }

    .news-title {
        color: var(--text-primary) !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        line-height: 1.4 !important;
        text-decoration: none !important;
    }

    .news-meta {
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        font-size: 0.8rem !important;
        color: var(--text-secondary) !important;
    }

    /* 滾動條美化 */
    ::-webkit-scrollbar {
        width: 8px !important;
        height: 8px !important;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-main) !important;
    }
    ::-webkit-scrollbar-thumb {
        background: #30363d !important;
        border-radius: 10px !important;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #484f58 !important;
    }

    /* 選項標籤 (Tags) */
    .scan-tag {
        display: inline-block !important;
        padding: 4px 10px !important;
        border-radius: 6px !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        margin-right: 6px !important;
        margin-bottom: 6px !important;
        background: rgba(88, 166, 255, 0.15) !important;
        color: var(--accent-blue) !important;
        border: 1px solid rgba(88, 166, 255, 0.3) !important;
    }

    /* 統計數字卡片 (Stat Card) - 機構級樣式 */
    .stat-card {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        height: 100% !important;
        box-shadow: 0 12px 40px rgba(0,0,0,0.5) !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .stat-header {
        display: flex !important;
        justify-content: space-between !important;
        align-items: flex-start !important;
        margin-bottom: 24px !important;
        border-bottom: 1px solid rgba(255,255,255,0.05) !important;
        padding-bottom: 16px !important;
    }

    .stat-title-group {
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
    }

    .stat-title-bar {
        width: 4px !important;
        height: 24px !important;
        border-radius: 2px !important;
    }

    .stat-title {
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        color: var(--text-primary) !important;
        margin: 0 !important;
    }

    .stat-badge {
        padding: 4px 12px !important;
        border-radius: 6px !important;
        font-size: 0.75rem !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    .rating-grid {
        display: grid !important;
        grid-template-columns: repeat(3, 1fr) !important;
        gap: 16px !important;
        margin-bottom: 24px !important;
    }

    .rating-item {
        background: rgba(255,255,255,0.03) !important;
        padding: 16px !important;
        border-radius: 12px !important;
        text-align: center !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
    }

    .rating-label {
        color: var(--text-secondary) !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        margin-bottom: 8px !important;
        text-transform: uppercase !important;
    }

    .rating-value {
        font-size: 1.6rem !important;
        font-weight: 900 !important;
        color: var(--accent-blue) !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    .rating-denominator {
        font-size: 0.8rem !important;
        color: var(--text-secondary) !important;
        margin-left: 2px !important;
    }

    .info-grid {
        display: grid !important;
        grid-template-columns: repeat(2, 1fr) !important;
        gap: 16px !important;
        margin-bottom: 16px !important;
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
    "tab_news": "📰 市場新聞",
    "tab_sniper": "🎯 股市狙擊手",
    "price_action": "價格走勢",
    "key_stats": "關鍵統計",
    "no_data": "查無此代碼，請確認格式是否正確 (例: 2330.TW 或 AAPL)。若代碼正確仍顯示此訊息，可能是 Yahoo Finance 暫時限制存取，請稍後再試。",
    "trailing_pe": "本益比",
    "forward_pe": "預測本益比",
    "div_yield": "殖利率",
    "high_52w": "52週高點",
    "low_52w": "52週低點",
    "beta": "Beta 係數",
    "tech_analysis": "技術指標詳解",
    "signal_label": "信號線",
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
        
    # 如果是 2330TW 這種格式 (沒加點)
    if re.match(r'^(\d{4,6})(TW|TWO)$', ticker):
        match = re.match(r'^(\d{4,6})(TW|TWO)$', ticker)
        return f"{match.group(1)}.{match.group(2)}"

    # 如果是 2330 TW 這種格式 (空格分隔)
    if re.match(r'^(\d{4,6})\s+(TW|TWO)$', ticker):
        match = re.match(r'^(\d{4,6})\s+(TW|TWO)$', ticker)
        return f"{match.group(1)}.{match.group(2)}"

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
        except Exception as e:
            st.sidebar.error(f"下載錯誤 ({t_code}): {str(e)}")
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

def get_signal_score(data, rsi_val, macd_val, signal_val, smc_data=None, entry_strategy=None, m_colors=None):
    """
    綜合信號評分系統 v4.0 (跨維度共識版)
    整合 技術指標 (Technical), SMC 結構 (Institutional), 與 AI 策略 (Quant)
    """
    # 預設顏色方案 (Default colors if m_colors not provided)
    if m_colors is None:
        m_colors = {
            "up": "#26a69a", "down": "#ef5350", "buy": "#26a69a", "sell": "#ef5350"
        }
    
    score_pts = 0
    current_price = float(data['Close'].iloc[-1])
    
    # 1. 技術面：中長期趨勢共振 (Technical Trend) - 基礎分
    ema20 = float(data['Close'].ewm(span=20).mean().iloc[-1])
    ema50 = float(data['Close'].ewm(span=50).mean().iloc[-1])
    ema200 = float(data['Close'].ewm(span=200).mean().iloc[-1])
    
    if ema20 > ema50 > ema200: score_pts += 3 # 多頭完美排列
    elif current_price > ema50: score_pts += 1 # 站上關鍵生命線
    
    # 2. 技術面：動能與強勢度 (Momentum)
    if macd_val > signal_val: score_pts += 1
    if 50 < rsi_val < 75: score_pts += 1 
    elif rsi_val < 35: score_pts += 1 # 極端超跌機會
    
    # 3. 技術面：量能支援 (Volume)
    v_ma20 = data['Volume'].rolling(20).mean().iloc[-1]
    if data['Volume'].iloc[-1] > v_ma20 * 1.3:
        if data['Close'].iloc[-1] > data['Open'].iloc[-1]: score_pts += 1
        
    # 4. SMC 結構修訂 (Institutional Correction) - 權重調整
    if smc_data:
        bias = smc_data.get('bias', '')
        zone = smc_data.get('zone', '')
        if "Bullish" in bias: score_pts += 2
        elif "Bearish" in bias: score_pts -= 2
        
        if "折價區" in zone: score_pts += 1
        elif "溢價區" in zone: score_pts -= 1
        
    # 5. AI 策略修訂 (Quant Correction) - 權重調整
    if entry_strategy:
        ai_action = entry_strategy.get('action', '')
        ai_score = entry_strategy.get('score', 50)
        
        if "買入" in ai_action or "試探" in ai_action:
            score_pts += (2 if ai_score > 70 else 1)
        elif "觀望" in ai_action or "減碼" in ai_action:
            score_pts -= (2 if ai_score > 70 else 1)

    # Mapping score to Rating (擴展閾值以適應多維度評分)
    if score_pts >= 7: return "核心看多 (共識)", m_colors["buy"], min(98, 80 + score_pts * 2)
    elif score_pts >= 4: return "趨勢偏多", m_colors["buy"], 75
    elif score_pts >= 1: return "震盪整理", "#787b86", 55
    elif score_pts <= -2: return "核心看空 (共識)", m_colors["sell"], 20
    elif score_pts < 0: return "趨勢偏空", m_colors["sell"], 35
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

def get_expert_insight(ticker, price, rsi, rating, macd_val, signal_val, buy_sigs, sell_sigs, current_date, smc_data=None, entry_strategy=None, m_colors=None):
    """
    生成專家診斷報告 (Generates Expert Technical Diagnosis Report) - 跨維度共識優化版
    綜合 技術指標 (Technical), SMC 結構 (Institutional), 與 AI 策略 (Quant)
    """
    # 1. 基礎分析 (Basic Analysis)
    rsi_status = "強勢區" if rsi > 60 else "超賣" if rsi < 30 else "中性"
    rsi_color = "#26a69a" if rsi > 60 else "#26a69a" if rsi < 30 else "#8B949E"
    rsi_desc = "RSI 進入 60 以上強勢區，波段動能正在釋放。" if rsi > 60 else "股價進入超賣區，可能存在波段築底機會。" if rsi < 30 else "RSI 處於中性區間，適合波段佈局。"
    
    macd_diff = macd_val - signal_val
    macd_status = "趨勢確認" if macd_diff > 0 else "趨勢轉弱"
    macd_color = "#26a69a" if macd_diff > 0 else "#ef5350"
    macd_desc = "MACD 柱狀體翻紅，波段多頭趨勢獲得確認。" if macd_diff > 0 else "MACD 動能放緩，波段可能進入整理期。"
    
    # 2. 跨維度共識檢查 (Cross-Dimension Consensus Check)
    # 技術面多頭判定
    tech_bullish = "波段強勢" in rating or "趨勢偏多" in rating
    # SMC 結構多頭判定
    smc_bullish = smc_data and ("Bullish" in smc_data.get('bias', '') or "折價區" in smc_data.get('zone', ''))
    smc_bearish = smc_data and ("Bearish" in smc_data.get('bias', '') or "溢價區" in smc_data.get('zone', ''))
    # AI 策略多頭判定
    ai_bullish = entry_strategy and ("買入" in entry_strategy.get('action', '') or "試探" in entry_strategy.get('action', ''))
    
    # 判斷是否存在矛盾 (Divergence Analysis)
    has_conflict = False
    conflict_desc = ""
    
    if tech_bullish and smc_bearish:
        has_conflict = True
        conflict_desc = "⚠️ 警告：技術指標偏多，但 SMC 顯示已進入溢價區，追高風險大，不建議在此買入。"
    elif not tech_bullish and smc_bullish:
        has_conflict = True
        conflict_desc = "⚠️ 提示：技術指標尚在整理，但價格已進入機構折價區，適合分批佈局底倉。"
    elif tech_bullish and not ai_bullish and entry_strategy and entry_strategy.get('score', 0) < 40:
        has_conflict = True
        conflict_desc = "⚠️ 注意：趨勢雖強但量化得分偏低，可能缺乏動能支持，建議減量參與。"

    # 3. 最終建議生成 (Final Consolidated Action)
    if has_conflict:
        action_status = "多空分歧"
        action_color = "#D29922"
        action_desc = conflict_desc
    else:
        # 達成共識
        if tech_bullish and (smc_bullish or not smc_data):
            action_status = "共振看多"
            action_color = "#26a69a"
            action_desc = "技術面與結構面達成看多共振，具備較高操作勝率。"
        elif not tech_bullish and smc_bearish:
            action_status = "空頭佔優"
            action_color = "#ef5350"
            action_desc = "技術指標走弱且處於溢價區，建議執行防禦性撤退。"
        else:
            action_status = "中性整理"
            action_color = "#58A6FF"
            action_desc = "目前各維度信號方向不一，建議保持輕倉並等待結構突破。"

    # 4. 即時信號覆蓋 (Signal Overrides)
    if current_date in buy_sigs:
        action_status = "🚀 波段買點"
        action_desc = "今日觸發關鍵【波段買入】訊號，結構全面轉強，建議進場。"
        action_color = "#26a69a"
    elif current_date in sell_sigs:
        action_status = "🔻 波段賣點"
        action_desc = "今日觸發關鍵【波段賣出】訊號，趨勢出現反轉跡象，建議撤退。"
        action_color = "#ef5350"

    # 5. 波段具體建議 (Swing Specific Advice)
    if rsi > 75:
        swing_advice = "短線指標極度超漲，波段持有者建議在此減碼 1/2，落袋為安。"
    elif rsi < 25:
        swing_advice = "短線指標極度超跌，具備強勁反彈潛力，可在此建立波段首筆倉位。"
    else:
        swing_advice = "建議守住 EMA50 關鍵水位，只要結構不破，波段趨勢即未結束。"

    return {
        "rsi": {"val": f"{rsi:.1f}", "status": rsi_status, "color": rsi_color, "desc": rsi_desc},
        "macd": {"val": f"{macd_diff:+.2f}", "status": macd_status, "color": macd_color, "desc": macd_desc},
        "action": {"status": action_status, "color": action_color, "desc": action_desc},
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
        dragmode=False,
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
    - 適應性學習：自動根據個股歷史表現調整分析權重
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

def get_ai_entry_strategy(data, rsi, ema20, ema50, bb_lower, health_scores, vr, atr, dynamic_weights=None, m_colors=None, is_discount=False, has_fvg=False, is_squeeze=False, is_unicorn=False):
    """
    Generates AI-driven entry strategy (Swing Trading Optimized v3.6 - SMC Integrated)
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
    
    # 權重分配：技術面(40%)、基本面(30%)、SMC/動能(30%)
    w_tech, w_fund, w_smc = dynamic_weights if dynamic_weights else (0.40, 0.30, 0.30)
    
    # 1. Technical Score (0-100)
    t_raw = 0
    ema200 = float(data['Close'].ewm(span=200).mean().iloc[-1])
    if current_price > ema50 > ema200: t_raw += 60 
    if 40 < rsi < 65: t_raw += 40
    
    # 2. Fundamental Score (0-100)
    f_raw = (p_score + l_score + c_score + g_score + q_score) * 2 # 轉為百分制
    
    # 3. SMC & Volume Score (0-100)
    s_raw = 0
    if is_discount: s_raw += 25  # 處於折價區 (Discount Zone)
    if has_fvg: s_raw += 20      # 有看漲 FVG 缺口
    if is_squeeze: s_raw += 15    # 處於擠壓狀態 (潛在爆發)
    if is_unicorn: s_raw += 40    # Unicorn 強力買入訊號 (OB + FVG)
    
    # 如果沒有 SMC 訊號，則參考成交量
    if s_raw == 0:
        if vr > 130: s_raw += 40
        elif vr > 100: s_raw += 20
        vol_ma20 = data['Volume'].rolling(20).mean().iloc[-1]
        if data['Volume'].iloc[-1] > vol_ma20 * 1.5: s_raw += 30
        price_change_3d = (current_price - data['Close'].iloc[-4]) / data['Close'].iloc[-4]
        if price_change_3d > 0.03: s_raw += 30 
    
    # 綜合評分
    total_score = (min(100, t_raw) * w_tech) + (min(100, f_raw) * w_fund) + (min(100, s_raw) * w_smc)
    
    # 策略生成
    if total_score > 70:
        action, color = "積極買入", "#26a69a"
        desc = "SMC 多頭共振且趨勢強勁，具備極高爆發潛力。"
        if is_unicorn: desc = "【Unicorn 訊號】OB + FVG 強力支撐，極佳入場點。"
        suggested_price = current_price
        target_price = current_price + (3.5 * atr)
    elif total_score > 50:
        action, color = "建議試探", "#26a69a"
        desc = "進入折價區或有 FVG 支撐，建議分批佈局。"
        suggested_price = ema20
        target_price = current_price + (2.5 * atr)
    elif total_score > 30:
        action, color = "少量參與", "#D29922"
        desc = "目前信號中性，SMC 結構尚在成形中。"
        suggested_price = ema20 * 0.98
        target_price = current_price + (1.5 * atr)
    else:
        action, color = "保守觀望", "#8B949E"
        desc = "處於溢價區或趨勢不明，建議等待回調至折價區。"
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
        "score": total_score,
        "is_smc_buy": is_discount or has_fvg or is_unicorn
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

def get_ai_exit_strategy(data, rsi, bb_upper, target_median, atr, health_scores=None, m_colors=None):
    """
    Generates AI-driven exit strategy (Dynamic Profit Taking v3.5)
    """
    if m_colors is None:
        m_colors = {"up": "#26a69a", "down": "#ef5350", "buy": "#26a69a", "sell": "#ef5350"}
    
    current_price = float(data['Close'].iloc[-1])
    base_target = target_median if target_median and target_median > current_price else bb_upper
    
    # 1. 退出壓力評分 (Exit Pressure Score)
    exit_score = 0
    if rsi > 80: exit_score += 55
    elif rsi > 70: exit_score += 35
    
    if current_price > bb_upper: exit_score += 25
    
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

def calculate_quant_factors(data, ticker_metadata, rsi_series, atr_series, financial_data=None):
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
        trend_score = 70 # 原為 65，微調以補償 SuperTrend
    elif close_prices.iloc[-1] > ema200.iloc[-1]:
        trend_score = 55 # 原為 50
    else:
        trend_score = 30
    
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

@st.cache_data(ttl=3600)
def analyze_smc(data):
    """
    Smart Money Concepts (SMC) 分析核心
    計算 BOS, CHoCH, OB, FVG 與市場偏向 (Daily Bias)
    """
    if data is None or len(data) < 30:
        return None

    df = data.copy()
    
    # 1. 識別分型 (Fractals / Swing Highs & Lows)
    df['SwingHigh'] = (df['High'] > df['High'].shift(1)) & (df['High'] > df['High'].shift(2)) & \
                      (df['High'] > df['High'].shift(-1)) & (df['High'] > df['High'].shift(-2))
    df['SwingLow'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'] < df['Low'].shift(2)) & \
                     (df['Low'] < df['Low'].shift(-1)) & (df['Low'] < df['Low'].shift(-2))

    # 2. 市場結構 (BOS & CHoCH)
    bos_events = []
    choch_events = []
    
    last_high_idx = None
    last_low_idx = None
    
    # 掃描歷史結構
    for i in range(5, len(df)):
        # 尋找最近的 Swing High/Low
        highs = df.iloc[:i][df.iloc[:i]['SwingHigh']]
        lows = df.iloc[:i][df.iloc[:i]['SwingLow']]
        
        if not highs.empty:
            l_high = highs['High'].iloc[-1]
            if df['Close'].iloc[i] > l_high:
                # 判斷是 BOS 還是 CHoCH (簡單邏輯：如果之前趨勢相反則是 CHoCH)
                if len(bos_events) > 0 and bos_events[-1]['type'] == 'Bearish BOS':
                    choch_events.append({'date': df.index[i], 'price': l_high, 'type': 'Bullish CHoCH'})
                else:
                    bos_events.append({'date': df.index[i], 'price': l_high, 'type': 'Bullish BOS'})
        
        if not lows.empty:
            l_low = lows['Low'].iloc[-1]
            if df['Close'].iloc[i] < l_low:
                if len(bos_events) > 0 and bos_events[-1]['type'] == 'Bullish BOS':
                    choch_events.append({'date': df.index[i], 'price': l_low, 'type': 'Bearish CHoCH'})
                else:
                    bos_events.append({'date': df.index[i], 'price': l_low, 'type': 'Bearish BOS'})

    # 3. 溢價/折價區間 (Premium/Discount)
    lookback = 40
    range_high = df['High'].rolling(window=lookback).max().iloc[-1]
    range_low = df['Low'].rolling(window=lookback).min().iloc[-1]
    equilibrium = (range_high + range_low) / 2
    
    current_close = df['Close'].iloc[-1]
    zone = "均衡區 (Equilibrium)"
    if current_close < equilibrium:
        zone = "折價區 (Discount - 買入區)"
    elif current_close > equilibrium:
        zone = "溢價區 (Premium - 賣出區)"
    
    # 4. 訂單塊 (Order Blocks)
    all_obs = []
    # 掃描過去 100 根 K 線尋找 OB
    for i in range(len(df)-100, len(df)-2):
        if i < 0: continue
        # Bullish OB
        if df['Close'].iloc[i] < df['Open'].iloc[i] and df['Close'].iloc[i+1] > df['High'].iloc[i]:
            all_obs.append({'type': 'Bullish OB', 'top': df['High'].iloc[i], 'bottom': df['Low'].iloc[i], 'start': df.index[i], 'end': df.index[-1]})
        # Bearish OB
        if df['Close'].iloc[i] > df['Open'].iloc[i] and df['Close'].iloc[i+1] < df['Low'].iloc[i]:
            all_obs.append({'type': 'Bearish OB', 'top': df['High'].iloc[i], 'bottom': df['Low'].iloc[i], 'start': df.index[i], 'end': df.index[-1]})

    # 5. 公允價值缺口 (FVG)
    all_fvgs = []
    for i in range(len(df)-60, len(df)-1):
        if i < 2: continue
        if df['Low'].iloc[i] > df['High'].iloc[i-2]:
            all_fvgs.append({'type': 'Bullish FVG', 'top': df['Low'].iloc[i], 'bottom': df['High'].iloc[i-2], 'start': df.index[i-1], 'end': df.index[-1]})
        elif df['High'].iloc[i] < df['Low'].iloc[i-2]:
            all_fvgs.append({'type': 'Bearish FVG', 'top': df['Low'].iloc[i-2], 'bottom': df['High'].iloc[i], 'start': df.index[i-1], 'end': df.index[-1]})

    # 6. 當前狀態
    structure = "中性 (Neutral)"
    if bos_events:
        structure = bos_events[-1]['type']
    if choch_events:
        structure = choch_events[-1]['type']

    bias = "中性"
    if "Bullish" in structure:
        bias = "看多 (Bullish)"
    elif "Bearish" in structure:
        bias = "看空 (Bearish)"

    # 獲取最新的看多與看空 OB
    bull_obs = [o for o in all_obs if o['type'] == 'Bullish OB']
    bear_obs = [o for o in all_obs if o['type'] == 'Bearish OB']

    return {
        "structure": structure,
        "bias": bias,
        "zone": zone,
        "equilibrium": equilibrium,
        "range": (range_low, range_high),
        "bos_events": bos_events[-5:], # 傳回最近 5 個
        "choch_events": choch_events[-5:],
        "obs": all_obs[-3:], # 傳回最近 3 個
        "fvgs": all_fvgs[-3:],
        "bull_ob": (bull_obs[-1]['bottom'], bull_obs[-1]['top']) if bull_obs else None,
        "bear_ob": (bear_obs[-1]['bottom'], bear_obs[-1]['top']) if bear_obs else None,
        "fvgs_list": all_fvgs[-3:],
        "price": current_close
    }

@st.cache_data(ttl=86400) # 24 小時更新一次
def get_yahoo_hot_tickers():
    """
    從 Yahoo Finance 獲取每日熱門趨勢標的與強勢族群
    """
    hot_tickers = []
    
    # 1. 獲取 Yahoo Trending (TW)
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/trending/TW"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            quotes = data.get('finance', {}).get('result', [{}])[0].get('quotes', [])
            for q in quotes:
                symbol = q.get('symbol')
                if symbol:
                    hot_tickers.append(symbol)
    except:
        pass

    # 2. 加入預定義的強勢產業領頭羊 (做為備選與擴充)
    sectors = {
        "AI/Server": ["2330.TW", "2317.TW", "2382.TW", "3231.TW", "6669.TW", "2376.TW", "3017.TW", "3661.TW"],
        "Semiconductor": ["2454.TW", "2303.TW", "3711.TW", "2379.TW", "3034.TW", "2337.TW", "6770.TW"],
        "Shipping": ["2603.TW", "2609.TW", "2615.TW", "2605.TW"],
        "Energy/EV": ["1513.TW", "1519.TW", "1503.TW", "1605.TW", "1608.TW"]
    }
    
    # 隨機選取部分產業龍頭加入掃描，增加多樣性
    import random
    for sector, symbols in sectors.items():
        hot_tickers.extend(random.sample(symbols, min(len(symbols), 3)))
        
    # 去重並過濾
    final_list = list(dict.fromkeys(hot_tickers))
    return ", ".join(final_list[:25]) # 最多回傳 25 檔

def run_sniper_scan(ticker_list):
    """
    股市狙擊手掃描 (Stock Sniper Scan)
    掃描多檔股票以尋找高勝率進場機會 (Unicorn, Squeeze, Breakouts)
    """
    results = []
    failed_tickers = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 預處理 ticker 列表，過濾重複、空值與雜訊
    noise = ['TW', 'TWO', 'T', 'O', 'STOCK']
    unique_tickers = [t.strip().upper() for t in ticker_list if t.strip()]
    unique_tickers = [t for t in unique_tickers if t not in noise]
    unique_tickers = list(dict.fromkeys(unique_tickers))
    
    for i, ticker in enumerate(unique_tickers):
        status_text.text(f"正在狙擊: {ticker} ({i+1}/{len(unique_tickers)})")
        progress_bar.progress((i + 1) / len(unique_tickers))
        
        try:
            # 獲取 60 天數據進行分析
            data, resolved = get_stock_data(ticker, period="60d", interval="1d")
            if data is None or len(data) < 30:
                failed_tickers.append(ticker)
                continue
                
            # 1. 基礎數據
            current_price = float(data['Close'].iloc[-1])
            prev_price = float(data['Close'].iloc[-2])
            change_pct = (current_price / prev_price - 1) * 100
            
            # 2. SMC 邏輯 (Enhanced with analyze_smc)
            smc_info = analyze_smc(data)
            has_smc = smc_info is not None
            
            # Unicorn 邏輯整合
            bull_ob_price = 0
            if has_smc and smc_info['bull_ob']:
                bull_ob_price = smc_info['bull_ob'][0]
            
            # 檢查 FVG 是否存在於最近 5 根 K 線
            has_recent_fvg = False
            if has_smc:
                has_recent_fvg = any(f[0] == "Bullish" for f in smc_info['fvgs'])
            
            # Unicorn Setup: Price in OB + Recent FVG
            has_unicorn = (data['Low'].iloc[-1] <= bull_ob_price * 1.01) and has_recent_fvg if bull_ob_price > 0 else False
            
            # 3. Squeeze & Volume
            ma20 = data['Close'].rolling(window=20).mean()
            ma5 = data['Close'].rolling(window=5).mean()
            ma60 = data['Close'].rolling(window=60).mean()
            std20 = data['Close'].rolling(window=20).std()
            bb_upper = ma20 + (2 * std20)
            bb_lower = ma20 - (2 * std20)
            bb_width = (bb_upper - bb_lower) / ma20
            
            # Squeeze logic
            is_squeeze = bb_width.iloc[-1] < bb_width.rolling(window=100).quantile(0.2).iloc[-1]
            # Squeeze Breakout: Price breaks above BB upper while squeeze is ending or just ended
            is_squeeze_breakout = (current_price > bb_upper.iloc[-1]) and (bb_width.iloc[-1] > bb_width.iloc[-2]) and (bb_width.iloc[-2] < bb_width.rolling(window=100).quantile(0.2).iloc[-2])

            # Volume Spike
            avg_vol = data['Volume'].rolling(window=20).mean()
            vol_spike = data['Volume'].iloc[-1] > avg_vol.iloc[-1] * 2

            # MA Alignment (多頭排列)
            ma_alignment = ma5.iloc[-1] > ma20.iloc[-1] > ma60.iloc[-1] if not ma60.isnull().all() else False
            
            # 4. 指標
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs.iloc[-1]))
            
            # 5. 評分與標籤
            tags = []
            score = 0
            
            # SMC Tags
            if has_smc:
                if smc_info['bias'] == "Bullish":
                    tags.append("🐂 Bias:Bull")
                    score += 15
                if "CHoCH" in smc_info['structure']:
                    tags.append("⚡ CHoCH")
                    score += 20
                if "BOS" in smc_info['structure']:
                    tags.append("🔗 BOS")
                    score += 10
                if smc_info['zone'].startswith("Discount"):
                    tags.append("💎 Discount")
                    score += 15
                if has_recent_fvg:
                    tags.append("🚀 GAP")
                    score += 10

            if has_unicorn: 
                tags.append("🦄 Unicorn")
                score += 30 # OB + FVG confluence
            if is_squeeze: 
                tags.append("🌀 Squeeze")
                score += 20
            if is_squeeze_breakout:
                tags.append("💥 SqzBreak")
                score += 30
            if vol_spike:
                tags.append("📊 VolSpike")
                score += 25
            if ma_alignment:
                tags.append("📈 BullTrend")
                score += 15
            if change_pct > 3: 
                tags.append("🚀 Breakout")
                score += 10
            if 30 < rsi < 45: 
                tags.append("📉 Reversal?")
                score += 15

            
            results.append({
                "代碼": resolved,
                "現價": f"{current_price:.2f}",
                "漲跌": f"{change_pct:+.2f}%",
                "RSI": f"{rsi:.1f}",
                "訊號標籤": " ".join(tags),
                "狙擊分數": score
            })
            
        except Exception:
            continue
            
    progress_bar.empty()
    status_text.empty()
    
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        return df_results.sort_values(by="狙擊分數", ascending=False), failed_tickers
    return df_results, failed_tickers

# Sidebar settings
with st.sidebar:
    st.markdown(f'<h2 style="font-size: 1.2rem; color: #FFFFFF; margin-bottom: 20px;">{t["settings"]}</h2>', unsafe_allow_html=True)
    
    # Actual Ticker Input
    ticker_input = st.text_input(
        t["ticker_label"], 
        key="main_ticker_input", 
        help="台股請加後綴，例如：2330.TW (上市) 或 8069.TWO (上櫃)；美股直接輸入代碼，例如：AAPL"
    )
    if ticker_input != st.session_state.ticker_input:
        st.session_state.ticker_input = ticker_input
        st.rerun()

    st.markdown("---")
    period = st.selectbox(t["period_label"], options=["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    interval = st.selectbox(t["interval_label"], options=["1d", "1wk", "1mo"], index=0)

    st.markdown("---")
    # 股市狙擊手按鈕 (Stock Sniper Button)
    if st.button("🎯 股市狙擊手", use_container_width=True, key="side_sniper_btn"):
        st.session_state.active_page = "sniper"
        st.rerun()
    
    # 如果在狙擊手頁面，顯示返回按鈕
    if st.session_state.get('active_page') == "sniper":
        if st.button("🔙 返回技術分析", use_container_width=True, key="side_back_btn"):
            st.session_state.active_page = "main"
            st.rerun()

# Header Section
# 頁面路由：如果 active_page 為 sniper，顯示狙擊手內容
if st.session_state.get('active_page') == "sniper":
    st.markdown(f'''<div class="data-card" style="border-left: 4px solid #D29922; background: rgba(210, 153, 34, 0.05); margin-bottom: 24px;">
    <h3 style="margin: 0; color: #D29922;">🎯 股市狙擊手 (Stock Sniper)</h3>
    <p style="font-size: 0.85rem; color: #8B949E; margin-top: 8px;">
        學習自 StockSniper 與 DailyDip 概念，自動掃描市場中的高勝率機會。<br>
        檢測項目：<b>Unicorn (OB+FVG)</b>、<b>Squeeze (動能擠壓)</b>、<b>Breakout (帶量突破)</b>。
    </p>
    </div>''', unsafe_allow_html=True)
    
    # 狙擊清單設定
    col_list, col_presets = st.columns([2, 1])
    
    with col_presets:
        preset = st.selectbox("🎯 快速選擇狙擊清單", options=[
            "自定義清單", 
            "🔥 Yahoo 每日熱門趨勢",
            "台灣 50 指數 (0050)", 
            "AI 概念股精選", 
            "半導體供應鏈", 
            "高股息熱門股"
        ], key="sniper_preset_select")
        
        presets = {
            "🔥 Yahoo 每日熱門趨勢": "DYNAMIC", # 標記為動態獲取
            "台灣 50 指數 (0050)": "2330.TW, 2317.TW, 2454.TW, 2308.TW, 2303.TW, 2382.TW, 3231.TW, 2412.TW, 2881.TW, 2882.TW, 3008.TW, 2603.TW, 2357.TW, 3711.TW, 2408.TW, 2379.TW, 1301.TW, 1303.TW, 2886.TW, 2891.TW",
            "AI 概念股精選": "2330.TW, 2317.TW, 2382.TW, 3231.TW, 6669.TW, 2376.TW, 2454.TW, 3017.TW, 3661.TW, 3443.TW, 2059.TW, 8210.TW",
            "半導體供應鏈": "2330.TW, 2454.TW, 2379.TW, 3034.TW, 2337.TW, 3711.TW, 6770.TW, 3532.TW, 6182.TWO, 8069.TWO",
            "高股息熱門股": "2317.TW, 2412.TW, 2881.TW, 2882.TW, 2886.TW, 2891.TW, 2892.TW, 2885.TW, 1101.TW, 2105.TW"
        }
        
        # 如果選擇了預設清單，直接更新 text_area 的 state
        if preset != "自定義清單":
            if presets[preset] == "DYNAMIC":
                with st.spinner("正在從 Yahoo Finance 抓取每日熱門標的..."):
                    st.session_state.sniper_input_area = get_yahoo_hot_tickers()
            else:
                st.session_state.sniper_input_area = presets[preset]
    
    with col_list:
        default_sniper_list = "2330.TW, 2317.TW, 2454.TW, 2308.TW, 2303.TW, 2382.TW, 3231.TW, 2412.TW, 2881.TW, 2882.TW, 3008.TW, 2603.TW, 1513.TW, 1519.TW, 2376.TW"
        # 確保 sniper_input_area 在 session_state 中初始化
        if 'sniper_input_area' not in st.session_state:
            st.session_state.sniper_input_area = default_sniper_list
        
        sniper_input = st.text_area("狙擊掃描清單 (逗號分隔)", height=100, key="sniper_input_area")
    
    col_scan, col_clear, col_spacer = st.columns([1, 1, 3])
    with col_scan:
        start_scan = st.button("🚀 開始狙擊掃描", type="primary", use_container_width=True)
    with col_clear:
        if st.button("🗑️ 清空列表", use_container_width=True):
            st.session_state.sniper_input_area = ""
            st.rerun()
    
    if start_scan:
        # 使用正則表達式拆分：支援逗號、空格、換行
        import re
        raw_list = re.split(r'[,\s\n]+', sniper_input.strip())
        ticker_list = [t.strip().upper() for t in raw_list if t.strip()]
        
        if not ticker_list:
            st.error("請輸入至少一個股票代碼。")
        else:
            scan_results, failed_list = run_sniper_scan(ticker_list)
            st.session_state.scan_results = scan_results
            st.session_state.failed_scan_list = failed_list
            st.session_state.has_scanned = True
    
    if st.session_state.get('has_scanned', False) and 'scan_results' in st.session_state:
        scan_results = st.session_state.scan_results
        failed_list = st.session_state.get('failed_scan_list', [])
        
        if failed_list:
            with st.expander(f"⚠️ 無法獲取數據的標的 ({len(failed_list)})"):
                st.write(", ".join(failed_list))
                st.info("請檢查代碼格式是否正確（台股需加 .TW 或 .TWO）或該股票是否已下市。")

        if not scan_results.empty:
            st.markdown("### 🔍 狙擊掃描結果")
            
            # Summary Metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("總掃描標的", len(scan_results))
            with m2:
                high_score_count = len(scan_results[scan_results['狙擊分數'] >= 60])
                st.metric("強烈訊號 (>=60)", high_score_count)
            with m3:
                avg_score = scan_results['狙擊分數'].mean()
                st.metric("平均狙擊分數", f"{avg_score:.1f}")
            with m4:
                best_ticker = scan_results.iloc[0]['代碼']
                st.metric("本日最優選", best_ticker)

            def color_score(val):
                color = "#ef5350" if val < 30 else "#D29922" if val < 60 else "#26a69a"
                return f'color: {color}; font-weight: bold; background: {color}10'

            st.dataframe(
                scan_results.style.map(color_score, subset=['狙擊分數']),
                use_container_width=True,
                height=500
            )
            
            # Top picks with more detail
            st.markdown("#### 🎯 核心狙擊推薦")
            top_picks = scan_results.head(3)
            cols = st.columns(len(top_picks))
            for idx, (i, row) in enumerate(top_picks.iterrows()):
                with cols[idx]:
                    st.markdown(f'''
                    <div style="padding: 20px; background: rgba(88, 166, 255, 0.05); border: 1px solid rgba(88, 166, 255, 0.2); border-radius: 12px; text-align: center;">
                        <div style="font-size: 0.8rem; color: #8B949E; margin-bottom: 5px;">TOP {idx+1}</div>
                        <div style="font-size: 1.5rem; font-weight: 800; color: #58A6FF;">{row['代碼']}</div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: #D29922; margin: 10px 0;">得分: {row['狙擊分數']}</div>
                        <div style="font-size: 0.8rem; color: #E0E0E0;">{row['訊號標籤']}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                    if st.button(f"查看 {row['代碼']} 詳情", key=f"detail_{row['代碼']}"):
                        st.session_state.pending_ticker_update = row['代碼']
                        st.session_state.active_page = "main"
                        st.rerun()

            csv = scan_results.to_csv(index=False).encode('utf-8-sig')
            st.download_button(label="📥 下載完整狙擊報表 (CSV)", data=csv, file_name=f"sniper_report_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv", use_container_width=True)
        else:
            st.warning("目前掃描範圍內無明顯狙擊信號，請更換掃描清單。")
    
    with st.expander("📖 了解狙擊策略 (Strategy Details)"):
        st.markdown("""
        ### 🎯 核心狙擊邏輯說明
        本系統整合了 **Smart Money Concepts (SMC)** 與傳統量化指標，旨在捕捉市場中的「機構足跡」與「動能爆發」。
        
        #### 1. 🦄 Unicorn (OB + FVG) - 權重: 40
        - **原理**：尋找價格回測「訂單塊 (Order Block)」且伴隨「公允價值缺口 (FVG)」的現象。這通常是機構投資者進場留下的痕跡。
        - **條件**：當前價格觸及 OB 區間，且近期出現過強勢突破缺口。
        
        #### 2. 💥 Squeeze Breakout - 權重: 30
        - **原理**：當布林通道極度擠壓後（波動率極低），價格帶量向上突破通道上軌。
        - **條件**：BB Width 處於歷史低位後開始擴張，且價格 > BB Upper。
        
        #### 3. 📊 Volume Spike - 權重: 25
        - **原理**：成交量是價格的先行指標。異常放量通常代表大戶進場或趨勢反轉。
        - **條件**：當日成交量 > 20日平均成交量 2 倍。
        
        #### 4. 📈 Bullish MA Alignment - 權重: 15
        - **原理**：短、中、長期均線呈多頭排列，代表趨勢強勁。
        - **條件**：MA5 > MA20 > MA60。
        
        #### 5. 🌀 Squeeze (動能擠壓) - 權重: 20
        - **原理**：波動率收斂中，代表即將發生大行情。
        - **條件**：BB Width < 過去 100 天的 20% 分位數。
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #8B949E; font-size: 0.7rem; padding: 20px;">'
        'INSTITUTIONAL TERMINAL v2.0 | REAL-TIME DATA VIA YAHOO FINANCE<br>'
        '© 2026 Financial Analytics Group. All rights reserved. For professional use only.'
        '</div>', 
        unsafe_allow_html=True
    )
    st.stop() # 停止執行後續的 UI 渲染 (Header, Metrics, etc.)

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

        # --- SMC (Smart Money Concepts) & Advanced Indicators ---
        # 1. Discount/Premium Zones (SMC)
        lookback_smc = 20
        recent_high = data['High'].rolling(window=lookback_smc).max()
        recent_low = data['Low'].rolling(window=lookback_smc).min()
        equilibrium = (recent_high + recent_low) / 2
        
        # 2. FVG (Fair Value Gap) Detection
        def detect_fvg(df):
            bull_fvg = (df['Low'].shift(-1) > df['High'].shift(1)) & (df['Close'] > df['Open'])
            bear_fvg = (df['High'].shift(-1) < df['Low'].shift(1)) & (df['Close'] < df['Open'])
            return bull_fvg, bear_fvg
        bull_fvg, bear_fvg = detect_fvg(data)
        
        # 3. Squeeze Detection (BB Width)
        bb_width = (bb_upper - bb_lower) / ma20
        is_squeeze = bb_width < bb_width.rolling(window=100).quantile(0.2) # 低於 20% 分位數視為擠壓

        # 4. Order Block (OB) Detection
        def detect_order_blocks(df, lookback=20):
            # 簡化的 OB 檢測：尋找一段強勢波動前的最後一根反向 K 線
            bull_ob_price = pd.Series(index=df.index, dtype=float)
            bear_ob_price = pd.Series(index=df.index, dtype=float)
            
            for i in range(lookback, len(df)-1):
                # Bullish OB: 強勢上漲 (例如 3 根 K 線漲幅 > ATR) 前的最後一根陰線
                if df['Close'].iloc[i+1] > df['High'].iloc[i] and (df['Close'].iloc[i+3] if i+3 < len(df) else df['Close'].iloc[-1]) > df['Close'].iloc[i] * 1.03:
                    if df['Close'].iloc[i] < df['Open'].iloc[i]:
                        bull_ob_price.iloc[i:] = df['Low'].iloc[i]
                
                # Bearish OB: 強勢下跌前的最後一根陽線
                if df['Close'].iloc[i+1] < df['Low'].iloc[i] and (df['Close'].iloc[i+3] if i+3 < len(df) else df['Close'].iloc[-1]) < df['Close'].iloc[i] * 0.97:
                    if df['Close'].iloc[i] > df['Open'].iloc[i]:
                        bear_ob_price.iloc[i:] = df['High'].iloc[i]
            
            return bull_ob_price, bear_ob_price
        
        bull_ob, bear_ob = detect_order_blocks(data)

        # 5. Unicorn Signal (OB + FVG Confirmation)
        has_unicorn_buy = (data['Low'] <= bull_ob) & (bull_fvg.rolling(window=5).sum() > 0)
        has_unicorn_sell = (data['High'] >= bear_ob) & (bear_fvg.rolling(window=5).sum() > 0)

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
            data=data, 
            rsi=float(rsi_series.iloc[-1]) if not rsi_series.empty else 50, 
            ema20=float(ema20.iloc[-1]) if not ema20.empty else current_price, 
            ema50=float(ema50.iloc[-1]) if not ema50.empty else current_price, 
            bb_lower=float(bb_lower.iloc[-1]) if not bb_lower.empty else current_price * 0.95, 
            health_scores=health_scores, 
            vr=float(vr_series.iloc[-1]) if not vr_series.empty else 100, 
            atr=float(atr_series.iloc[-1]) if not atr_series.empty else 0, 
            dynamic_weights=best_weights,
            is_discount=bool(data['Close'].iloc[-1] < equilibrium.iloc[-1]),
            has_fvg=bool(bull_fvg.iloc[-1]),
            is_squeeze=bool(is_squeeze.iloc[-1]),
            is_unicorn=bool(has_unicorn_buy.iloc[-1])
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
            
            # --- SMC 專業策略補充 (SMC Professional Strategy Supplement) ---
            if has_unicorn_buy.iloc[i]:
                inst_strats.append({"name": "Unicorn 買入", "desc": "SMC: Order Block + FVG 共振，極高勝率進場結構。"})
            elif has_unicorn_sell.iloc[i]:
                inst_strats.append({"name": "Unicorn 賣出", "desc": "SMC: Bearish OB + FVG 共振，機構出貨結構。"})
            
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
                        data=data.iloc[:i+1], rsi=h_rsi, ema20=h_ema20, ema50=h_ema50, bb_lower=h_bb_lower, 
                        health_scores=health_scores, vr=h_vr, atr=h_atr, 
                        dynamic_weights=best_weights, 
                        m_colors=m_colors,
                        is_discount=bool(data['Close'].iloc[i] < equilibrium.iloc[i]),
                        has_fvg=bool(bull_fvg.iloc[i]),
                        is_squeeze=bool(is_squeeze.iloc[i]),
                        is_unicorn=bool(has_unicorn_buy.iloc[i])
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
        
        # 頂部關鍵指標顯示 (Top Key Metrics Display) - Extravagant Grid Version
        prev_price = float(data['Close'].iloc[-2])
        change = current_price - prev_price
        change_pct = (change/prev_price*100)
        delta_color = m_colors["up"] if change >= 0 else m_colors["down"]
        delta_bg = f"{delta_color}22"
        
        volume = float(data['Volume'].iloc[-1])
        market_cap = ticker_metadata.get('marketCap', 'N/A')
        market_cap_val = f"{market_cap/1e12:.2f}T" if isinstance(market_cap, (int, float)) else "N/A"
        day_high = ticker_metadata.get('dayHigh', 'N/A')
        day_high_val = f"{day_high}" if isinstance(day_high, (int, float)) else "N/A"

        st.markdown(f'''
             <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-bottom: 24px;">
                 <div class="metric-card" style="flex-direction: row !important; justify-content: space-between !important; align-items: center !important; padding: 10px 14px !important; min-height: 45px !important; white-space: nowrap !important;">
                     <div class="metric-label" style="margin: 0 !important; font-size: 0.75rem !important; opacity: 0.8;">{t["current_price"]}</div>
                     <div class="metric-value" style="font-size: 1.1rem !important; margin: 0 !important; font-weight: 700;">{current_price:.2f}</div>
                 </div>
                 <div class="metric-card" style="flex-direction: row !important; justify-content: space-between !important; align-items: center !important; padding: 10px 14px !important; min-height: 45px !important; white-space: nowrap !important;">
                     <div class="metric-label" style="margin: 0 !important; font-size: 0.75rem !important; opacity: 0.8;">{t["change"]}</div>
                     <div class="metric-value" style="color: {delta_color}; background: none; -webkit-text-fill-color: {delta_color}; font-size: 1.1rem !important; margin: 0 !important; font-weight: 700;">{change:+.2f}({change_pct:+.2f}%)</div>
                 </div>
                 <div class="metric-card" style="flex-direction: row !important; justify-content: space-between !important; align-items: center !important; padding: 10px 14px !important; min-height: 45px !important; white-space: nowrap !important;">
                     <div class="metric-label" style="margin: 0 !important; font-size: 0.75rem !important; opacity: 0.8;">{t["volume"]}</div>
                     <div class="metric-value" style="font-size: 1.1rem !important; margin: 0 !important; font-weight: 700;">{volume/1e6:.2f}M</div>
                 </div>
                 <div class="metric-card" style="flex-direction: row !important; justify-content: space-between !important; align-items: center !important; padding: 10px 14px !important; min-height: 45px !important; white-space: nowrap !important;">
                     <div class="metric-label" style="margin: 0 !important; font-size: 0.75rem !important; opacity: 0.8;">{t["market_cap"]}</div>
                     <div class="metric-value" style="font-size: 1.1rem !important; margin: 0 !important; font-weight: 700;">{market_cap_val}</div>
                 </div>
                 <div class="metric-card" style="flex-direction: row !important; justify-content: space-between !important; align-items: center !important; padding: 10px 14px !important; min-height: 45px !important; white-space: nowrap !important;">
                     <div class="metric-label" style="margin: 0 !important; font-size: 0.75rem !important; opacity: 0.8;">當日最高</div>
                     <div class="metric-value" style="font-size: 1.1rem !important; margin: 0 !important; font-weight: 700;">{day_high_val}</div>
                 </div>
             </div>
         ''', unsafe_allow_html=True)

        # --- AI 決策與 SMC 分析 (AI Decision & SMC Analysis) ---
        # 提前計算以支援綜合評分共識
        with st.spinner(f"正在生成 {ticker_input} 深度共識分析..."):
            atr = float(atr_series.iloc[-1])
            
            # 將關鍵數據存入 session_state
            st.session_state['ai_score'] = ai_score
            st.session_state['atr'] = atr
            st.session_state['current_price'] = current_price
            st.session_state['ticker_input'] = ticker_input
            
            # AI 進出場策略計算
            entry_strategy = get_ai_entry_strategy(
                data=data, 
                rsi=float(rsi_series.iloc[-1]), 
                ema20=float(ema20.iloc[-1]), 
                ema50=float(ema50.iloc[-1]), 
                bb_lower=float(bb_lower.iloc[-1]), 
                health_scores=health_scores, 
                vr=float(vr_series.iloc[-1]), 
                atr=float(atr_series.iloc[-1]), 
                dynamic_weights=best_weights,
                m_colors=m_colors,
                is_discount=bool(current_price < equilibrium.iloc[-1]),
                has_fvg=bool(bull_fvg.iloc[-1]),
                is_squeeze=bool(is_squeeze.iloc[-1]),
                is_unicorn=bool(has_unicorn_buy.iloc[-1])
            )
            
            target_median = ticker_metadata.get('targetMedianPrice', None)
            exit_strategy = get_ai_exit_strategy(
                data, float(rsi_series.iloc[-1]), float(bb_upper.iloc[-1]), 
                target_median, float(atr_series.iloc[-1]), health_scores=health_scores, m_colors=m_colors
            )

            # SMC 結構化分析
            smc_data = analyze_smc(data)

        # 技術面評分徽章 (Technical Rating Badge) - 現在整合了 SMC 與 AI 共識
        sig_text, sig_color, sig_val = get_signal_score(
            data, float(rsi_series.iloc[-1]), float(macd_series.iloc[-1]), float(signal_series.iloc[-1]), 
            smc_data=smc_data,
            entry_strategy=entry_strategy,
            m_colors=m_colors
        )
        
        st.markdown("---")
            
        tab1, tab2, tab3 = st.tabs([t["tab_overview"], t["tab_tech"], t["tab_news"]])

        with tab1:
            # 1. 快速建議與分析摘要 (Quick Recommendation & Analysis Summary)
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
                smc_data=smc_data,
                entry_strategy=entry_strategy,
                m_colors=m_colors
            )
            
            # 顯示 AI 進化狀態與市場關聯 (Display Evolution Status & Market Correlation)
            st.markdown(f'''<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin-bottom: 24px;">
<!-- AI Engine Status Card -->
<div style="display: flex; justify-content: space-between; align-items: center; padding: 16px 20px; background: rgba(88, 166, 255, 0.05); border: 1px solid rgba(88, 166, 255, 0.2); border-radius: 12px; backdrop-filter: blur(8px); position: relative; overflow: hidden;">
    <div style="position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: #58A6FF;"></div>
    <div style="display: flex; align-items: center; gap: 15px;">
        <div style="font-size: 1.6rem; filter: drop-shadow(0 0 8px rgba(88, 166, 255, 0.4));">🧬</div>
        <div>
            <div style="color: #58A6FF; font-size: 1rem; font-weight: 800; letter-spacing: 0.5px;">AI 智能量化引擎</div>
            <div style="color: #8B949E; font-size: 0.8rem; font-weight: 500;">模式：全自動深度學習優化</div>
        </div>
    </div>
    <div style="text-align: right; background: rgba(63, 185, 80, 0.1); padding: 8px 16px; border-radius: 10px; border: 1px solid rgba(63, 185, 80, 0.2);">
        <div style="color: #3FB950; font-size: 1.2rem; font-weight: 900; font-family: 'JetBrains Mono', monospace;">{learn_acc*100:.1f}%</div>
        <div style="color: #8B949E; font-size: 0.7rem; font-weight: 700; text-transform: uppercase;">預測準確率</div>
    </div>
</div>

<!-- Market Correlation Card -->
<div style="display: flex; justify-content: space-between; align-items: center; padding: 16px 20px; background: rgba(188, 140, 242, 0.05); border: 1px solid rgba(188, 140, 242, 0.2); border-radius: 12px; backdrop-filter: blur(8px); position: relative; overflow: hidden;">
    <div style="position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: #BC8CF2;"></div>
    <div style="display: flex; align-items: center; gap: 15px;">
        <div style="font-size: 1.6rem; filter: drop-shadow(0 0 8px rgba(188, 140, 242, 0.4));">📊</div>
        <div>
            <div style="color: #BC8CF2; font-size: 1rem; font-weight: 800; letter-spacing: 0.5px;">大盤關聯分析</div>
            <div style="color: #8B949E; font-size: 0.8rem; font-weight: 500;">相關性: {market_corr:.2f} | 貝塔: {market_beta:.2f}</div>
        </div>
    </div>
    <div style="text-align: right; background: {"rgba(255, 123, 114, 0.1)" if relative_strength < 0 else "rgba(63, 185, 80, 0.1)"}; padding: 8px 16px; border-radius: 10px; border: 1px solid {"rgba(255, 123, 114, 0.2)" if relative_strength < 0 else "rgba(63, 185, 80, 0.2)"};">
        <div style="color: {"#FF7B72" if relative_strength < 0 else "#3FB950"}; font-size: 1.2rem; font-weight: 900; font-family: 'JetBrains Mono', monospace;">
            {relative_strength:+.1f}%
        </div>
        <div style="color: #8B949E; font-size: 0.7rem; font-weight: 700; text-transform: uppercase;">相對強弱</div>
    </div>
</div>
</div>''', unsafe_allow_html=True)

            # --- SMC Smart Analysis (Daily Dip Style) ---
            if smc_data:
                bias_color = m_colors["up"] if "看多" in smc_data['bias'] else m_colors["down"] if "看空" in smc_data['bias'] else "#8B949E"
                bias_icon = "🟢" if "看多" in smc_data['bias'] else "🔴" if "看空" in smc_data['bias'] else "⚪"
                
                st.markdown(f"""
                <div class="data-card" style="border-left: 5px solid {bias_color} !important; margin-bottom: 24px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <h3 style="margin: 0; font-size: 1.25rem; font-weight: 800; color: var(--text-primary);">🛡️ SMC 結構化智能分析 <span style="font-size: 0.8rem; color: var(--text-secondary); font-weight: 500; margin-left: 8px;">Institutional Context</span></h3>
                        <div style="padding: 6px 16px; border-radius: 30px; background: {bias_color}15; color: {bias_color}; border: 1px solid {bias_color}33; font-weight: 800; font-size: 0.9rem; display: flex; align-items: center; gap: 8px; box-shadow: 0 0 15px {bias_color}11;">
                            {bias_icon} 市場偏向: {smc_data['bias']}
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px;">
                        <div class="info-box">
                            <div class="info-header">市場結構 (Structure)</div>
                            <div class="info-row"><span class="info-label">當前狀態</span><span class="info-value" style="color: {bias_color}; font-weight: 800;">{smc_data['structure']}</span></div>
                            <div class="info-row"><span class="info-label">區間定位</span><span class="info-value" style="color: var(--accent-gold); font-weight: 700;">{smc_data['zone']}</span></div>
                        </div>
                        <div class="info-box">
                            <div class="info-header">關鍵水位 (Levels)</div>
                            <div class="info-row"><span class="info-label">均衡價格 (EQ)</span><span class="info-value" style="font-family: 'JetBrains Mono';">{smc_data['equilibrium']:.2f}</span></div>
                            <div class="info-row"><span class="info-label">區間高/低</span><span class="info-value" style="font-family: 'JetBrains Mono';">{smc_data['range'][1]:.1f} / {smc_data['range'][0]:.1f}</span></div>
                        </div>
                        <div class="info-box">
                            <div class="info-header">最近訂單塊 (Order Block)</div>
                            <div class="info-row"><span class="info-label">看多 OB</span><span class="info-value" style="color: {m_colors['up']}; font-weight: 700;">{f"{smc_data['bull_ob'][0]:.1f}-{smc_data['bull_ob'][1]:.1f}" if smc_data['bull_ob'] else "未發現"}</span></div>
                            <div class="info-row"><span class="info-label">看空 OB</span><span class="info-value" style="color: {m_colors['down']}; font-weight: 700;">{f"{smc_data['bear_ob'][0]:.1f}-{smc_data['bear_ob'][1]:.1f}" if smc_data['bear_ob'] else "未發現"}</span></div>
                        </div>
                        <div class="info-box">
                            <div class="info-header">失衡缺口 (FVG)</div>
                            <div class="info-row"><span class="info-label">最新缺口</span><span class="info-value" style="font-weight: 700;">{f"{smc_data['fvgs'][-1]['type']} ({smc_data['fvgs'][-1]['bottom']:.1f}-{smc_data['fvgs'][-1]['top']:.1f})" if smc_data['fvgs'] else "已回補"}</span></div>
                            <div class="info-row"><span class="info-label">缺口數量</span><span class="info-value">{len(smc_data['fvgs'])}</span></div>
                        </div>
                    </div>
                    <div style="margin-top: 20px; font-size: 0.85rem; color: var(--text-secondary); background: rgba(88, 166, 255, 0.05); border-radius: 8px; padding: 12px 16px; border: 1px solid rgba(88, 166, 255, 0.1); line-height: 1.6;">
                        <span style="color: var(--accent-blue); font-weight: 800; margin-right: 8px;">💡 Daily Dip 策略提示:</span> {'當前處於 <span style="color:#3FB950;font-weight:700;">折價區 (Discount)</span>，若出現看多角色反轉 (CHoCH) 且價格回測 OB/FVG，為高勝率進場點。' if '折價區' in smc_data['zone'] else '目前價格處於 <span style="color:#FF7B72;font-weight:700;">溢價區 (Premium)</span> 或均衡點，建議等待回撤至折價區尋找機會。'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # --- Market Correlation Card ---
            # (已經整合到上方的 AI Engine Status Card 網格中)


            # 推薦建議區塊 (Recommendation Grid)
            st.markdown(f'''<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px;">
<!-- 當前操作建議 -->
<div class="metric-card" style="border-top: 4px solid {insight_report['action']['color']} !important;">
    <div class="metric-label">當前操作建議</div>
    <div class="metric-value" style="color: {insight_report['action']['color']} !important; -webkit-text-fill-color: {insight_report['action']['color']} !important;">{insight_report['action']['status']}</div>
    <div style="color: #C9D1D9; font-size: 0.85rem; margin-top: 10px; font-weight: 500;">{insight_report['action']['desc']}</div>
</div>

<!-- AI 建議買點 -->
<div class="metric-card" style="border-top: 4px solid {entry_strategy['color']} !important;">
    <div class="metric-label">AI 建議買點 ({entry_strategy['confidence']})</div>
    <div class="metric-value" style="color: {entry_strategy['color']} !important; -webkit-text-fill-color: {entry_strategy['color']} !important;">{c_symbol}{entry_strategy['price']:.1f}</div>
    <div style="color: #FF7B72; font-size: 0.9rem; font-weight: 700; margin-top: 5px;">停損: {c_symbol}{entry_strategy['stop_loss']:.1f}</div>
    <div style="color: #C9D1D9; font-size: 0.8rem; margin-top: 8px; font-weight: 500;">{entry_strategy['desc']}</div>
</div>

<!-- AI 建議賣點 -->
<div class="metric-card" style="border-top: 4px solid {exit_strategy['color']} !important;">
    <div class="metric-label">AI 建議賣點 ({exit_strategy['confidence']})</div>
    <div class="metric-value" style="color: {exit_strategy['color']} !important; -webkit-text-fill-color: {exit_strategy['color']} !important;">{c_symbol}{exit_strategy['price']:.1f}</div>
    <div style="color: #3FB950; font-size: 0.9rem; font-weight: 700; margin-top: 5px;">保命線: {c_symbol}{exit_strategy['trailing_stop']:.1f}</div>
    <div style="color: #C9D1D9; font-size: 0.8rem; margin-top: 8px; font-weight: 500;">{exit_strategy['desc']}</div>
</div>
</div>''', unsafe_allow_html=True)

            # --- SMC 診斷與結構 (SMC Diagnosis & Structure) ---
            is_disc = current_price < equilibrium.iloc[-1]
            smc_zone = "折價區 (Discount)" if is_disc else "溢價區 (Premium)"
            smc_zone_color = "#3FB950" if is_disc else "#FF7B72"
            
            fvg_status = "看漲 FVG" if bull_fvg.iloc[-1] else ("看跌 FVG" if bear_fvg.iloc[-1] else "無明顯缺口")
            fvg_color = "#3FB950" if bull_fvg.iloc[-1] else ("#FF7B72" if bear_fvg.iloc[-1] else "#8B949E")
            
            squeeze_status = "動能擠壓中" if is_squeeze.iloc[-1] else "動能釋放中"
            squeeze_color = "#BC8CF2" if is_squeeze.iloc[-1] else "#8B949E"
            
            unicorn_text = "🦄 Unicorn 買入" if has_unicorn_buy.iloc[-1] else ("🚨 Unicorn 賣出" if has_unicorn_sell.iloc[-1] else "無特殊結構")
            unicorn_color = "#3FB950" if has_unicorn_buy.iloc[-1] else ("#FF7B72" if has_unicorn_sell.iloc[-1] else "#8B949E")

            st.markdown(f'''<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 24px;">
<div class="metric-card" style="flex-direction: row !important; justify-content: space-between !important; align-items: center !important; padding: 10px 14px !important; min-height: 45px !important; white-space: nowrap !important;">
    <div style="color: #8B949E; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; margin: 0 !important; opacity: 0.8;">市場區域</div>
    <div style="color: {smc_zone_color}; font-size: 1rem; font-weight: 700; margin: 0 !important;">{smc_zone}</div>
</div>
<div class="metric-card" style="flex-direction: row !important; justify-content: space-between !important; align-items: center !important; padding: 10px 14px !important; min-height: 45px !important; white-space: nowrap !important;">
    <div style="color: #8B949E; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; margin: 0 !important; opacity: 0.8;">失衡狀態</div>
    <div style="color: {fvg_color}; font-size: 1rem; font-weight: 700; margin: 0 !important;">{fvg_status}</div>
</div>
<div class="metric-card" style="flex-direction: row !important; justify-content: space-between !important; align-items: center !important; padding: 10px 14px !important; min-height: 45px !important; white-space: nowrap !important;">
    <div style="color: #8B949E; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; margin: 0 !important; opacity: 0.8;">動能擠壓</div>
    <div style="color: {squeeze_color}; font-size: 1rem; font-weight: 700; margin: 0 !important;">{squeeze_status}</div>
</div>
<div class="metric-card" style="flex-direction: row !important; justify-content: space-between !important; align-items: center !important; padding: 10px 14px !important; min-height: 45px !important; white-space: nowrap !important;">
    <div style="color: #8B949E; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; margin: 0 !important; opacity: 0.8;">專業訊號</div>
    <div style="color: {unicorn_color}; font-size: 1rem; font-weight: 700; margin: 0 !important;">{unicorn_text}</div>
</div>
</div>''', unsafe_allow_html=True)

            # Overview Tab: Advanced Plotly Chart
            financial_data = get_financial_data(resolved_ticker)
            quant_factors = calculate_quant_factors(
                data, ticker_metadata, rsi_series, atr_series, 
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

                # --- SMC Visualizations (Overview) ---
                if smc_data:
                    # 1. BOS & CHoCH Lines
                    for event in smc_data.get('bos_events', []):
                        color = m_colors["up"] if "Bullish" in event['type'] else m_colors["down"]
                        fig.add_hline(y=event['price'], line_dash="dash", line_color=color, opacity=0.4, 
                                     annotation_text=f" {event['type']}", annotation_position="top left", row=1, col=1)
                    
                    for event in smc_data.get('choch_events', []):
                        color = m_colors["up"] if "Bullish" in event['type'] else m_colors["down"]
                        fig.add_hline(y=event['price'], line_dash="dot", line_color=color, opacity=0.6,
                                     annotation_text=f" {event['type']}", annotation_position="bottom left", row=1, col=1)

                    # 2. Order Blocks (OB)
                    for ob in smc_data.get('obs', []):
                        color = "rgba(38, 166, 154, 0.15)" if "Bullish" in ob['type'] else "rgba(239, 83, 80, 0.15)"
                        fig.add_shape(type="rect", x0=ob['start'], y0=ob['bottom'], x1=ob['end'], y1=ob['top'],
                                     fillcolor=color, line_width=0, layer="below", row=1, col=1)
                    
                    # 3. Fair Value Gaps (FVG)
                    for fvg in smc_data.get('fvgs', []):
                        color = "rgba(88, 166, 255, 0.1)" if "Bullish" in fvg['type'] else "rgba(255, 110, 64, 0.1)"
                        fig.add_shape(type="rect", x0=fvg['start'], y0=fvg['bottom'], x1=fvg['end'], y1=fvg['top'],
                                     fillcolor=color, line_width=0, layer="below", row=1, col=1)

                fig.update_layout(
                    template="plotly_dark",
                    height=650,
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=10, r=10, t=50, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    dragmode=False,
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
                    fixedrange=True,
                    spikemode="across", spikesnap="cursor", spikedash="dot", spikethickness=1, spikecolor="#8B949E"
                )
                fig.update_yaxes(
                    showgrid=True, gridwidth=1, gridcolor='rgba(48, 54, 61, 0.3)', 
                    zeroline=False, tickfont=dict(color="#8B949E", size=10),
                    fixedrange=True,
                    side="right" # 價格放在右側更符合交易習慣
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False})

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
                    dragmode=False,
                    hovermode='closest'
                )
                st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})
                
                # 綜合評分顯示 - 升級版
                avg_score = sum(values) / len(values)
                score_color = "#26a69a" if avg_score > 65 else "#D29922" if avg_score > 50 else "#ef5350"
                
                # 計算與前一交易日的變化 (這裡模擬一個小幅波動，或是實際計算)
                # 為了簡單起見，我們先顯示狀態標籤
                status_text = "市場領導者" if avg_score > 80 else "趨勢強勁" if avg_score > 65 else "中性整理" if avg_score > 50 else "弱勢觀察"
                
                st.markdown(f'''<div class="data-card" style="text-align: center; background: linear-gradient(135deg, #1C2128 0%, #0D1117 100%); border: 1px solid #30363D; border-top: 3px solid {score_color}; padding: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); position: relative; overflow: hidden;">
<div style="position: absolute; top: 0; right: 0; width: 80px; height: 80px; background: radial-gradient(circle at top right, {score_color}10, transparent); pointer-events: none;"></div>
<div style="color: #8B949E; font-size: 0.8rem; font-weight: 800; margin-bottom: 12px; letter-spacing: 2px; text-transform: uppercase;">Quantitative Score</div>
<div style="display: flex; justify-content: center; align-items: baseline; gap: 6px;">
    <div style="font-size: 3.2rem; font-weight: 900; color: {score_color}; line-height: 1; text-shadow: 0 0 25px {score_color}33;">{avg_score:.1f}</div>
    <div style="color: #8B949E; font-size: 1.1rem; font-weight: 600; opacity: 0.5;">/ 100</div>
</div>
<div style="margin-top: 16px; padding: 6px 18px; background: {score_color}10; color: {score_color}; border-radius: 8px; font-size: 0.95rem; font-weight: 700; display: inline-block; border: 1px solid {score_color}20;">
{status_text}
</div>
<div style="margin-top: 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 15px; text-align: left; border-top: 1px solid #30363D; padding-top: 16px;">
<div style="font-size: 0.85rem; color: #8B949E; display: flex; align-items: center; gap: 8px;">
    <span style="width: 7px; height: 7px; background: {"#26a69a" if values[0]>60 else "#ef5350"}; border-radius: 50%; box-shadow: 0 0 8px {"#26a69a" if values[0]>60 else "#ef5350"};"></span>
    主力: <span style="color: {"#26a69a" if values[0]>60 else "#ef5350"}; font-weight: 700;">{"看多" if values[0]>60 else "保守"}</span>
</div>
<div style="font-size: 0.85rem; color: #8B949E; display: flex; align-items: center; gap: 8px;">
    <span style="width: 7px; height: 7px; background: {"#26a69a" if values[4]>60 else "#ef5350"}; border-radius: 50%; box-shadow: 0 0 8px {"#26a69a" if values[4]>60 else "#ef5350"};"></span>
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
                
                perf_html = f"""<div class="stat-card" style="margin-top: 15px; display: flex; justify-content: space-around; align-items: center; gap: 20px; background: rgba(88, 166, 255, 0.02); border: 1px solid #30363D; padding: 15px !important;">
<div style="text-align: center; flex: 1;">
<p class="rating-label" style="margin-bottom: 5px !important;">年度營收 ({latest_year})</p>
<div style="display: flex; flex-direction: column; align-items: center;">
<span class="rating-value" style="font-size: 1.4rem !important;">{c_symbol}{rev_latest/1e6:,.0f}M</span>
<span style="font-size: 0.75rem; color: {rev_growth_color}; font-weight: 700; margin-top: 4px;">
{rev_growth:+.1f}% YoY
</span>
</div>
</div>
<div style="width: 1px; height: 35px; background: #30363D;"></div>
<div style="text-align: center; flex: 1;">
<p class="rating-label" style="margin-bottom: 5px !important;">年度淨利 ({latest_year})</p>
<div style="display: flex; flex-direction: column; align-items: center;">
<span class="rating-value" style="font-size: 1.4rem !important;">{c_symbol}{ni_latest/1e6:,.0f}M</span>
<span style="font-size: 0.75rem; color: {ni_growth_color}; font-weight: 700; margin-top: 4px;">
{ni_growth:+.1f}% YoY
</span>
</div>
</div>
</div>"""


            col_gauge, col_stats_data = st.columns([1, 2])
            
            with col_gauge:
                st.plotly_chart(create_tv_gauge(sig_val, sig_text, sig_color, m_colors=m_colors), use_container_width=True, config={'displayModeBar': False})
            
            with col_stats_data:
                # 根據健康得分定義動態樣式 (Define dynamic styling based on health scores)
                profitability_score, leverage_score, cashflow_score = p_score, l_score, c_score
                avg_health = (profitability_score + leverage_score + cashflow_score) / 3
                health_border = "#26a69a" if avg_health > 7 else "#ef5350" if avg_health < 4 else "#30363D"
                health_bg = "rgba(38, 166, 154, 0.05)" if avg_health > 7 else "rgba(239, 83, 80, 0.05)" if avg_health < 4 else "#161B22"
                
                st.markdown(f'''<div class="stat-card">
<div class="stat-header">
<div class="stat-title-group">
<div class="stat-title-bar" style="background: {health_border};"></div>
<h3 class="stat-title">{t["key_stats"]}</h3>
</div>
<div style="text-align: right;">
<div style="font-size: 0.75rem; color: #8B949E; margin-bottom: 4px; font-weight: 600;">財務健康狀態</div>
<span class="stat-badge" style="background: {health_border}20; color: {health_border}; border: 1px solid {health_border}40;">
{'優異' if avg_health > 7 else '警示' if avg_health < 4 else '穩定'}
</span>
</div>
</div>
<div class="rating-grid">
<div class="rating-item">
<div class="rating-label">獲利能力</div>
<div class="rating-value">{profitability_score}<span class="rating-denominator">/10</span></div>
</div>
<div class="rating-item">
<div class="rating-label">財務槓桿</div>
<div class="rating-value">{leverage_score}<span class="rating-denominator">/10</span></div>
</div>
<div class="rating-item">
<div class="rating-label">現金流量</div>
<div class="rating-value">{cashflow_score}<span class="rating-denominator">/10</span></div>
</div>
</div>
<div class="info-grid">
<div class="info-box">
    <p class="info-header">估值分析</p>
    <div class="info-row"><span class="info-label">{t['trailing_pe']}</span><span class="info-value">{ticker_metadata.get('trailingPE', 'N/A')}</span></div>
    <div class="info-row"><span class="info-label">{t['forward_pe']}</span><span class="info-value">{ticker_metadata.get('forwardPE', 'N/A')}</span></div>
    <div class="info-row"><span class="info-label">{t['div_yield']}</span><span class="info-value" style="color: #3FB950;">{ticker_metadata.get('dividendYield', 0)*100:.2f}%</span></div>
</div>
<div class="info-box">
    <p class="info-header">價格區間</p>
    <div class="info-row"><span class="info-label">52 週最高</span><span class="info-value">{ticker_metadata.get('fiftyTwoWeekHigh', 'N/A')}</span></div>
    <div class="info-row"><span class="info-label">52 週最低</span><span class="info-value">{ticker_metadata.get('fiftyTwoWeekLow', 'N/A')}</span></div>
    <div class="info-row"><span class="info-label">Beta 係數</span><span class="info-value">{ticker_metadata.get('beta', 'N/A')}</span></div>
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
                smc_data=smc_data,
                entry_strategy=entry_strategy,
                m_colors=m_colors
            )
            
            # 專業診斷區塊佈局 (Professional Diagnostic Section with Grid Layout)
            st.markdown(f'''<div class="data-card" style="border-top: 6px solid {insight_report['action']['color']} !important; box-shadow: 0 15px 40px {insight_report['action']['color']}20, 0 0 20px {insight_report['action']['color']}10 !important; padding: 18px !important;">
<h3 style="margin-top:0; color: {insight_report['action']['color']}; font-size: 1.25rem; font-weight: 950; display: flex; align-items: center; gap: 12px; margin-bottom: 18px; letter-spacing: -0.3px; text-shadow: 0 0 20px {insight_report['action']['color']}33, 0 3px 10px rgba(0,0,0,0.4);">
<span style="font-size: 1.6rem; filter: drop-shadow(0 0 10px {insight_report['action']['color']}33);">⚖️</span> {ticker_display_name} - 專家技術診斷報告
</h3>
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 18px;">
<!-- RSI Diagnostic -->
<div class="data-card" style="background: linear-gradient(180deg, rgba(48, 54, 61, 0.4) 0%, rgba(13, 17, 23, 0.9) 100%); border: 1px solid rgba(255,255,255,0.1) !important; border-left: 6px solid {insight_report['rsi']['color']} !important; margin-bottom: 0; padding: 16px !important; box-shadow: 0 10px 25px rgba(0,0,0,0.5) !important;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
        <span style="color: #8B949E; font-size: 0.75rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1px;">RSI 強度</span>
        <span style="background: {insight_report['rsi']['color']}; color: white; padding: 4px 10px; border-radius: 8px; font-size: 0.7rem; font-weight: 950; box-shadow: 0 4px 10px {insight_report['rsi']['color']}33; text-shadow: 0 2px 4px rgba(0,0,0,0.4);">{insight_report['rsi']['status']}</span>
    </div>
    <div style="font-size: 1.8rem; font-weight: 950; color: #FFFFFF; margin-bottom: 10px; letter-spacing: -1px; text-shadow: 0 0 15px rgba(255,255,255,0.12);">{insight_report['rsi']['val']}</div>
    <div style="color: #E0E0E0; font-size: 0.85rem; line-height: 1.6; font-weight: 500; text-shadow: 0 1px 2px rgba(0,0,0,0.5);">{insight_report['rsi']['desc']}</div>
</div>

<!-- MACD Diagnostic -->
<div class="data-card" style="background: linear-gradient(180deg, rgba(48, 54, 61, 0.4) 0%, rgba(13, 17, 23, 0.9) 100%); border: 1px solid rgba(255,255,255,0.1) !important; border-left: 6px solid {insight_report['macd']['color']} !important; margin-bottom: 0; padding: 16px !important; box-shadow: 0 10px 25px rgba(0,0,0,0.5) !important;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
        <span style="color: #8B949E; font-size: 0.75rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1px;">MACD 差值</span>
        <span style="background: {insight_report['macd']['color']}; color: white; padding: 4px 10px; border-radius: 8px; font-size: 0.7rem; font-weight: 950; box-shadow: 0 4px 10px {insight_report['macd']['color']}33; text-shadow: 0 2px 4px rgba(0,0,0,0.4);">{insight_report['macd']['status']}</span>
    </div>
    <div style="font-size: 1.8rem; font-weight: 950; color: #FFFFFF; margin-bottom: 10px; letter-spacing: -1px; text-shadow: 0 0 15px rgba(255,255,255,0.12);">{insight_report['macd']['val']}</div>
    <div style="color: #E0E0E0; font-size: 0.85rem; line-height: 1.6; font-weight: 500; text-shadow: 0 1px 2px rgba(0,0,0,0.5);">{insight_report['macd']['desc']}</div>
</div>

<!-- AI Layout Suggestion -->
<div class="data-card" style="background: linear-gradient(180deg, rgba(88, 166, 255, 0.1) 0%, rgba(13, 17, 23, 0.95) 100%); border: 2px dashed rgba(88, 166, 255, 0.35) !important; border-left: 6px solid #58A6FF !important; margin-bottom: 0; padding: 16px !important; box-shadow: 0 15px 35px rgba(0,0,0,0.5), inset 0 0 15px rgba(88, 166, 255, 0.08) !important;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
        <span style="color: #58A6FF; font-size: 0.75rem; font-weight: 900; text-transform: uppercase; letter-spacing: 1.5px; text-shadow: 0 0 8px rgba(88, 166, 255, 0.3);">AI 建議佈局</span>
        <span style="color: {entry_strategy['color']}; font-weight: 950; font-size: 1rem; text-shadow: 0 0 15px {entry_strategy['color']}33, 0 2px 5px rgba(0,0,0,0.5);">{entry_strategy['action']}</span>
    </div>
    <div style="font-size: 1.05rem; color: #FFFFFF; font-weight: 900; margin-bottom: 6px; letter-spacing: -0.2px;">策略部署 (${entry_strategy['price']:.1f})</div>
    <div style="color: #E0E0E0; font-size: 0.8rem; line-height: 1.5; font-weight: 500;">{entry_strategy['desc']}</div>
</div>

<!-- Swing Advice -->
<div class="data-card" style="background: linear-gradient(180deg, rgba(255, 215, 0, 0.06) 0%, rgba(13, 17, 23, 0.95) 100%); border: 1.5px solid rgba(255, 215, 0, 0.2) !important; border-left: 5px solid #D29922 !important; margin-bottom: 0; padding: 14px !important; box-shadow: 0 12px 25px rgba(0,0,0,0.4), inset 0 0 12px rgba(255, 215, 0, 0.03) !important;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
        <span style="color: #D29922; font-size: 0.75rem; font-weight: 900; text-transform: uppercase; letter-spacing: 1.2px; text-shadow: 0 0 6px rgba(210, 153, 34, 0.25);">波段操作建議</span>
        <span style="font-size: 1.3rem; filter: drop-shadow(0 0 6px rgba(210, 153, 34, 0.3));">💡</span>
    </div>
    <div style="font-size: 1.05rem; color: #FFFFFF; font-weight: 900; margin-bottom: 6px; letter-spacing: -0.2px;">操作方針</div>
    <div style="color: #E0E0E0; font-size: 0.8rem; line-height: 1.5; font-weight: 500;">{insight_report['swing_advice']}</div>
</div>
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

<!-- SMC Bias Card -->
<div class="data-card" style="margin-bottom: 0; border-top: 4px solid {m_colors['up'] if 'Bullish' in smc_data.get('bias', '') else m_colors['down'] if 'Bearish' in smc_data.get('bias', '') else '#8B949E'}; background: linear-gradient(145deg, rgba(38, 166, 154, 0.05) 0%, #161B22 100%);">
    <div style="color: #8B949E; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px;">SMC 市場結構偏向</div>
    <div style="font-size: 1.2rem; font-weight: 800; margin: 12px 0; color: #F0F6FC; letter-spacing: -0.2px;">
        {smc_data.get('bias', 'N/A')}
    </div>
    <div style="color: #8B949E; font-size: 0.75rem; font-family: 'Roboto Mono', monospace; font-weight: 500;">
        結構狀態: {smc_data.get('structure', 'N/A')}
    </div>
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
            # fig_tech.add_trace(go.Scatter(x=data.index, y=supertrend_line, name='SuperTrend', line=dict(color=st_color, width=1.5, dash='dash'), opacity=0.6))

            fig_tech.update_layout(
                title="<b>EMA 趨勢與布林通道分析</b>",
                template="plotly_dark", height=500,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=50, b=10),
                dragmode=False,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
                hovermode="x unified",
                xaxis=dict(showgrid=True, gridcolor='rgba(48, 54, 61, 0.2)', fixedrange=True),
                yaxis=dict(showgrid=True, gridcolor='rgba(48, 54, 61, 0.2)', side="right", fixedrange=True)
            )
            st.plotly_chart(fig_tech, use_container_width=True, config={'displayModeBar': False})


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
                    dragmode=False,
                    hovermode="x unified",
                    xaxis=dict(showgrid=False, fixedrange=True),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', side="right", fixedrange=True)
                )
                st.plotly_chart(fig_rsi, use_container_width=True, config={'displayModeBar': False})
            
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
                    dragmode=False,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
                    hovermode="x unified",
                    xaxis=dict(showgrid=False, fixedrange=True),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', side="right", fixedrange=True)
                )
                st.plotly_chart(fig_macd, use_container_width=True, config={'displayModeBar': False})

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
                    dragmode=False,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
                    hovermode="x unified",
                    xaxis=dict(showgrid=False, fixedrange=True),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', side="right", fixedrange=True)
                )
                st.plotly_chart(fig_kd, use_container_width=True, config={'displayModeBar': False})
            
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
                    dragmode=False,
                    hovermode="x unified",
                    xaxis=dict(showgrid=False, fixedrange=True),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', side="right", fixedrange=True)
                )
                st.plotly_chart(fig_vr, use_container_width=True, config={'displayModeBar': False})


        with tab3:
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
        with st.expander("🔍 診斷資訊 (Diagnostic Info)"):
            st.write(f"輸入代碼: `{ticker_input}`")
            st.write(f"解析代碼: `{resolved_ticker}`")
            st.write(f"查詢範圍: `{period}`")
            st.write(f"資料密度: `{interval}`")
            if st.button("嘗試清除快取並重試"):
                st.cache_data.clear()
                st.rerun()
            st.info("提示：台股請確保代碼後有 .TW 或 .TWO，例如 2330.TW")
else:
    st.info("請在側邊欄輸入股票代碼以開始分析（例如：2330.TW）")