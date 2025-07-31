import streamlit as st
from ui.stock_tab import render_stock_tab
from ui.ai_tab import render_ai_tab
from ui.news_tab import render_stock_news_tab
from ui.bist100_tab import render_bist100_tab
from ui.ml_tab import render_ml_prediction_tab
from ui.ml_prediction_tab import render_ml_prediction_tab as render_ml_scan_tab
from ui.analysis_history_tab import render_analysis_history_tab
from ui.portfolio_tab import render_portfolio_tab
from ui.technical_screener_tab import render_technical_screener_tab
from ui.enhanced_stock_screener_tab import render_enhanced_stock_screener_tab
from ui.stock_profiler_ui import render_stock_profiler_tab
from ui.comprehensive_scanner_tab import render_comprehensive_scanner_tab
from data.db_utils import DB_FILE, get_analysis_results
from data.stock_data import get_market_summary, get_popular_stocks
from data.announcements import get_announcements, get_all_announcements
from data.utils import get_analysis_result, save_analysis_result, get_favorites
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import re
import traceback
import os
from pathlib import Path
import random
import sys
import platform

# Loglama yapılandırması
logger = logging.getLogger(__name__)

def add_to_favorites(stock_symbol):
    """
    Bir hisse senedini favorilere ekler.
    
    Args:
        stock_symbol (str): Eklenecek hisse senedi sembolü
    
    Returns:
        bool: İşlem başarılıysa True, aksi halde False
    """
    try:
        # Session state'i kontrol et
        if 'favorite_stocks' not in st.session_state:
            st.session_state.favorite_stocks = []
        
        # Sembolü düzenle
        stock_symbol = stock_symbol.upper().strip()
        
        # Favorilere ekle
        if stock_symbol not in st.session_state.favorite_stocks:
            st.session_state.favorite_stocks.append(stock_symbol)
            logger.info(f"{stock_symbol} favorilere eklendi")
            return True
        
        logger.info(f"{stock_symbol} zaten favorilerde")
        return False
    except Exception as e:
        logger.error(f"Favori eklenirken hata: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def remove_from_favorites(stock_symbol):
    """
    Bir hisseyi favorilerden çıkarır.
    
    Args:
        stock_symbol (str): Çıkarılacak hisse sembolü
    """
    try:
        # Favorilerden çıkar
        if 'favorite_stocks' in st.session_state and stock_symbol in st.session_state.favorite_stocks:
            st.session_state.favorite_stocks.remove(stock_symbol)
            logger.info(f"{stock_symbol} favorilerden çıkarıldı")
            return True
        return False
    except Exception as e:
        logger.error(f"Favori çıkarılırken hata: {str(e)}")
        return False

def is_favorite(stock_symbol):
    """
    Bir hisse senedinin favorilerde olup olmadığını kontrol eder.
    
    Args:
        stock_symbol (str): Hisse senedi sembolü
        
    Returns:
        bool: Favorilerde ise True, değilse False
    """
    if 'favorite_stocks' not in st.session_state:
        return False
    
    return stock_symbol.upper().strip() in st.session_state.favorite_stocks

def main():
    """
    Ana uygulama arayüzü
    """
    
    # CSS stil ekle - Sade tasarım
    st.markdown("""
    <style>
        /* Ana stil ayarları */
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            color: #333;
            background-color: #f9f9f9;
        }
        
        /* Streamlit başlık stillerini düzenle */
        h1 {
            font-size: 1.5rem !important;
            margin-top: 0 !important;
            margin-bottom: 0.5rem !important;
            padding-top: 0 !important;
        }
        
        h2 {
            font-size: 1.3rem !important;
            margin-top: 0 !important;
            margin-bottom: 0.4rem !important;
        }
        
        h3 {
            font-size: 1.1rem !important;
            margin-top: 0 !important;
            margin-bottom: 0.3rem !important;
        }
        
        /* Scrollbar stil ayarları */
        ::-webkit-scrollbar {
            width: 8px;
            background: #f1f1f1;
        }
        ::-webkit-scrollbar-thumb {
            background: #ccc;
            border-radius: 5px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #aaa;
        }
        
        /* Sade header */
        .main-header {
            background-color: #f5f5f5;
            padding: 1.5rem;
            border-radius: 5px;
            border: 1px solid #e0e0e0;
            margin-bottom: 1.5rem;
            text-align: center;
        }
        
        .main-header h1 {
            font-weight: 600;
            margin-bottom: 0.3rem;
            font-size: 1.8rem;
            color: #333;
        }
        
        .main-header p {
            font-size: 1rem;
            color: #666;
        }
        
        /* Sade sekme tasarımı */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.1rem;
            border-radius: 5px;
            padding: 0.2rem;
            background-color: #f5f5f5;
            border: 1px solid #e0e0e0;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 4px;
            padding: 0.3rem 0.7rem;
            font-weight: 500;
            background-color: #f5f5f5;
            transition: all 0.2s ease;
            border: none !important;
            font-size: 0.8rem;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e9e9e9;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #e0e0e0 !important;
            color: #333 !important;
        }
        
        /* Sidebar stil ayarları */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
            border-right: 1px solid #e0e0e0;
        }
        
        [data-testid="stSidebar"] > div:first-child {
            background-color: #f8f9fa;
            padding-top: 1rem;
        }
        
        /* Sidebar başlık */
        [data-testid="stSidebar"] h1 {
            font-size: 1.2rem !important;
            color: #333;
            margin-bottom: 1rem;
        }
        
        /* Sidebar selectbox */
        [data-testid="stSidebar"] .stSelectbox > div > div {
            background-color: white;
            border-radius: 5px;
            border: 1px solid #ddd;
        }
        
        /* Ana içerik alanını genişlet */
        .main .block-container {
            max-width: 100%;
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
        }
        
        /* Streamlit elementleri arasındaki boşlukları azalt */
        div.element-container {
            margin-top: 0.3rem !important;
            margin-bottom: 0.3rem !important;
        }
        
        /* Streamlit yazı tipini ince ve küçük yap */
        .stMarkdown p {
            font-size: 0.9rem !important;
            margin-bottom: 0.3rem !important;
            line-height: 1.3 !important;
        }
        
        /* Sade kartlar */
        div[data-testid="stBlock"] {
            background-color: #fff;
            padding: 0.5rem;
            border-radius: 5px;
            border: 1px solid #e0e0e0;
            margin-bottom: 0.5rem;
        }
        
        /* Sade metrik kartları */
        [data-testid="stMetric"] {
            background-color: #f9f9f9;
            padding: 0.5rem;
            border-radius: 5px;
            border: 1px solid #e0e0e0;
        }
        
        [data-testid="stMetricValue"] {
            font-weight: 600;
            font-size: 1.2rem;
        }
        
        [data-testid="stMetricDelta"] {
            font-weight: 500;
            font-size: 0.9rem;
        }
        
        /* Sade butonlar */
        .stButton button {
            border-radius: 4px;
            font-weight: 500;
            background-color: #f0f0f0;
            color: #333;
            border: 1px solid #ddd;
            padding: 0.5rem 1rem;
            transition: all 0.2s ease;
        }
        
        .stButton button:hover {
            background-color: #e5e5e5;
        }
        
        /* Sade analiz kartları */
        .analysis-card {
            background-color: #fff;
            border-radius: 5px;
            border: 1px solid #e0e0e0;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        .card-header {
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 0.8rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #eee;
            color: #333;
        }
        
        /* Sade tablolar */
        table {
            border-collapse: separate;
            width: 100%;
            border-radius: 5px;
            overflow: hidden;
            border-spacing: 0;
            border: 1px solid #e0e0e0;
        }
        
        thead {
            background-color: #f5f5f5;
        }
        
        th {
            text-align: left;
            padding: 0.8rem;
            font-weight: 600;
            color: #333;
            border-bottom: 1px solid #e0e0e0;
        }
        
        td {
            padding: 0.7rem 0.8rem;
            border-top: 1px solid #eee;
            color: #444;
        }
        
        tr:hover {
            background-color: #f9f9f9;
        }
        
        /* Input alanları */
        .stTextInput > div > div > input {
            border-radius: 4px;
            border: 1px solid #ddd;
            padding: 0.5rem 0.8rem;
            font-size: 0.9rem;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #aaa;
        }
        
        .stSelectbox > div > div > select {
            border-radius: 4px;
            padding: 0.5rem 0.8rem;
            font-size: 0.9rem;
            border: 1px solid #ddd;
            background-color: white;
        }
        
        .stSelectbox > div > div[data-baseweb="select"] > div {
            border-radius: 4px;
            border: 1px solid #ddd;
        }
        
        .stSelectbox > div > div[data-baseweb="select"] > div:focus-within {
            border-color: #aaa;
        }
        
        /* Diğer stil ayarları */
        .css-18e3th9 {
            padding-top: 0.5rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        
        footer {
            visibility: hidden;
        }
        
        /* Grafik container'ı */
        .chart-container {
            background-color: #fff;
            padding: 1rem;
            border-radius: 5px;
            border: 1px solid #e0e0e0;
            margin-bottom: 1rem;
        }
        
        /* Hisse kartları için özel stil */
        .stock-card {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.8rem;
            background-color: #f9f9f9;
            border-radius: 5px;
            margin-bottom: 0.5rem;
            border: 1px solid #e0e0e0;
        }
        
        .stock-card:hover {
            background-color: #f5f5f5;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Session state değişkenlerini başlat
    if 'analyzed_stocks' not in st.session_state:
        st.session_state.analyzed_stocks = {}
    
    if 'favorite_stocks' not in st.session_state:
        st.session_state.favorite_stocks = []
    
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    if 'last_predictions' not in st.session_state:
        st.session_state.last_predictions = {}
    
    if 'realtime_data' not in st.session_state:
        st.session_state.realtime_data = {}
    
    if 'signals_cache' not in st.session_state:
        st.session_state.signals_cache = {}
    
    if 'trend_cache' not in st.session_state:
        st.session_state.trend_cache = {}
    
    if 'stock_analysis_results' not in st.session_state:
        st.session_state.stock_analysis_results = {}
    
    # Piyasa verilerini çek
    market_data = get_market_summary()
    popular_stocks_data = get_popular_stocks()
    
    # Uygulamanın ana başlığı - Sade tasarım
    st.markdown('<div class="main-header"><h1>Borsa İstanbul Hisse Analiz Paneli</h1><p>Teknik analiz, yapay zeka tahminleri ve piyasa görünümü</p></div>', unsafe_allow_html=True)
    
    # Favori hisseler bölümü - yatay düzen
    if st.session_state.favorite_stocks:
        st.markdown('<div style="padding: 0.5rem; background-color: #f9f9f9; border-radius: 5px; border: 1px solid #e0e0e0; margin-bottom: 1rem;">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-bottom: 0.5rem;'>⭐ Favori Hisseler</h4>", unsafe_allow_html=True)
        
        cols = st.columns(min(len(st.session_state.favorite_stocks), a=5))  
        for idx, stock_symbol in enumerate(st.session_state.favorite_stocks):
            col_idx = idx % 5
            with cols[col_idx]:
                st.markdown(f"<div style='text-align: center; margin-bottom: 0.5rem;'><strong>{stock_symbol}</strong></div>", unsafe_allow_html=True)       
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔍", key=f"analyze_{stock_symbol}", help="Hisseyi analiz et"):
                        # Analiz sekmesine yönlendir ve bu hisseyi analiz et
                        st.session_state.selected_stock_for_analysis = stock_symbol
                        st.info(f"{stock_symbol} analizi için Hisse Analizi sekmesine gidin.")
                with col2:
                    if st.button("❌", key=f"remove_{stock_symbol}", help="Favorilerden çıkar"):
                        if remove_from_favorites(stock_symbol):
                            st.success(f"{stock_symbol} favorilerden çıkarıldı.")
                        else:
                            st.error("Hisse çıkarılırken bir hata oluştu.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Sidebar navigasyon menüsü
    st.sidebar.title("📊 Borsa Analiz Paneli")
    
    # Sayfa seçim menüsü
    selected_page = st.sidebar.selectbox(
        "Sayfa Seçin:",
        [
            "🔍 Hisse Analizi",
            "📊 BIST100 Genel Bakış", 
            "🧠 Yapay Zeka",
            "📈 ML Tahminleri",
            "🔎 ML Tarama",
            "🎯 ML Backtest",
            "🏷️ Hisse Profilleri",
            "🔎 Teknik Tarama",
            "🚀 Gelişmiş Tarayıcı",
            "🔍 Kapsamlı Tarayıcı",
            "📝 Analiz Geçmişi",
            "📰 Haberler",
            "💼 Portföy"
        ],
        key="main_page_selector"
    )
    
    # Sidebar'da ek bilgiler
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Hızlı Erişim")
    
    # ML Backtest için hızlı erişim
    if st.sidebar.button("🎯 Hızlı Backtest", help="ML modelinizi hızlıca test edin"):
        selected_page = "🎯 ML Backtest"
        st.rerun()
    
    # Favoriler
    if st.session_state.favorite_stocks:
        st.sidebar.markdown("### ⭐ Favoriler")
        for stock in st.session_state.favorite_stocks[:5]:  # İlk 5 favoriyi göster
            if st.sidebar.button(f"📊 {stock}", key=f"sidebar_{stock}", help=f"{stock} hissesini analiz et"):
                st.session_state.selected_stock_for_analysis = stock
                selected_page = "🔍 Hisse Analizi"
                st.rerun()
    
    # Ana içerik
    col1, col2 = st.columns([7, 3])
    
    with col1:
        # Seçilen sayfayı render et
        if selected_page == "🔍 Hisse Analizi":
            render_stock_tab()
        elif selected_page == "📊 BIST100 Genel Bakış":
            render_bist100_tab()
        elif selected_page == "🧠 Yapay Zeka":
            render_ai_tab()
        elif selected_page == "📈 ML Tahminleri":
            render_ml_prediction_tab()
        elif selected_page == "🔎 ML Tarama":
            render_ml_scan_tab()
        elif selected_page == "🎯 ML Backtest":
            from ui.ml_backtest_tab import render_ml_backtest_tab
            render_ml_backtest_tab()
        elif selected_page == "🏷️ Hisse Profilleri":
            render_stock_profiler_tab()
        elif selected_page == "🔎 Teknik Tarama":
            render_technical_screener_tab()
        elif selected_page == "🚀 Gelişmiş Tarayıcı":
            render_enhanced_stock_screener_tab()
        elif selected_page == "🔍 Kapsamlı Tarayıcı":
            render_comprehensive_scanner_tab()
        elif selected_page == "📝 Analiz Geçmişi":
            render_analysis_history_tab()
        elif selected_page == "📰 Haberler":
            render_stock_news_tab()
        elif selected_page == "💼 Portföy":
            render_portfolio_tab()
    
    with col2:
        # Piyasa Güncellemeleri kartı - Sade tasarım
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)  
        st.markdown('<div class="card-header">📊 Piyasa Güncellemeleri</div>', unsafe_allow_html=True)
        
        # BIST100 Verileri
        bist_data = market_data["bist100"]
        bist_status = bist_data["status"]
        bist_color = "green" if bist_status == "yükseliş" else ("red" if bist_status == "düşüş" else "gray")
        bist_arrow = "↑" if bist_status == "yükseliş" else ("↓" if bist_status == "düşüş" else "→")
        
        st.markdown(f"""
        <div style="padding: 0.8rem; background-color: #f5f5f5; border-radius: 5px; margin-bottom: 0.8rem; border: 1px solid #e0e0e0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
                <strong>BIST100</strong>
                <span style="color: {bist_color}; font-weight: 600;">{bist_data["value"]:,.2f} {bist_arrow}</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.9rem; color: #666;">
                <span>Değişim: {'+' if bist_data["change"] > 0 else ''}{bist_data["change_percent"]:.2f}%</span>
                <span>Hacim: {bist_data["volume"]:.1f}B ₺</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # USD/TRY Verileri
        usd_data = market_data["usdtry"]
        usd_status = usd_data["status"]
        usd_color = "green" if usd_status == "yükseliş" else ("red" if usd_status == "düşüş" else "gray")
        usd_arrow = "↑" if usd_status == "yükseliş" else ("↓" if usd_status == "düşüş" else "→")
        
        st.markdown(f"""
        <div style="padding: 0.8rem; background-color: #f5f5f5; border-radius: 5px; margin-bottom: 0.8rem; border: 1px solid #e0e0e0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
                <strong>USD/TRY</strong>
                <span style="color: {usd_color}; font-weight: 600;">{usd_data["value"]:.2f} {usd_arrow}</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.9rem; color: #666;">
                <span>Değişim: {'+' if usd_data["change"] > 0 else ''}{usd_data["change_percent"]:.2f}%</span>
                <span>24s: {usd_data["range"]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # ALTIN/ONS Verileri
        gold_data = market_data["gold"]
        gold_status = gold_data["status"]
        gold_color = "green" if gold_status == "yükseliş" else ("red" if gold_status == "düşüş" else "gray")
        gold_arrow = "↑" if gold_status == "yükseliş" else ("↓" if gold_status == "düşüş" else "→")
        
        st.markdown(f"""
        <div style="padding: 0.8rem; background-color: #f5f5f5; border-radius: 5px; margin-bottom: 0.8rem; border: 1px solid #e0e0e0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
                <strong>ALTIN/ONS</strong>
                <span style="color: {gold_color}; font-weight: 600;">{gold_data["value"]:.1f} {gold_arrow}</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.9rem; color: #666;">
                <span>Değişim: {'+' if gold_data["change"] > 0 else ''}{gold_data["change_percent"]:.2f}%</span>
                <span>24s: {gold_data["range"]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Son Analizler kartı - Sade tasarım
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)  
        st.markdown('<div class="card-header">🔄 Son Analizler</div>', unsafe_allow_html=True)
        
        # Veritabanından son analizleri getir
        last_analyses = get_analysis_results(limit=5)  # Son 5 analizi al   
        
        if last_analyses:
            for analysis in last_analyses:
                # Analize ait bilgileri göster
                stock = analysis["symbol"]
                analysis_id = analysis.get("id", 0)  # Analiz ID'sini al, yoksa 0 kullan
                recommendation = analysis["result_data"].get("recommendation", "") if analysis["result_data"] else ""
                recommendation_color = "green" if "AL" in recommendation else ("red" if "SAT" in recommendation else "gray")
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"""
                    <div style="padding: 0.5rem 0;">
                        <strong>{stock}</strong>
                        <div style="font-size: 0.8rem; color: {recommendation_color};">{recommendation}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button("🔄", key=f"reanalyze_{stock}_{analysis_id}", help="Yeniden analiz et"):
                        st.session_state.selected_stock_for_analysis = stock
                        st.info(f"{stock} yeniden analizi için Hisse Analizi sekmesine gidin.")
                
                st.markdown("<hr style='margin: 0.3rem 0; border-color: #eee;'>", unsafe_allow_html=True)
        else:
            st.info("Henüz analiz yapılmadı.")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Popüler Hisseler kartı - Sade tasarım
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)  
        st.markdown('<div class="card-header">🔥 Popüler Hisseler</div>', unsafe_allow_html=True)
        
        for stock in popular_stocks_data:
            symbol = stock["symbol"]
            name = stock["name"]
            change_percent = stock["change_percent"]
            change_text = f"+{change_percent:.2f}%" if change_percent > 0 else f"{change_percent:.2f}%"
            change_color = "green" if change_percent > 0 else ("red" if change_percent < 0 else "gray")
            
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.8rem;
                 background-color: #f5f5f5; border-radius: 5px; margin-bottom: 0.5rem; border: 1px solid #e0e0e0;">
                <div>
                    <strong>{symbol}</strong>
                    <div style="font-size: 0.85rem; color: #666;">{name}</div>
                </div>
                <div>
                    <div style="color: {change_color}; font-weight: 600; text-align: right;">{change_text}</div>
                    <div style="font-size: 0.85rem; color: #666; text-align: right;">{stock["value"]:.2f} ₺</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Market Duyuruları Bölümü - Sade tasarım
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)  
        st.markdown('<div class="card-header">📢 Önemli Duyurular</div>', unsafe_allow_html=True)
        
        display_duyurular()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Analiz Sonuçları kartı - Sade tasarım
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)  
        st.markdown('<div class="card-header">📊 Analiz Sonuçları</div>', unsafe_allow_html=True)
        
        if st.session_state.analyzed_stocks:
            for symbol, data in list(st.session_state.analyzed_stocks.items())[-3:]:
                st.markdown(f"""
                <div style="padding: 0.8rem; background-color: #f5f5f5; border-radius: 5px; margin-bottom: 0.8rem; border: 1px solid #e0e0e0;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
                        <strong>{symbol}</strong>
                        <span style="color: green; font-weight: 600; background-color: rgba(0, 128, 0, 0.05); padding: 0.2rem 0.5rem; border-radius: 4px;">AL</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.9rem; color: #666;">
                        <span>Trend: Yükseliş</span>
                        <span>Risk: Orta</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Henüz hisse analizi yapılmadı.")
        
        st.markdown("</div>", unsafe_allow_html=True)

def display_duyurular():
    """
    Duyuruları görüntüler.
    """
    try:
        # get_announcements yerine get_all_announcements kullan - dinamik + sabit duyurular
        duyurular = get_all_announcements()
        
        if not duyurular:
            return
        
        st.markdown("""
        <div style="margin-bottom: 20px;">
            <h3 style="color: #334155; font-size: 20px; font-weight: 600; margin-bottom: 16px;">
                <i class="fas fa-bullhorn"></i> Önemli Duyurular
            </h3>
            <div class="duyurular-container">
        """, unsafe_allow_html=True)
        
        for duyuru in duyurular:
            st.markdown(f"""
            <div style="background-color: {duyuru['renk']}; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="color: {duyuru['koyu_renk']}; font-weight: 600; font-size: 15px;">{duyuru['baslik']}</span>
                    <span style="color: #64748B; font-size: 12px;">{duyuru['zaman']}</span>
                </div>
                <p style="color: #334155; font-size: 14px; margin: 4px 0;">{duyuru['icerik']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Duyurular görüntülenirken hata oluştu: {str(e)}")
        logger.error(traceback.format_exc())
        st.warning("Duyurular yüklenirken bir sorun oluştu.") 