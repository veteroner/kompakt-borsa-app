import streamlit as st
import os
import sys

# Streamlit Cloud için sayfa konfigürasyonu
st.set_page_config(
    page_title="Kompakt Borsa Analiz Uygulaması",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cloud ortamı için path ayarları
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from ui.stock_tab import render_stock_tab
    from ui.bist100_tab import render_bist100_tab
    from ui.ml_prediction_tab import render_ml_prediction_tab
    from data.db_utils import DB_FILE, get_analysis_results
    from data.stock_data import get_market_summary, get_popular_stocks
    from data.announcements import get_announcements, get_all_announcements  
    from data.utils import get_analysis_result, save_analysis_result, get_favorites
except ImportError as e:
    st.error(f"Modül import hatası: {e}")
    st.info("Lütfen tüm gerekli dosyaların projenizde bulunduğundan emin olun.")
    st.stop()

import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import re
import traceback
from pathlib import Path

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
    Kompakt borsa uygulaması - BIST 100, Hisse Analizi ve ML Tahminleri
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
        
        /* Ana içerik alanını genişlet */
        .main .block-container {
            max-width: 100%;
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
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
        
        /* Analiz kartları */
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
        
        footer {
            visibility: hidden;
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
    try:
        market_data = get_market_summary()
        popular_stocks_data = get_popular_stocks()
    except Exception as e:
        logger.error(f"Piyasa verileri alınırken hata: {str(e)}")
        # Streamlit Cloud için varsayılan veriler
        market_data = {
            "bist100": {
                "value": 10000, 
                "change": 0, 
                "change_percent": 0, 
                "volume": 0, 
                "status": "bilinmiyor"
            },
            "usdtry": {
                "value": 34.0,
                "change": 0,
                "change_percent": 0,
                "range": "33.5-34.5",
                "status": "bilinmiyor"
            },
            "gold": {
                "value": 2650.0,
                "change": 0,
                "change_percent": 0,
                "range": "2640-2660",
                "status": "bilinmiyor"
            }
        }
        popular_stocks_data = [
            {"symbol": "THYAO", "name": "Türk Hava Yolları", "value": 250.0, "change_percent": 2.5},
            {"symbol": "GARAN", "name": "Garanti BBVA", "value": 85.0, "change_percent": -1.2},
            {"symbol": "ASELS", "name": "Aselsan", "value": 75.0, "change_percent": 1.8},
            {"symbol": "AKBNK", "name": "Akbank", "value": 55.0, "change_percent": -0.5},
            {"symbol": "EREGL", "name": "Ereğli Demir Çelik", "value": 40.0, "change_percent": 3.2}
        ]
    
    # Uygulamanın ana başlığı
    st.markdown('<div class="main-header"><h1>📊 Kompakt Borsa Analiz Uygulaması</h1><p>BIST 100 Genel Bakış • Hisse Analizi • ML Yükseliş Tahminleri</p></div>', unsafe_allow_html=True)
    
    # Favori hisseler bölümü - yatay düzen
    if st.session_state.favorite_stocks:
        st.markdown('<div style="padding: 0.5rem; background-color: #f9f9f9; border-radius: 5px; border: 1px solid #e0e0e0; margin-bottom: 1rem;">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-bottom: 0.5rem;'>⭐ Favori Hisseler</h4>", unsafe_allow_html=True)
        
        cols = st.columns(min(len(st.session_state.favorite_stocks), 5))  
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
                            st.rerun()
                        else:
                            st.error("Hisse çıkarılırken bir hata oluştu.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Ana içerik alanı - Tab yapısı
    tab1, tab2, tab3 = st.tabs(["📊 BIST 100 Genel Bakış", "🔍 Hisse Analizi", "🧠 ML Yükseliş Tahmini"])
    
    # Piyasa özeti için sidebar
    with st.sidebar:
        st.title("📊 Piyasa Özeti")
        
        # BIST100 Verileri
        try:
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
        except Exception as e:
            st.error("BIST 100 verisi alınamadı")
        
        # Son Analizler
        st.markdown("### 🔄 Son Analizler")
        
        # Veritabanından son analizleri getir
        try:
            last_analyses = get_analysis_results(limit=3)  # Son 3 analizi al   
            
            if last_analyses:
                for analysis in last_analyses:
                    stock = analysis.get("symbol", "N/A")
                    analysis_id = analysis.get("id", 0)
                    result_data = analysis.get("result_data", {})
                    if isinstance(result_data, dict):
                        recommendation = result_data.get("recommendation", "")
                    else:
                        recommendation = ""
                    recommendation_color = "green" if "AL" in recommendation else ("red" if "SAT" in recommendation else "gray")
                    
                    col1, col2 = st.columns([3, 1])
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
                            st.info(f"{stock} için Hisse Analizi sekmesine gidin.")
            else:
                st.info("Henüz analiz yapılmadı.")
        except Exception as e:
            # Streamlit Cloud'da database sorunları olabilir
            st.info("Analiz geçmişi şu anda yüklenemiyor.")
            # Örnek veriler göster
            sample_analyses = [
                {"symbol": "THYAO", "recommendation": "AL"},
                {"symbol": "GARAN", "recommendation": "SAT"},
                {"symbol": "ASELS", "recommendation": "AL"}
            ]
            
            for idx, analysis in enumerate(sample_analyses):
                stock = analysis["symbol"]
                recommendation = analysis["recommendation"]
                recommendation_color = "green" if "AL" in recommendation else "red"
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"""
                    <div style="padding: 0.5rem 0;">
                        <strong>{stock}</strong>
                        <div style="font-size: 0.8rem; color: {recommendation_color};">{recommendation}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button("🔄", key=f"sample_reanalyze_{stock}_{idx}", help="Yeniden analiz et"):
                        st.session_state.selected_stock_for_analysis = stock
                        st.info(f"{stock} için Hisse Analizi sekmesine gidin.")
        
        # Popüler Hisseler
        st.markdown("### 🔥 Popüler Hisseler")
        
        if popular_stocks_data:
            for stock in popular_stocks_data[:5]:  # İlk 5 tanesini göster
                symbol = stock["symbol"]
                name = stock["name"]
                change_percent = stock["change_percent"]
                change_text = f"+{change_percent:.2f}%" if change_percent > 0 else f"{change_percent:.2f}%"
                change_color = "green" if change_percent > 0 else ("red" if change_percent < 0 else "gray")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"""
                    <div style="padding: 0.3rem 0;">
                        <strong>{symbol}</strong>
                        <div style="font-size: 0.8rem; color: #666;">{name[:20]}...</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div style="text-align: right; padding: 0.3rem 0;">
                        <div style="color: {change_color}; font-weight: 600;">{change_text}</div>
                        <div style="font-size: 0.8rem; color: #666;">{stock["value"]:.2f} ₺</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Popüler hisse verileri yüklenemedi.")
    
    # Tab içerikleri
    with tab1:
        render_bist100_tab()
    
    with tab2:
        render_stock_tab()
    
    with tab3:
        render_ml_prediction_tab()

if __name__ == "__main__":
    main() 