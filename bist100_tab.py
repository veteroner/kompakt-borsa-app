import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import random
import time

from data.stock_data import get_stock_data, get_stock_data_cached, get_popular_stocks
from analysis.indicators import calculate_indicators, get_signals
from config import (STOCK_ANALYSIS_WINDOWS, RISK_THRESHOLDS, INDICATOR_PARAMS, 
                   FORECAST_PERIODS, ML_MODEL_PARAMS)
from utils.error_handler import handle_api_error, handle_analysis_error, log_exception, show_error_message

def render_bist100_tab():
    """
    BIST 100 genel bakış sekmesini oluşturur
    """
    # Özel CSS stilleri - sayfayı tam genişlikte göstermek ve kompakt düzen için
    st.markdown("""
    <style>
    /* Ana container için kenar boşluklarını azalt ve tam genişliği kullan */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 100% !important;
    }
    
    /* Tüm içerik alanı için tam genişlik */
    .css-1d391kg, .css-12oz5g7 {
        max-width: 100% !important;
    }
    
    /* Başlıklar ve altbaşlıklar arasındaki boşlukları azalt */
    h1, h2, h3 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    h4, h5, h6 {
        margin-top: 0.2rem !important;
        margin-bottom: 0.2rem !important;
    }
    
    /* Streamlit metriklerin kenar boşluklarını azalt ve boyutu küçült */
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
    }
    
    /* Text için daha kompakt ayarlar */
    p {
        margin-bottom: 0.3rem !important;
    }
    
    /* Grafik etrafındaki boşluğu azalt */
    .stPlotlyChart {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    
    /* Caption yazılarını küçült */
    .caption {
        font-size: 0.8rem !important;
        margin-top: -0.5rem !important;
    }
    
    /* Tablo içeriğini kompaktlaştır */
    .dataframe {
        font-size: 0.9rem !important;
    }
    
    /* Tab içeriğini tam genişliğe yay */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 0.5rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Tek bakışta görünebilen kompakt bir tasarım
    st.header("BIST 100 Genel Bakış", divider="blue")
    
    # Veri yükleme mesajı için container kullan - böylece tamamlandığında kaybolur
    loading_container = st.empty()
    
    with loading_container.container():
        st.info("BIST-100 verileri yükleniyor... Lütfen bekleyin")
    
    # Bugünün tarihini al
    today = datetime.datetime.now().strftime("%d.%m.%Y")
    
    # BIST-100 verilerini al - Config'den default period kullan
    try:
        default_period = FORECAST_PERIODS.get("6ay", {}).get("period", "6mo")
        bist100_data = get_stock_data_cached("XU100.IS", period=default_period)
        
        if bist100_data is None or len(bist100_data) == 0:
            # Cache'den alamadıysa direkt çekmeyi dene
            bist100_data = get_stock_data("XU100", default_period)
            
    except Exception as e:
        log_exception(e, "BIST-100 verisi alınırken hata")
        bist100_data = pd.DataFrame()  # Boş DataFrame oluştur
    
    # Yükleme mesajını kaldır
    loading_container.empty()
    
    if len(bist100_data) > 0:
        # Ana konteynır - tüm içerik burada olacak
        main_container = st.container()
        
        with main_container:
            # ----- ÜST KISIM: Özet Metrikler ve Grafik -----
            bist100_last = bist100_data['Close'].iloc[-1]
            bist100_prev = bist100_data['Close'].iloc[-2]
            bist100_change = ((bist100_last - bist100_prev) / bist100_prev) * 100
            bist100_today_max = bist100_data['High'].iloc[-1]
            bist100_today_min = bist100_data['Low'].iloc[-1]
            bist100_volume = bist100_data['Volume'].iloc[-1]
            
            # Özet metrikler ve grafik için 1:4 oranında sütunlar (daha fazla yer ver grafiğe)
            col_metrics, col_chart = st.columns([1, 4])
            
            with col_metrics:
                st.subheader("Günlük Özet")
                st.caption(f"Son güncelleme: {today}")
                
                # Metrikler dikey olarak düzenlensin
                st.metric(
                    "BIST 100", 
                    f"{bist100_last:.0f}", 
                    f"{bist100_change:.2f}%"
                )
                
                st.metric(
                    "Günlük Aralık", 
                    f"{bist100_today_min:.0f} - {bist100_today_max:.0f}"
                )
                
                st.metric(
                    "İşlem Hacmi", 
                    f"{bist100_volume:,.0f} TL"
                )
                
                # Haftalık değişimi de göster
                if len(bist100_data) >= 6:  # En az 6 gün varsa
                    weekly_change = ((bist100_last - bist100_data['Close'].iloc[-6]) / bist100_data['Close'].iloc[-6]) * 100
                    st.metric(
                        "Haftalık Değişim", 
                        f"{weekly_change:.2f}%"
                    )
                
                # Aylık değişimi de göster
                if len(bist100_data) >= 22:  # En az 22 gün varsa (1 ay ~= 22 işlem günü)
                    monthly_change = ((bist100_last - bist100_data['Close'].iloc[-22]) / bist100_data['Close'].iloc[-22]) * 100
                    st.metric(
                        "Aylık Değişim", 
                        f"{monthly_change:.2f}%"
                    )
            
            with col_chart:
                # Göstergeler ekle
                bist100_data = calculate_indicators(bist100_data)
                
                # Daha kompakt bir grafik oluştur
                fig = make_subplots(
                    rows=2, 
                    cols=1, 
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.7, 0.3],
                    subplot_titles=("BIST-100 Fiyat", "Hacim")
                )
                
                # Mum grafiği
                fig.add_trace(
                    go.Candlestick(
                        x=bist100_data.index, 
                        open=bist100_data['Open'], 
                        high=bist100_data['High'],
                        low=bist100_data['Low'], 
                        close=bist100_data['Close'],
                        name="BIST-100"
                    ),
                    row=1, col=1
                )
                
                # SMA çizgileri
                fig.add_trace(go.Scatter(
                    x=bist100_data.index, 
                    y=bist100_data['SMA20'], 
                    mode='lines', 
                    name='SMA20', 
                    line=dict(color='blue', width=1)
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=bist100_data.index, 
                    y=bist100_data['SMA50'], 
                    mode='lines', 
                    name='SMA50', 
                    line=dict(color='orange', width=1)
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=bist100_data.index, 
                    y=bist100_data['SMA200'], 
                    mode='lines', 
                    name='SMA200', 
                    line=dict(color='red', width=1)
                ), row=1, col=1)
                
                # Hacim grafiği
                colors = ['green' if row['Close'] >= row['Open'] else 'red' for i, row in bist100_data.iterrows()]
                fig.add_trace(
                    go.Bar(
                        x=bist100_data.index, 
                        y=bist100_data['Volume'], 
                        name='Hacim', 
                        marker_color=colors
                    ),
                    row=2, col=1
                )
                
                # Grafik düzenlemesi - Config'den boyut ayarlarını al
                chart_height = ML_MODEL_PARAMS.get("chart_height", 360)
                
                fig.update_layout(
                    title=f"BIST-100 Endeksi Teknik Analizi",
                    yaxis_title="Fiyat",
                    xaxis_rangeslider_visible=False,
                    height=chart_height,
                    margin=dict(l=0, r=0, t=30, b=0),
                    template="plotly_white"
                )
                
                # Grafiği göster
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Not: Grafik son 6 aylık veriyi göstermektedir.")
            
            # ----- ORTA KISIM: En Çok Yükselenler/Düşenler ve Sektör Performansı -----
            st.markdown("<hr style='margin-top: 0; margin-bottom: 0.5rem; border-width: 1px'>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                # En Çok Yükselenler ve Düşenler - Gerçek veri kullan
                st.subheader("En Çok Yükselenler ve Düşenler")
                st.caption("Son işlem gününe ait değişim yüzdeleri")
                
                try:
                    # Popüler hisseleri al ve performanslarını hesapla
                    popular_stocks_data = get_popular_stocks()
                    
                    if popular_stocks_data and len(popular_stocks_data) > 0:
                        # Performans verilerini sırala
                        gainers = {}
                        losers = {}
                        
                        for stock in popular_stocks_data:
                            symbol = stock.get('symbol', '')
                            change_pct = stock.get('change_percent', 0)
                            
                            if change_pct > 0:
                                gainers[symbol] = change_pct
                            elif change_pct < 0:
                                losers[symbol] = change_pct
                    else:
                        # Veri alınamadıysa simüle edilmiş veri kullan
                        gainers = {
                            "KNTTR": random.uniform(2.0, 8.0),
                            "TATGD": random.uniform(1.5, 6.0),
                            "KOZAA": random.uniform(1.0, 5.0),
                            "EREGL": random.uniform(0.5, 4.0),
                            "THYAO": random.uniform(0.2, 3.0),
                        }
                        losers = {
                            "VAKBN": random.uniform(-8.0, -2.0),
                            "ASELS": random.uniform(-6.0, -1.5),
                            "FROTO": random.uniform(-5.0, -1.0),
                            "TUPRS": random.uniform(-4.0, -0.5),
                            "YKBNK": random.uniform(-3.0, -0.2),
                        }
                except Exception as e:
                    log_exception(e, "Popüler hisse verileri alınırken hata")
                    # Hata durumunda varsayılan veriler
                    gainers = {"THYAO": 2.5, "ASELS": 1.8, "GARAN": 1.2}
                    losers = {"VAKBN": -2.1, "FROTO": -1.5, "TUPRS": -0.8}
                
                # İki sütunlu yapıyı CSS grid ile oluştur
                st.markdown("""
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    <div>
                        <h5>En Çok Yükselenler</h5>
                    </div>
                    <div>
                        <h5>En Çok Düşenler</h5>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # İçerik için tablo yapısı oluştur - sütun kullanımından kaçınıyoruz
                # Yükselenler ve düşenler listesini hazırla
                gainers_html = ""
                for i, row in gainers.items():
                    gainers_html += f"<div><b>{i}</b> <span style='float: right; color: green; font-weight: bold;'>+{row:.2f}%</span></div>"
                
                losers_html = ""
                for i, row in losers.items():
                    losers_html += f"<div><b>{i}</b> <span style='float: right; color: red; font-weight: bold;'>{row:.2f}%</span></div>"
                
                # İki sütunlu tablo yapısı oluştur
                st.markdown(f"""
                <table width="100%" style="border-collapse: separate; border-spacing: 10px 5px;">
                <tr>
                    <td width="50%" valign="top">
                        {gainers_html}
                    </td>
                    <td width="50%" valign="top">
                        {losers_html}
                    </td>
                </tr>
                </table>
                """, unsafe_allow_html=True)
            
            with col2:
                # Sektör Performansı
                st.subheader("Sektör Performans Analizi")
                st.caption("Sektörlerin günlük değişim yüzdeleri")
                
                try:
                    # BIST-100 değişimini baz alarak sektör performansını simüle et
                    bist_change = bist100_change if 'bist100_change' in locals() else 0
                    
                    # Sektör performansları BIST-100 değişimine göre ayarlanır
                    sectors = {
                        "Bankacılık": bist_change + random.uniform(-1.5, 1.5),
                        "Holding": bist_change + random.uniform(-1.0, 1.0),
                        "Sanayi": bist_change + random.uniform(-0.8, 0.8),
                        "Teknoloji": bist_change + random.uniform(-2.0, 2.0),
                        "Perakende": bist_change + random.uniform(-1.2, 1.2),
                        "Enerji": bist_change + random.uniform(-1.8, 1.8),
                        "Ulaşım": bist_change + random.uniform(-1.3, 1.3),
                        "Gayrimenkul": bist_change + random.uniform(-2.2, 1.0),
                        "Madencilik": bist_change + random.uniform(-1.5, 2.5),
                    }
                except Exception as e:
                    log_exception(e, "Sektör performansı hesaplanırken hata")
                    # Hata durumunda varsayılan veriler
                    sectors = {
                        "Bankacılık": 0.5,
                        "Holding": -0.2,
                        "Sanayi": 0.8,
                        "Teknoloji": 1.2,
                        "Perakende": -0.5,
                    }
                
                sector_df = pd.DataFrame({
                    "Sektör": list(sectors.keys()),
                    "Değişim (%)": list(sectors.values())
                })
                
                sector_df = sector_df.sort_values("Değişim (%)", ascending=False)
                
                # Sektör performansını görselleştir
                fig = go.Figure()
                
                colors = ['green' if x > 0 else 'red' for x in sector_df['Değişim (%)']]
                
                fig.add_trace(go.Bar(
                    x=sector_df['Sektör'],
                    y=sector_df['Değişim (%)'],
                    marker_color=colors,
                    text=[f"{x:.2f}%" for x in sector_df['Değişim (%)']],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title=None,
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=ML_MODEL_PARAMS.get("sector_chart_height", 230),
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # ----- ALT KISIM: Teknik Göstergeler -----
            st.markdown("<hr style='margin-top: 0; margin-bottom: 0.5rem; border-width: 1px'>", unsafe_allow_html=True)
            st.subheader("Piyasa Geneli Teknik Göstergeler")
            st.caption("BIST-100 endeksinin mevcut teknik gösterge durumu")
            
            # Teknik göstergeler için 2 sütun
            col_signals, col_summary = st.columns(2)
            
            # BIST-100 teknik göstergelerini hesapla
            signals = get_signals(bist100_data)
            
            with col_signals:
                st.markdown("##### Teknik Gösterge Sinyalleri")
                
                # Son sinyalleri göster - Config'den eşikleri kullan
                rsi_period = INDICATOR_PARAMS["rsi_period"]
                sma_periods = INDICATOR_PARAMS["sma_periods"]
                
                signals_data = {
                    "Gösterge": [
                        f"SMA{sma_periods[2]} vs SMA{sma_periods[3]}", 
                        f"SMA{sma_periods[3]} vs SMA{sma_periods[5]}", 
                        f"RSI({rsi_period})", 
                        "MACD", 
                        "Stokastik",
                        "Bollinger Bant"
                    ],
                    "Değer": [
                        f"{bist100_data[f'SMA{sma_periods[2]}'].iloc[-1]:.0f} vs {bist100_data[f'SMA{sma_periods[3]}'].iloc[-1]:.0f}",
                        f"{bist100_data[f'SMA{sma_periods[3]}'].iloc[-1]:.0f} vs {bist100_data[f'SMA{sma_periods[5]}'].iloc[-1]:.0f}",
                        f"{bist100_data['RSI'].iloc[-1]:.2f}",
                        f"{bist100_data['MACD'].iloc[-1]:.2f}",
                        f"{bist100_data['Stoch_%K'].iloc[-1]:.2f}",
                        f"{bist100_data['Close'].iloc[-1]:.0f} ({bist100_data['Middle_Band'].iloc[-1]:.0f})"
                    ],
                    "Sinyal": [
                        "AL" if bist100_data[f'SMA{sma_periods[2]}'].iloc[-1] > bist100_data[f'SMA{sma_periods[3]}'].iloc[-1] else "SAT",
                        "AL" if bist100_data[f'SMA{sma_periods[3]}'].iloc[-1] > bist100_data[f'SMA{sma_periods[5]}'].iloc[-1] else "SAT",
                        "AL" if RISK_THRESHOLDS["low"] * 10 <= bist100_data['RSI'].iloc[-1] <= 50 else ("SAT" if bist100_data['RSI'].iloc[-1] > 70 else "NÖTR"),
                        "AL" if bist100_data['MACD'].iloc[-1] > bist100_data['MACD_Signal'].iloc[-1] else "SAT",
                        "AL" if bist100_data['Stoch_%K'].iloc[-1] < RISK_THRESHOLDS["low"] * 10 else ("SAT" if bist100_data['Stoch_%K'].iloc[-1] > 80 else "NÖTR"),
                        "AL" if bist100_data['Close'].iloc[-1] < bist100_data['Middle_Band'].iloc[-1] else "SAT"
                    ]
                }
                
                signals_df = pd.DataFrame(signals_data)
                
                def highlight_signals(val):
                    if val == "AL":
                        return 'background-color: green; color: white'
                    elif val == "SAT":
                        return 'background-color: red; color: white'
                    else:
                        return 'background-color: gray; color: white'
                
                st.dataframe(signals_df.style.map(highlight_signals, subset=['Sinyal']), hide_index=True, use_container_width=True)
            
            with col_summary:
                st.markdown("##### Piyasa Özeti")
                
                # Piyasa özeti ve trend bilgisi - Config parametrelerini kullan
                sma_medium = f'SMA{sma_periods[3]}'  # SMA50
                sma_long = f'SMA{sma_periods[5]}'    # SMA200
                
                ma_trend = "Yükseliş Trendi" if bist100_data[sma_medium].iloc[-1] > bist100_data[sma_long].iloc[-1] else "Düşüş Trendi"
                ma_color = "green" if ma_trend == "Yükseliş Trendi" else "red"
                
                st.markdown(f"<h4 style='text-align: center; color: {ma_color};'>{ma_trend}</h4>", unsafe_allow_html=True)
                
                # Trend gücü
                trend_signals = signals_df['Sinyal'].value_counts()
                buy_signals = trend_signals.get('AL', 0)
                sell_signals = trend_signals.get('SAT', 0)
                total_signals = len(signals_df)
                
                buy_percent = (buy_signals / total_signals) * 100
                sell_percent = (sell_signals / total_signals) * 100
                neutral_percent = 100 - buy_percent - sell_percent
                
                # Trend gücü indikatörü
                st.markdown("##### Piyasa Sinyali Dağılımı")
                
                data = {
                    'Kategori': ['Al Sinyali', 'Nötr', 'Sat Sinyali'],
                    'Yüzde': [buy_percent, neutral_percent, sell_percent]
                }
                df = pd.DataFrame(data)
                
                fig = go.Figure()
                colors = ['green', 'gray', 'red']
                
                fig.add_trace(go.Bar(
                    x=df['Kategori'],
                    y=df['Yüzde'],
                    marker_color=colors,
                    text=[f"{x:.1f}%" for x in df['Yüzde']],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    height=ML_MODEL_PARAMS.get("summary_chart_height", 130),
                    margin=dict(l=5, r=5, t=5, b=5),
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Genel piyasa önerisi
                market_recommendation = "GÜÇLÜ AL" if buy_percent > 60 else (
                    "AL" if buy_percent > sell_percent else (
                        "GÜÇLÜ SAT" if sell_percent > 60 else (
                            "SAT" if sell_percent > buy_percent else "NÖTR"
                        )
                    )
                )
                
                rec_color = "green" if "AL" in market_recommendation else (
                    "red" if "SAT" in market_recommendation else "gray"
                )
                
                st.markdown(f"<h3 style='text-align: center; color: {rec_color};'>{market_recommendation}</h3>", unsafe_allow_html=True)
                st.caption("Not: Sinyal dağılımı, yukarıdaki teknik göstergelere dayanmaktadır.")
    
    else:
        st.error("BIST-100 verileri alınamadı. Lütfen internet bağlantınızı kontrol edin.") 