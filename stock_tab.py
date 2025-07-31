import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.stock_data import get_stock_data, get_company_info, get_stock_data_cached
from analysis.indicators import calculate_indicators, get_signals
from analysis.charts import create_stock_chart, detect_chart_patterns
from data.db_utils import save_analysis_result

# Config import'ları - STOCK_ANALYSIS_WINDOWS ekledim
from config import (FORECAST_PERIODS, DEFAULT_FORECAST_PERIOD, RISK_THRESHOLDS, 
                   RECOMMENDATION_THRESHOLDS, ML_MODEL_PARAMS, STOCK_ANALYSIS_WINDOWS)
from utils.error_handler import handle_api_error, handle_analysis_error, log_exception, show_error_message
from utils.analysis_utils import calculate_risk_level, calculate_recommendation, determine_trend, generate_analysis_summary

def render_stock_tab():
    """
    Hisse analiz sekmesini oluşturur ve teknik analiz sonuçlarını gösterir
    """
    st.header("Hisse Senedi Teknik Analizi")
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    # Session state'den selected_stock_for_analysis'i kontrol et - sadece bu sekme için
    initial_stock = ""
    if 'selected_stock_for_analysis' in st.session_state and st.session_state.selected_stock_for_analysis:
        initial_stock = st.session_state.selected_stock_for_analysis
        # Değişkeni kullandıktan sonra temizle
        st.session_state.selected_stock_for_analysis = ""
    
    with col1:
        stock_symbol = st.text_input("Hisse Senedi Kodu", value=initial_stock)
    
    # Yapılandırma dosyasından tahmin süreleri ve varsayılan değerleri al
    forecast_periods = FORECAST_PERIODS
    default_forecast_period = DEFAULT_FORECAST_PERIOD
    
    with col2:
        forecast_period = st.selectbox(
            "Tahmin Süresi",
            list(forecast_periods.keys()),
            index=list(forecast_periods.keys()).index(default_forecast_period),
            help="Hissenin gelecekteki performansını tahmin etmek istediğiniz süreyi seçin"
        )
        
        # Seçilen tahmin süresine göre otomatik olarak zaman ve veri aralığını ayarla
        period = forecast_periods[forecast_period]["period"]
        interval = forecast_periods[forecast_period]["interval"]
    
    with col3:
        # Boş satır ekleyerek hizalamayı düzeltiyoruz
        st.write("")
        # Gelişmiş ayarlar için açılır kapanır menü
        with st.expander("Gelişmiş Ayarlar"):
            period = st.selectbox(
                "Zaman Aralığı",
                ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
                index=list(["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]).index(period),
                help="Analizde kullanılacak geçmiş veri miktarı"
            )
            
            interval = st.selectbox(
                "Veri Aralığı",
                ["1d", "1wk", "1mo"],
                index=list(["1d", "1wk", "1mo"]).index(interval),
                help="Verilerin hangi sıklıkta örnekleneceği (günlük, haftalık, aylık)"
            )
        
    with col4:
        # Boş satır ekleyerek hizalamayı düzeltiyoruz
        st.write("")
        refresh = st.button("Analiz Et", use_container_width=True)
    
    # Sadece buton tıklandığında veya initial stock varsa analiz yap
    analyze_stock = refresh or (initial_stock != "" and stock_symbol != "")
    
    if analyze_stock:
        with st.spinner(f"{stock_symbol} verisi alınıyor ve analiz ediliyor..."):
            try:
                # Symbol validation and formatting
                stock_symbol = stock_symbol.upper().strip()
                
                # Get stock data - pass parameters directly
                df = get_stock_data(stock_symbol, period)
                
                if len(df) == 0:
                    show_error_message("no_data")
                    return
                
                # Calculate indicators
                try:
                    df_with_indicators = calculate_indicators(df)
                except Exception as e:
                    log_exception(e, "Göstergeler hesaplanırken hata")
                    show_error_message("indicator_error")
                    return
                
                # Calculate signals
                signals = get_signals(df_with_indicators)
                
                # Şirket bilgilerini al
                try:
                    company_info = get_company_info(stock_symbol)
                except Exception as e:
                    log_exception(e, "Şirket bilgileri alınırken hata")
                    company_info = {"name": stock_symbol}
                
                # Önce gerekli hesaplamaları yapalım
                # Son fiyat değerleri
                last_price = df['Close'].iloc[-1]
                prev_close = df['Close'].iloc[-2] if len(df) > 1 else last_price
                price_change = df['Close'].pct_change().iloc[-1] * 100
                
                # Config'den window değerlerini al
                window_1w = STOCK_ANALYSIS_WINDOWS["week_1"]     # 1 hafta = 5 iş günü
                window_1m = STOCK_ANALYSIS_WINDOWS["month_1"]    # 1 ay = 21 iş günü  
                window_3m = STOCK_ANALYSIS_WINDOWS["month_3"]    # 3 ay = 63 iş günü
                window_1y = STOCK_ANALYSIS_WINDOWS["year_1"]     # 1 yıl = 252 iş günü
                
                price_change_1w = df['Close'].pct_change(window_1w).iloc[-1] * 100 if len(df) > window_1w else None
                price_change_1m = df['Close'].pct_change(window_1m).iloc[-1] * 100 if len(df) > window_1m else None
                price_change_3m = df['Close'].pct_change(window_3m).iloc[-1] * 100 if len(df) > window_3m else None
                price_change_1y = df['Close'].pct_change(window_1y).iloc[-1] * 100 if len(df) > window_1y else None
                
                # Volatilite hesaplaması
                volatility_window = STOCK_ANALYSIS_WINDOWS["volatility_window"]  # Config'den al
                volatility = df['Close'].pct_change().rolling(window=volatility_window).std().iloc[-1] * 100 if len(df) > volatility_window else None
                
                # Risk seviyesi - Config'den eşikleri kullanarak hesaplanıyor
                risk_level, risk_color = calculate_risk_level(volatility, RISK_THRESHOLDS)
                
                # Son fiyat değişimi
                price_change_text = f"%{price_change:.2f}" if price_change else "N/A"
                price_change_color = "green" if price_change and price_change > 0 else ("red" if price_change and price_change < 0 else "gray")
                
                # Trend yönü - Parametrik yaklaşım ile hesaplanıyor
                trend_info = determine_trend(df_with_indicators, ["SMA20", "SMA50", "SMA200"])
                short_term = trend_info["short_term"]
                medium_term = trend_info["medium_term"]
                long_term = trend_info["long_term"]
                trend_direction = trend_info["direction"]
                
                # Genel piyasa durumu - BIST-100 referans alınarak  
                try:
                    bist100_data = get_stock_data_cached("XU100.IS", period="1mo")
                    
                    if bist100_data is not None and not bist100_data.empty:
                        bist100_last = bist100_data['Close'].iloc[-1]
                        bist100_change_1d = ((bist100_last / bist100_data['Close'].iloc[-2]) - 1) * 100
                        bist100_change_1w = ((bist100_last / bist100_data['Close'].iloc[-6]) - 1) * 100 if len(bist100_data) >= 6 else 0
                        
                        market_mood = "yükseliş eğiliminde" if bist100_change_1d > 0 and bist100_change_1w > 0 else (
                            "düşüş eğiliminde" if bist100_change_1d < 0 and bist100_change_1w < 0 else (
                                "kararsız seyretmekte"
                            )
                        )
                        market_info = f"BIST-100 endeksi günlük %{bist100_change_1d:.2f}, haftalık %{bist100_change_1w:.2f} değişimle {market_mood}."
                    else:
                        market_info = "BIST-100 endeksine dair güncel bilgi alınamadı."
                except Exception as e:
                    market_info = "BIST-100 endeksine dair güncel bilgi alınamadı."
                
                # Hisse ile ilgili haberler - opsiyonel
                news_info = ""
                news_items = []
                try:
                    from data.news_data import get_stock_news
                    news = get_stock_news(stock_symbol, max_results=3, news_period="1w")
                    
                    if news and len(news) > 0:
                        news_info = "Son haberler: "
                        for i, item in enumerate(news[:3]):
                            news_info += f"{item.get('title', 'Haber başlığı yok')}. "
                            news_items.append({
                                "title": item.get('title', 'Haber başlığı yok'),
                                "source": item.get('source', 'Kaynak bilinmiyor'),
                                "url": item.get('link', '#'),
                                "date": item.get('published_datetime', datetime.now())
                            })
                    else:
                        news_info = f"{stock_symbol} ile ilgili son bir haftada önemli bir haber bulunamadı."
                except Exception as e:
                    # Haber alınamadığında log'a kaydet ama devam et
                    log_exception(e, f"{stock_symbol} için haber alınırken hata")
                    news_info = f"{stock_symbol} için haber bilgisi şu anda alınamıyor."
                
                # Final recommendation - Parametrik yaklaşım ile hesaplanıyor
                ma_summary = signals['Total_MA_Signal'].iloc[-1]
                osc_summary = signals['Total_Oscillator_Signal'].iloc[-1]
                total_summary = signals['Total_Signal'].iloc[-1]
                
                ma_count = len([col for col in signals.columns if ('SMA' in col or 'EMA' in col) and col.endswith('_Signal')])
                osc_count = len([col for col in signals.columns if ('RSI' in col or 'Stoch' in col or 'MACD' in col or 'Williams' in col) and col.endswith('_Signal')])
                
                # Buy/sell sinyalleri sayısını hesapla
                ma_buy_count = sum(1 for col in signals.columns if ('SMA' in col or 'EMA' in col) and col.endswith('_Signal') and signals[col].iloc[-1] > 0)
                ma_sell_count = sum(1 for col in signals.columns if ('SMA' in col or 'EMA' in col) and col.endswith('_Signal') and signals[col].iloc[-1] < 0)
                
                osc_buy_count = sum(1 for col in signals.columns if ('RSI' in col or 'Stoch' in col or 'MACD' in col or 'Williams' in col) and col.endswith('_Signal') and signals[col].iloc[-1] > 0)
                osc_sell_count = sum(1 for col in signals.columns if ('RSI' in col or 'Stoch' in col or 'MACD' in col or 'Williams' in col) and col.endswith('_Signal') and signals[col].iloc[-1] < 0)
                
                # Tavsiye hesaplama - Config parametrelerini kullan
                rec_text, rec_color = calculate_recommendation(total_summary, ma_count + osc_count, RECOMMENDATION_THRESHOLDS)
                
                # Risk değerlendirmesi
                risk_desc = "düşük riskli" if risk_level == "DÜŞÜK" else ("orta riskli" if risk_level == "ORTA" else "yüksek riskli")
                
                # Öneriye göre aksiyon belirleme
                action = ""
                if "AL" in rec_text:
                    if "GÜÇLÜ" in rec_text:
                        action = "alım için uygun görünüyor"
                    else:
                        action = "dikkatli bir şekilde alım için değerlendirilebilir"
                elif "SAT" in rec_text:
                    if "GÜÇLÜ" in rec_text:
                        action = "satış için uygun görünüyor"
                    else:
                        action = "satış düşünülebilir"
                else:
                    action = "bekleme pozisyonunda kalınması uygun olabilir"
                
                # Fiyat değişim bilgilerini bir sözlüğe topluyorum
                price_changes = {
                    "1w": price_change_1w,
                    "1m": price_change_1m,
                    "3m": price_change_3m,
                    "1y": price_change_1y
                }
                
                # Analiz özeti için yardımcı fonksiyonu kullanıyorum
                simple_analysis = generate_analysis_summary(
                    stock_symbol,
                    trend_info,
                    risk_level,
                    rec_text,
                    price_changes,
                    market_info,
                    news_info
                )
                
                # Yapay Zeka Değerlendirmesi ve Hisse Bilgileri kartını en üstte göster
                col_left, col_right = st.columns([3, 1])
                
                with col_left:
                    # Değerlendirmeyi göster
                    st.markdown(
                        f"""
                        <div style="padding: 15px; border-radius: 5px; background-color: #f0f7ff; border-left: 5px solid {rec_color};">
                        <h4 style="margin-top: 0;">Yapay Zeka Değerlendirmesi</h4>
                        <p style="font-size: 16px;">{simple_analysis}</p>
                        <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                            <div><strong>Trend:</strong> <span style="color: {'green' if trend_direction.startswith('yükseliş') else 'red' if trend_direction.startswith('düşüş') else 'orange'}">{trend_direction}</span></div>
                            <div><strong>Risk:</strong> <span style="color: {risk_color}">{risk_level}</span></div>
                            <div><strong>Öneri:</strong> <span style="color: {rec_color}">{rec_text}</span></div>
                        </div>
                        <p style="font-size: 12px; color: gray; margin-bottom: 0; margin-top: 10px;"><i>Not: Bu değerlendirme teknik analiz verilerine, BIST-100 endeks durumuna ve güncel haberlere dayanmaktadır ve bir yatırım tavsiyesi değildir.</i></p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                with col_right:
                    # Hisse bilgilerini kart olarak göster
                    st.markdown(
                        f"""
                        <div style="padding: 15px; border-radius: 5px; background-color: #f8f9fa; border: 1px solid #dee2e6;">
                        <h4 style="margin-top: 0; text-align: center;">{stock_symbol} Bilgileri</h4>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <div><strong>Son Fiyat:</strong></div>
                            <div style="color: {price_change_color};">{last_price:.2f} TL ({price_change:.2f}%)</div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <div><strong>Önceki Kapanış:</strong></div>
                            <div>{prev_close:.2f} TL</div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <div><strong>Volatilite (20g):</strong></div>
                            <div>{f"{volatility:.2f}%" if volatility is not None else "N/A"}</div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <div><strong>1 Haftalık:</strong></div>
                            <div style="color: {'green' if price_change_1w and price_change_1w > 0 else 'red'};">{f"{price_change_1w:.2f}" if price_change_1w is not None else 'N/A'}{" %" if price_change_1w is not None else ""}</div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <div><strong>1 Aylık:</strong></div>
                            <div style="color: {'green' if price_change_1m and price_change_1m > 0 else 'red'};">{f"{price_change_1m:.2f}" if price_change_1m is not None else 'N/A'}{" %" if price_change_1m is not None else ""}</div>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <div><strong>3 Aylık:</strong></div>
                            <div style="color: {'green' if price_change_3m and price_change_3m > 0 else 'red'};">{f"{price_change_3m:.2f}" if price_change_3m is not None else 'N/A'}{" %" if price_change_3m is not None else ""}</div>
                        </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Create chart
                fig = create_stock_chart(df_with_indicators, stock_symbol)
                
                # Display chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Volume analysis
                volume_window = STOCK_ANALYSIS_WINDOWS["volume_window"]  # Config'den al
                avg_volume_20d = df['Volume'].rolling(window=volume_window).mean().iloc[-1] if len(df) > volume_window else None
                last_volume = df['Volume'].iloc[-1]
                volume_change = (last_volume / avg_volume_20d - 1) * 100 if avg_volume_20d else None
                
                # Price range - 52 haftalık yüksek/düşük
                week_52_window = STOCK_ANALYSIS_WINDOWS["week_52"]  # Config'den al
                high_52w = df['High'].rolling(window=week_52_window).max().iloc[-1] if len(df) > week_52_window else df['High'].max()
                low_52w = df['Low'].rolling(window=week_52_window).min().iloc[-1] if len(df) > week_52_window else df['Low'].min()
                
                # Additional metrics
                col1, col2, col3, col4 = st.columns(4)
                
                if price_change_1y is not None:
                    col1.metric("1 Yıllık Değişim", f"{price_change_1y:.2f}%")
                
                col2.metric("52H En Yüksek", f"{high_52w:.2f} TL")
                col3.metric("52H En Düşük", f"{low_52w:.2f} TL")
                
                if volume_change is not None:
                    vol_delta = f"{volume_change:.2f}%"
                    col4.metric("Hacim (20g Ort.)", f"{int(last_volume):,}", vol_delta)
                
                # Technical indicators
                st.subheader("Teknik Göstergeler")
                
                # Create three columns
                col1, col2, col3 = st.columns(3)
                
                # Moving Averages
                with col1:
                    st.markdown("##### Hareketli Ortalamalar")
                    ma_data = {
                        "Gösterge": ["SMA(5)", "SMA(10)", "SMA(20)", "SMA(50)", "SMA(100)", "SMA(200)",
                                  "EMA(5)", "EMA(10)", "EMA(20)", "EMA(50)", "EMA(100)", "EMA(200)"],
                        "Değer": [
                            f"{df_with_indicators['SMA5'].iloc[-1]:.2f}",
                            f"{df_with_indicators['SMA10'].iloc[-1]:.2f}",
                            f"{df_with_indicators['SMA20'].iloc[-1]:.2f}",
                            f"{df_with_indicators['SMA50'].iloc[-1]:.2f}",
                            f"{df_with_indicators['SMA100'].iloc[-1]:.2f}",
                            f"{df_with_indicators['SMA200'].iloc[-1]:.2f}",
                            f"{df_with_indicators['EMA5'].iloc[-1]:.2f}",
                            f"{df_with_indicators['EMA10'].iloc[-1]:.2f}",
                            f"{df_with_indicators['EMA20'].iloc[-1]:.2f}",
                            f"{df_with_indicators['EMA50'].iloc[-1]:.2f}",
                            f"{df_with_indicators['EMA100'].iloc[-1]:.2f}",
                            f"{df_with_indicators['EMA200'].iloc[-1]:.2f}"
                        ],
                        "Sinyal": [
                            "AL" if signals['SMA5_Signal'].iloc[-1] > 0 else "SAT",
                            "AL" if signals['SMA10_Signal'].iloc[-1] > 0 else "SAT",
                            "AL" if signals['SMA20_Signal'].iloc[-1] > 0 else "SAT",
                            "AL" if signals['SMA50_Signal'].iloc[-1] > 0 else "SAT",
                            "AL" if signals['SMA100_Signal'].iloc[-1] > 0 else "SAT",
                            "AL" if signals['SMA200_Signal'].iloc[-1] > 0 else "SAT",
                            "AL" if signals['EMA5_Signal'].iloc[-1] > 0 else "SAT",
                            "AL" if signals['EMA10_Signal'].iloc[-1] > 0 else "SAT",
                            "AL" if signals['EMA20_Signal'].iloc[-1] > 0 else "SAT",
                            "AL" if signals['EMA50_Signal'].iloc[-1] > 0 else "SAT",
                            "AL" if signals['EMA100_Signal'].iloc[-1] > 0 else "SAT",
                            "AL" if signals['EMA200_Signal'].iloc[-1] > 0 else "SAT"
                        ]
                    }
                    
                    ma_df = pd.DataFrame(ma_data)
                    
                    # Analiz sonuçlarını kaydet
                    analysis_result = {
                        "symbol": stock_symbol,
                        "company_name": company_info.get("name", ""),
                        "last_price": last_price,
                        "price_change": price_change,
                        "recommendation": rec_text,
                        "trend": trend_direction,
                        "risk_level": risk_level,
                        "ma_signals": {
                            "buy": ma_buy_count,
                            "sell": ma_sell_count,
                            "total": ma_count
                        },
                        "oscillator_signals": {
                            "buy": osc_buy_count,
                            "sell": osc_sell_count,
                            "total": osc_count
                        },
                        "moving_averages": {
                            "SMA5": df_with_indicators['SMA5'].iloc[-1],
                            "SMA20": df_with_indicators['SMA20'].iloc[-1],
                            "SMA50": df_with_indicators['SMA50'].iloc[-1],
                            "SMA200": df_with_indicators['SMA200'].iloc[-1]
                        },
                        "oscillators": {
                            "RSI": df_with_indicators['RSI'].iloc[-1],
                            "MACD": df_with_indicators['MACD'].iloc[-1],
                            "Stochastic": df_with_indicators['Stoch_%K'].iloc[-1]
                        },
                        "price_history": {
                            "1w": price_change_1w,
                            "1m": price_change_1m,
                            "3m": price_change_3m,
                            "1y": price_change_1y
                        },
                        "support_resistance": {
                            "support1": low_52w,
                            "resistance1": high_52w
                        },
                        "analysis_summary": simple_analysis,
                        "news": news_items
                    }
                    
                    # Analiz sonuçlarını kaydet
                    save_analysis_result(
                        symbol=stock_symbol, 
                        analysis_type="teknik", 
                        price=last_price, 
                        result_data=analysis_result, 
                        indicators=None, 
                        notes=simple_analysis
                    )
                    
                    def color_ma_cells(val):
                        if val == "AL":
                            return 'background-color: green; color: white'
                        elif val == "SAT":
                            return 'background-color: red; color: white'
                        return ''
                    
                    st.dataframe(ma_df.style.map(color_ma_cells, subset=['Sinyal']), hide_index=True, use_container_width=True)
                
                # Oscillators
                with col2:
                    st.markdown("##### Osilatörler")
                    
                    osc_data = {
                        "Gösterge": ["RSI(14)", "MACD", "Stochastic %K", "Stochastic %D", "Williams %R", "CCI(20)", "ATR(14)"],
                        "Değer": [
                            f"{df_with_indicators['RSI'].iloc[-1]:.2f}",
                            f"{df_with_indicators['MACD'].iloc[-1]:.4f}",
                            f"{df_with_indicators['Stoch_%K'].iloc[-1]:.2f}",
                            f"{df_with_indicators['Stoch_%D'].iloc[-1]:.2f}",
                            f"{df_with_indicators['Williams_%R'].iloc[-1]:.2f}",
                            f"{df_with_indicators.get('CCI', pd.Series([0])).iloc[-1]:.2f}" if 'CCI' in df_with_indicators else "N/A",
                            f"{df_with_indicators.get('ATR', pd.Series([0])).iloc[-1]:.4f}" if 'ATR' in df_with_indicators else "N/A"
                        ],
                        "Sinyal": [
                            "AL" if signals['RSI_Signal'].iloc[-1] > 0 else ("SAT" if signals['RSI_Signal'].iloc[-1] < 0 else "NÖTR"),
                            "AL" if signals['MACD_Signal'].iloc[-1] > 0 else "SAT",
                            "AL" if signals['Stoch_Signal'].iloc[-1] > 0 else ("SAT" if signals['Stoch_Signal'].iloc[-1] < 0 else "NÖTR"),
                            "AL" if signals['Stoch_Signal'].iloc[-1] > 0 else ("SAT" if signals['Stoch_Signal'].iloc[-1] < 0 else "NÖTR"),
                            "AL" if signals['Williams_%R_Signal'].iloc[-1] > 0 else ("SAT" if signals['Williams_%R_Signal'].iloc[-1] < 0 else "NÖTR"),
                            "NÖTR",
                            "NÖTR"
                        ]
                    }
                    
                    osc_df = pd.DataFrame(osc_data)
                    
                    def color_signal(val):
                        if val == "AL":
                            return 'background-color: green; color: white'
                        elif val == "SAT":
                            return 'background-color: red; color: white'
                        return 'background-color: gray; color: white'
                    
                    st.dataframe(osc_df.style.map(color_signal, subset=['Sinyal']), hide_index=True, use_container_width=True)
                
                # Summary
                with col3:
                    st.markdown("##### Analiz Sonucu")
                    
                    summary_data = {
                        "Kategori": ["Hareketli Ortalamalar", "Osilatörler", "Toplam Analiz"],
                        "Al Sinyali": [f"{ma_buy_count}/{ma_count}", f"{osc_buy_count}/{osc_count}", f"{ma_buy_count + osc_buy_count}/{ma_count + osc_count}"],
                        "Sat Sinyali": [f"{ma_sell_count}/{ma_count}", f"{osc_sell_count}/{osc_count}", f"{ma_sell_count + osc_sell_count}/{ma_count + osc_count}"],
                        "Sonuç": [
                            "GÜÇLÜ AL" if ma_summary > ma_count * 0.6 else ("AL" if ma_summary > 0 else ("GÜÇLÜ SAT" if ma_summary < -ma_count * 0.6 else ("SAT" if ma_summary < 0 else "NÖTR"))),
                            "GÜÇLÜ AL" if osc_summary > osc_count * 0.6 else ("AL" if osc_summary > 0 else ("GÜÇLÜ SAT" if osc_summary < -osc_count * 0.6 else ("SAT" if osc_summary < 0 else "NÖTR"))),
                            "GÜÇLÜ AL" if total_summary > (ma_count + osc_count) * 0.6 else ("AL" if total_summary > 0 else ("GÜÇLÜ SAT" if total_summary < -(ma_count + osc_count) * 0.6 else ("SAT" if total_summary < 0 else "NÖTR")))
                        ]
                    }
                    
                    summary_df = pd.DataFrame(summary_data)
                    
                    def get_signal_color(val):
                        if val == "GÜÇLÜ AL":
                            return 'background-color: darkgreen; color: white'
                        elif val == "AL":
                            return 'background-color: green; color: white'
                        elif val == "GÜÇLÜ SAT":
                            return 'background-color: darkred; color: white'
                        elif val == "SAT":
                            return 'background-color: red; color: white'
                        return 'background-color: gray; color: white'
                    
                    st.dataframe(summary_df.style.map(get_signal_color, subset=['Sonuç']), hide_index=True, use_container_width=True)
                    
                    # Final recommendation
                    rec_text = summary_df['Sonuç'].iloc[2]
                    rec_color = "green" if "AL" in rec_text else ("red" if "SAT" in rec_text else "gray")
                    st.markdown(f"<h3 style='text-align: center; color: {rec_color};'>{rec_text}</h3>", unsafe_allow_html=True)
                    
                    # Risk level - Config'den eşikleri kullan
                    risk_level_display = "YÜKSEK" if volatility and volatility > RISK_THRESHOLDS["medium"] else (
                        "ORTA" if volatility and volatility > RISK_THRESHOLDS["low"] else "DÜŞÜK"
                    )
                    risk_color_display = "red" if risk_level_display == "YÜKSEK" else (
                        "orange" if risk_level_display == "ORTA" else "green"
                    )
                    st.markdown(f"<p style='text-align: center;'>Risk Seviyesi: <span style='color: {risk_color_display};'>{risk_level_display}</span></p>", unsafe_allow_html=True)
                
                # Analyze chart patterns
                st.subheader("Grafik Desenleri ve Destek/Direnç Seviyeleri")
                
                patterns = detect_chart_patterns(df)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Destek ve Direnç Seviyeleri")
                    
                    levels_data = {
                        "Tip": ["Son Kapanış"],
                        "Değer": [f"{patterns['last_close']:.2f}"],
                        "Uzaklık (%)": ["0.00%"]
                    }
                    
                    # Add support levels
                    for i, level in enumerate(patterns['support_levels']):
                        distance = ((level / patterns['last_close']) - 1) * 100
                        levels_data["Tip"].append(f"Destek {i+1}")
                        levels_data["Değer"].append(f"{level:.2f}")
                        levels_data["Uzaklık (%)"].append(f"{distance:.2f}%")
                    
                    # Add resistance levels
                    for i, level in enumerate(patterns['resistance_levels']):
                        distance = ((level / patterns['last_close']) - 1) * 100
                        levels_data["Tip"].append(f"Direnç {i+1}")
                        levels_data["Değer"].append(f"{level:.2f}")
                        levels_data["Uzaklık (%)"].append(f"{distance:.2f}%")
                    
                    # Create dataframe
                    levels_df = pd.DataFrame(levels_data)
                    
                    # Apply color coding
                    def color_levels(val, col_name):
                        if col_name == 'Tip':
                            if "Destek" in val:
                                return 'background-color: green; color: white'
                            elif "Direnç" in val:
                                return 'background-color: red; color: white'
                            elif "Kapanış" in val:
                                return 'background-color: blue; color: white'
                        return ''
                    
                    st.dataframe(levels_df.style.apply(lambda x: [color_levels(val, col_name) for val, col_name in zip(x, levels_df.columns)], axis=1), hide_index=True, use_container_width=True)
                
                with col2:
                    st.markdown("##### Fibonacci Seviyeleri")
                    
                    fib_levels = patterns['fibonacci_levels']
                    fib_data = {
                        "Fibonacci Seviyesi": list(fib_levels.keys()),
                        "Fiyat": [f"{level:.2f}" for level in fib_levels.values()],
                        "Uzaklık (%)": [f"{((level / patterns['last_close']) - 1) * 100:.2f}%" for level in fib_levels.values()]
                    }
                    
                    fib_df = pd.DataFrame(fib_data)
                    
                    def color_fib_levels(val):
                        if "0.0" in val:
                            return 'background-color: gray; color: white'
                        elif "0.236" in val:
                            return 'background-color: #7986CB; color: white'
                        elif "0.382" in val:
                            return 'background-color: #5C6BC0; color: white'
                        elif "0.5" in val:
                            return 'background-color: #3F51B5; color: white'
                        elif "0.618" in val:
                            return 'background-color: #3949AB; color: white'
                        elif "0.786" in val:
                            return 'background-color: #303F9F; color: white'
                        elif "1.0" in val:
                            return 'background-color: #283593; color: white'
                        return ''
                    
                    st.dataframe(fib_df.style.map(color_fib_levels, subset=['Fibonacci Seviyesi']), hide_index=True, use_container_width=True)
                
                # Pattern Detection
                st.markdown("##### Tespit Edilen Desenler")
                
                detected_patterns = []
                
                if patterns.get('double_top', False):
                    detected_patterns.append(("Çift Tepe", "Düşüş Sinyali"))
                
                if patterns.get('double_bottom', False):
                    detected_patterns.append(("Çift Dip", "Yükseliş Sinyali"))
                    
                if patterns.get('head_and_shoulders', False):
                    detected_patterns.append(("Baş ve Omuzlar", "Düşüş Sinyali"))
                    
                if patterns.get('inverse_head_and_shoulders', False):
                    detected_patterns.append(("Ters Baş ve Omuzlar", "Yükseliş Sinyali"))
                    
                if patterns.get('uptrend_channel', False):
                    detected_patterns.append(("Yükseliş Kanalı", "Devam Trendi"))
                    
                if patterns.get('downtrend_channel', False):
                    detected_patterns.append(("Düşüş Kanalı", "Devam Trendi"))
                    
                if patterns.get('triangle_ascending', False):
                    detected_patterns.append(("Yükselen Üçgen", "Yükseliş Sinyali"))
                    
                if patterns.get('triangle_descending', False):
                    detected_patterns.append(("Alçalan Üçgen", "Düşüş Sinyali"))
                    
                if patterns.get('triangle_symmetrical', False):
                    detected_patterns.append(("Simetrik Üçgen", "Kırılma Beklentisi"))
                
                if detected_patterns:
                    pattern_data = {
                        "Desen": [p[0] for p in detected_patterns],
                        "Sinyal Tipi": [p[1] for p in detected_patterns],
                        "Güvenilirlik": ["Yüksek" if "Çift" in p[0] or "Baş" in p[0] else "Orta" for p in detected_patterns]
                    }
                    
                    pattern_df = pd.DataFrame(pattern_data)
                    
                    def color_pattern_signal(val):
                        if "Yükseliş" in val:
                            return 'background-color: green; color: white'
                        elif "Düşüş" in val:
                            return 'background-color: red; color: white'
                        else:
                            return 'background-color: gray; color: white'
                    
                    st.dataframe(pattern_df.style.map(color_pattern_signal, subset=['Sinyal Tipi']), hide_index=True)
                    
                    # Visualize patterns
                    for pattern_name, _ in detected_patterns:
                        st.markdown(f"**{pattern_name} Deseni Görselleştirmesi:**")
                        # Burada desen görselleştirmesi eklenebilir
                else:
                    st.markdown("Belirgin bir desen tespit edilemedi.")
                    
                # Trend Analysis
                st.subheader("Trend Analizi")
                
                # Trend bilgilerini kullanarak tablo oluştur
                trend_data = {
                    "Dönem": ["Kısa Vadeli (20 gün)", "Orta Vadeli (50 gün)", "Uzun Vadeli (200 gün)"],
                    "Trend": [trend_info["short_term"], trend_info["medium_term"], trend_info["long_term"]],
                    "Güç": [
                        f"{trend_info['short_term_strength']:.2f}%",
                        f"{trend_info['medium_term_strength']:.2f}%",
                        f"{trend_info['long_term_strength']:.2f}%"
                    ]
                }
                
                trend_df = pd.DataFrame(trend_data)
                
                def color_trend(val):
                    if val == "Yükseliş":
                        return 'background-color: green; color: white'
                    elif val == "Düşüş":
                        return 'background-color: red; color: white'
                    return ''
                
                st.dataframe(trend_df.style.map(color_trend, subset=['Trend']), hide_index=True)
                
                # Genel trend durumunu göster
                st.markdown(f"<h3 style='text-align: center; color: {trend_info['color']};'>{trend_info['overall']}</h3>", unsafe_allow_html=True)
                
            except Exception as e:
                error_trace = log_exception(e, "Analiz sırasında beklenmeyen hata")
                show_error_message("analysis_error", str(e))
                st.error(error_trace)
                
    else:
        st.info("Hisse senedi kodunu girin ve 'Analiz Et' butonuna tıklayın.")
        
        # Örnek metin
        st.markdown("### Örnek Teknik Analiz Gösterimi")
        st.markdown("""
        Hisse senedi analizi grafiği, teknik göstergeler ve destek/direnç seviyeleri burada görüntülenecektir.
        Yukarıdaki alana bir hisse senedi kodu girin ve 'Analiz Et' butonuna tıklayın.
        """)