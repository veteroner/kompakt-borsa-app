import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import traceback
import io
import logging

# Config ve error handling
from config import API_KEYS, ML_MODEL_PARAMS, FORECAST_PERIODS, STOCK_ANALYSIS_WINDOWS
from utils.error_handler import handle_api_error, handle_analysis_error, log_exception, show_error_message

# Yapay zeka modüllerini ekle
from ai.api import initialize_gemini_api
from data.ai_functions import (
    ai_market_sentiment, 
    ai_stock_analysis,
    ai_price_prediction,
    ai_sector_analysis,
    ai_portfolio_recommendation,
    ai_technical_interpretation,
    load_gemini_pro
)

# Veri ve analiz modüllerinden gerekli fonksiyonları ekle
from data.stock_data import get_stock_data, get_company_info
from data.news_data import get_stock_news
from analysis.indicators import calculate_indicators
from analysis.charts import create_stock_chart
from data.db_utils import save_analysis_result

# Loglama yapılandırması
logger = logging.getLogger(__name__)

def render_ai_tab():
    """
    Yapay Zeka sekmesini oluşturur
    """
    st.header("Yapay Zeka Analizleri", divider="rainbow")
    
    # İşlem günlüğü expander'ı - varsayılan olarak kapalı
    log_expander = st.expander("İşlem Günlüğü (Detaylar için tıklayın)", expanded=False)
    
    # Gemini API'yi başlat - Config'den API key al
    try:
        gemini_api_key = API_KEYS.get("GEMINI_API_KEY", "")
        if not gemini_api_key:
            st.error("Gemini API anahtarı yapılandırılmamış. Config dosyasını kontrol edin.")
            with log_expander:
                st.error("GEMINI_API_KEY config'de bulunamadı.")
            return
            
        gemini_pro = initialize_gemini_api()
        if gemini_pro is None:
            st.warning("Yapay zeka servisi şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin.")
            with log_expander:
                st.warning("Gemini API başlatılamadı.")
            return
            
    except Exception as e:
        log_exception(e, "Gemini API başlatılırken hata")
        st.error("Yapay zeka servisi başlatılamadı.")
        with log_expander:
            st.error(f"API başlatma hatası: {str(e)}")
        return
    
    # AI sekmeleri
    ai_tabs = st.tabs(["🔮 Piyasa Duyarlılığı", "🧠 Hisse Analizi", "📈 Fiyat Tahmini", 
                        "📊 Sektör Analizi", "💰 Portföy Önerileri", "📉 Teknik Analiz"])
    
    with ai_tabs[0]:
        st.subheader("Piyasa Genel Duyarlılığı")
        st.markdown("Bu bölümde yapay zeka, piyasanın genel durumunu analiz eder ve yatırımcı duyarlılığını değerlendirir.")
        
        if st.button("Piyasa Duyarlılığı Analizi", type="primary", key="market_sentiment"):
            try:
                # Log mesajlarını expander'a yönlendir
                with log_expander:
                    st.info("Piyasa Duyarlılığı Analizi başlatılıyor...")
                
                # Spinner yerine container kullan, böylece tüm içerik daha düzenli görünür
                result_container = st.container()
                
                # Minimal spinner
                with st.spinner(""):
                    # AI analizi - standart çağrı
                    try:
                        sentiment_result = ai_market_sentiment(gemini_pro)
                        
                        # Return değerini kontrol et - tuple mu string mi?
                        if isinstance(sentiment_result, tuple):
                            sentiment_text, sentiment_data = sentiment_result
                        else:
                            sentiment_text = sentiment_result
                            sentiment_data = None
                            
                    except Exception as ai_error:
                        log_exception(ai_error, "AI market sentiment analizi sırasında hata")
                        with log_expander:
                            st.error(f"AI analiz hatası: {str(ai_error)}")
                        sentiment_text = "Piyasa duyarlılığı analizi şu anda yapılamıyor. Lütfen daha sonra tekrar deneyin."
                        sentiment_data = None
                    
                    # Temiz bir format içinde metni göster
                    with result_container:
                        st.markdown(f"""
                        <div style="background-color:#f0f7ff; padding:15px; border-radius:10px; border-left:5px solid #0066ff; margin-top:10px;">
                        {sentiment_text}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Eğer fallback sonuçlar varsa (API bağlantısı yoksa), görselleri göster
                        if sentiment_data:
                            st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)  # Daha az boşluk
                            
                            # Sütunları tanımla
                            col1, col2, col3, col4, col5 = st.columns(5)
                            
                            # Piyasa Duyarlılığı
                            with col1:
                                try:
                                    mood = sentiment_data.get('market_mood', 'Nötr')  # Varsayılan değer 'Nötr'
                                    mood_color = "green" if mood == "Olumlu" else ("red" if mood == "Olumsuz" else "orange")
                                    st.markdown(f"<h4 style='text-align:center; color:{mood_color}; font-size:1.1em;'>{mood}</h4>", unsafe_allow_html=True)
                                    st.markdown("<p style='text-align:center; font-size:0.9em;'>Piyasa Duyarlılığı</p>", unsafe_allow_html=True)
                                except Exception as e:
                                    st.markdown("<h4 style='text-align:center; color:orange; font-size:1.1em;'>Nötr</h4>", unsafe_allow_html=True)
                                    st.markdown("<p style='text-align:center; font-size:0.9em;'>Piyasa Duyarlılığı</p>", unsafe_allow_html=True)
                                    with log_expander:
                                        st.error(f"Piyasa duyarlılığı değerinde hata: {str(e)}")
                            
                            # Güven Oranı
                            with col2:
                                try:
                                    confidence = sentiment_data.get('confidence', 75)  # Varsayılan değer 75
                                    confidence_color = "green" if confidence > 75 else ("orange" if confidence > 50 else "red")
                                    st.markdown(f"<h4 style='text-align:center; color:{confidence_color}; font-size:1.1em;'>%{confidence}</h4>", unsafe_allow_html=True)
                                    st.markdown("<p style='text-align:center; font-size:0.9em;'>Güven Oranı</p>", unsafe_allow_html=True)
                                except Exception as e:
                                    st.markdown("<h4 style='text-align:center; color:orange; font-size:1.1em;'>%75</h4>", unsafe_allow_html=True)
                                    st.markdown("<p style='text-align:center; font-size:0.9em;'>Güven Oranı</p>", unsafe_allow_html=True)
                                    with log_expander:
                                        st.error(f"Güven oranı değerinde hata: {str(e)}")
                            
                            # Trend Gücü
                            with col3:
                                try:
                                    strength = sentiment_data.get('trend_strength', 50)  # Varsayılan değer 50
                                    strength_color = "green" if strength > 70 else ("orange" if strength > 40 else "red")
                                    st.markdown(f"<h4 style='text-align:center; color:{strength_color}; font-size:1.1em;'>%{strength}</h4>", unsafe_allow_html=True)
                                    st.markdown("<p style='text-align:center; font-size:0.9em;'>Trend Gücü</p>", unsafe_allow_html=True)
                                except Exception as e:
                                    st.markdown("<h4 style='text-align:center; color:orange; font-size:1.1em;'>%50</h4>", unsafe_allow_html=True)
                                    st.markdown("<p style='text-align:center; font-size:0.9em;'>Trend Gücü</p>", unsafe_allow_html=True)
                                    with log_expander:
                                        st.error(f"Trend gücü değerinde hata: {str(e)}")
                            
                            # Beklenen Volatilite
                            with col4:
                                try:
                                    volatility = sentiment_data.get('volatility_expectation', 'Orta')  # Varsayılan değer 'Orta'
                                    volatility_color = "green" if volatility == "Düşük" else ("red" if volatility == "Yüksek" else "orange")
                                    st.markdown(f"<h4 style='text-align:center; color:{volatility_color}; font-size:1.1em;'>{volatility}</h4>", unsafe_allow_html=True)
                                    st.markdown("<p style='text-align:center; font-size:0.9em;'>Beklenen Volatilite</p>", unsafe_allow_html=True)
                                except Exception as e:
                                    st.markdown("<h4 style='text-align:center; color:orange; font-size:1.1em;'>Orta</h4>", unsafe_allow_html=True)
                                    st.markdown("<p style='text-align:center; font-size:0.9em;'>Beklenen Volatilite</p>", unsafe_allow_html=True)
                                    with log_expander:
                                        st.error(f"Volatilite değerinde hata: {str(e)}")
                            
                            # Tavsiye
                            with col5:
                                try:
                                    recommendation = sentiment_data.get('overall_recommendation', 'Tut')  # Varsayılan değer 'Tut'
                                    rec_color = "green" if recommendation == "Al" else ("red" if recommendation == "Sat" else "orange")
                                    st.markdown(f"<h4 style='text-align:center; color:{rec_color}; font-size:1.1em;'>{recommendation}</h4>", unsafe_allow_html=True)
                                    st.markdown("<p style='text-align:center; font-size:0.9em;'>Tavsiye</p>", unsafe_allow_html=True)
                                except Exception as e:
                                    st.markdown("<h4 style='text-align:center; color:orange; font-size:1.1em;'>Tut</h4>", unsafe_allow_html=True)
                                    st.markdown("<p style='text-align:center; font-size:0.9em;'>Tavsiye</p>", unsafe_allow_html=True)
                                    with log_expander:
                                        st.error(f"Tavsiye değerinde hata: {str(e)}")
            except Exception as e:
                st.error(f"Piyasa duyarlılığı analizi sırasında bir hata oluştu: {str(e)}")
                with log_expander:
                    st.error(f"Hata detayı: {str(e)}")
    
    with ai_tabs[1]:
        st.subheader("Hisse Senedi Analizi")
        st.markdown("Seçtiğiniz hisse senedi için yapay zeka detaylı analiz yapar.")
        
        # Config'den default stock al
        default_stock = ML_MODEL_PARAMS.get("default_stock", "THYAO")
        if not default_stock.endswith('.IS'):
            default_stock += '.IS'
            
        stock_symbol = st.text_input("Hisse Senedi Sembolü", value=default_stock, key="ai_stock_symbol")
        stock_symbol = stock_symbol.upper()
        if not stock_symbol.endswith('.IS') and not stock_symbol == "":
            stock_symbol += '.IS'
        
        if st.button("Hisse Analizi", type="primary", key="stock_analysis"):
            results_container = st.container()
            
            # Log mesajlarını expander'a yönlendir
            with log_expander:
                st.info(f"{stock_symbol} için yapay zeka analizi yapılıyor...")
            
            with st.spinner(""):
                try:
                    # Hisse verilerini al - Config'den period kullan
                    data_period = FORECAST_PERIODS.get("6ay", {}).get("period", "6mo")
                    stock_data = get_stock_data(stock_symbol, period=data_period)
                    
                    with log_expander:
                        st.info(f"Hisse verileri alındı ({data_period}), analiz yapılıyor...")
                    
                    if stock_data is not None and not stock_data.empty:
                        # Göstergeleri hesapla
                        try:
                            stock_data_with_indicators = calculate_indicators(stock_data)
                            with log_expander:
                                st.info("Göstergeler hesaplandı, YZ analizi başlatılıyor...")
                            # Analizi çalıştır
                            try:
                                analysis_result = ai_stock_analysis(gemini_pro, stock_symbol, stock_data_with_indicators)
                                # Return değerini kontrol et
                                if not analysis_result or analysis_result.strip() == "":
                                    analysis_result = f"{stock_symbol} için AI analizi tamamlanamadı. Lütfen daha sonra tekrar deneyin."
                            except Exception as ai_error:
                                log_exception(ai_error, "AI hisse analizi sırasında hata")
                                with log_expander:
                                    st.error(f"AI analiz hatası: {str(ai_error)}")
                                analysis_result = f"{stock_symbol} için AI analizi şu anda yapılamıyor. Temel teknik analiz sonuçları aşağıda gösterilmektedir."
                        except Exception as indicator_error:
                            log_exception(indicator_error, "Göstergeler hesaplanırken hata")
                            with log_expander:
                                st.error(f"Gösterge hesaplama hatası: {str(indicator_error)}")
                            raise
                        # Sonuçları göster - sonuçları results_container içinde göster
                        with results_container:
                            st.markdown(f"""
                            <div style="background-color:#f0f7ff; padding:15px; border-radius:10px; border-left:5px solid #0066ff; margin-top:10px;">
                            {analysis_result}
                            </div>
                            """, unsafe_allow_html=True)
                            # Grafiği göster
                            try:
                                fig = create_stock_chart(stock_data_with_indicators, stock_symbol)
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as chart_error:
                                log_exception(chart_error, "Grafik oluşturulurken hata")
                                with log_expander:
                                    st.error(f"Grafik oluşturma hatası: {str(chart_error)}")
                                st.warning("Grafik gösterilemedi, ancak analiz tamamlandı.")
                            # Analiz sonuçlarını kaydet
                            company_info = get_company_info(stock_symbol)
                            last_price = stock_data['Close'].iloc[-1]
                            price_change = (stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]) / stock_data['Close'].iloc[-2] * 100
                            # Basit trend tespiti
                            if stock_data['Close'].iloc[-1] > stock_data['SMA20'].iloc[-1]:
                                trend_direction = "Yükseliş"
                            else:
                                trend_direction = "Düşüş"
                            # Risk seviyesi tespiti - Config'den threshold kullan
                            volatility = stock_data['Close'].pct_change().std() * 100
                            if volatility > ML_MODEL_PARAMS.get("default_volatility", 3.0):
                                risk_level = "Yüksek"
                            elif volatility > (ML_MODEL_PARAMS.get("default_volatility", 3.0) / 2):
                                risk_level = "Orta"
                            else:
                                risk_level = "Düşük"
                            # Analiz sonuçlarını kaydet
                            ai_analysis_result = {
                                "symbol": stock_symbol,
                                "company_name": company_info.get("name", ""),
                                "last_price": last_price,
                                "price_change": price_change,
                                "recommendation": "AI Analiz",
                                "trend": trend_direction,
                                "risk_level": risk_level,
                                "analysis_summary": analysis_result,
                                "analysis_type": "ai"
                            }
                            save_analysis_result(
                                symbol=stock_symbol, 
                                analysis_type="ai", 
                                price=last_price, 
                                result_data=ai_analysis_result, 
                                indicators=None, 
                                notes=analysis_result
                            )
                    else:
                        with results_container:
                            st.error(f"{stock_symbol} için veri alınamadı.")
                
                except Exception as e:
                    with results_container:
                        st.error(f"Hisse analizi sırasında bir hata oluştu: {str(e)}")
                    with log_expander:
                        st.error(f"Hata detayı: {str(e)}")
    
    with ai_tabs[2]:
        st.subheader("Fiyat Tahmini")
        st.markdown("Yapay zeka, seçtiğiniz hisse senedi için kısa ve orta vadeli fiyat tahminleri yapar.")
        
        # Config'den default stock al
        default_price_stock = ML_MODEL_PARAMS.get("default_stock", "THYAO")
        if not default_price_stock.endswith('.IS'):
            default_price_stock += '.IS'
            
        price_symbol = st.text_input("Hisse Senedi Sembolü", value=default_price_stock, key="ai_price_symbol")
        price_symbol = price_symbol.upper()
        if not price_symbol.endswith('.IS') and not price_symbol == "":
            price_symbol += '.IS'
        
        if st.button("Fiyat Tahmini", type="primary", key="price_prediction"):
            with st.spinner(f"{price_symbol} için fiyat tahmini yapılıyor..."):
                try:
                    # Hisse verilerini al - Config'den period kullan
                    data_period = FORECAST_PERIODS.get("6ay", {}).get("period", "6mo")
                    price_data = get_stock_data(price_symbol, period=data_period)
                    
                    if price_data is not None and not price_data.empty:
                        # Göstergeleri hesapla
                        try:
                            price_data_with_indicators = calculate_indicators(price_data)
                            # Tahmini çalıştır
                            try:
                                prediction_result = ai_price_prediction(gemini_pro, price_symbol, price_data_with_indicators)
                                # Return değerini kontrol et - tuple mu string mi?
                                if isinstance(prediction_result, tuple):
                                    prediction_text, prediction_data = prediction_result
                                else:
                                    prediction_text = prediction_result
                                    prediction_data = None
                                # Boş result kontrolü
                                if not prediction_text or prediction_text.strip() == "":
                                    prediction_text = f"{price_symbol} için fiyat tahmini tamamlanamadı."
                            except Exception as ai_error:
                                log_exception(ai_error, "AI fiyat tahmini sırasında hata")
                                prediction_text = f"{price_symbol} için fiyat tahmini şu anda yapılamıyor."
                                prediction_data = None
                        except Exception as indicator_error:
                            log_exception(indicator_error, "Fiyat tahmini göstergeleri hesaplanırken hata")
                            st.error(f"Göstergeler hesaplanırken hata: {str(indicator_error)}")
                            return
                        # Sonuçları göster
                        st.markdown(f"""
                        <div style="background-color:#f0f7ff; padding:15px; border-radius:10px; border-left:5px solid #0066ff;">
                        {prediction_text}
                        </div>
                        """, unsafe_allow_html=True)
                        # Eğer fallback sonuçlar varsa (API bağlantısı yoksa), tahmin grafiği göster
                        if prediction_data:
                            st.subheader("Tahmin Grafiği")
                            # Tahmin verilerini hazırla
                            current_price = prediction_data['current_price']
                            future_dates = []
                            future_prices = []
                            # Config'den tahmin gün sayısını al
                            forecast_days = ML_MODEL_PARAMS.get("chart_history_days", 30)
                            target_price = prediction_data['predicted_price_30d']
                            # Gelecek tarihleri oluştur
                            last_date = price_data.index[-1]
                            for i in range(1, forecast_days + 1):
                                if isinstance(last_date, pd.Timestamp):
                                    future_date = last_date + pd.Timedelta(days=i)
                                else:
                                    future_date = datetime.now() + timedelta(days=i)
                                future_dates.append(future_date)
                            # Fiyat tahmini yap
                            for i in range(forecast_days):
                                progress = i / (forecast_days - 1)  # 0 to 1
                                # Basit doğrusal enterpolasyon
                                day_price = current_price + (target_price - current_price) * progress
                                # Rastgele dalgalanmalar ekle
                                random_factor = np.random.uniform(-1, 1) * prediction_data['confidence'] / 500
                                day_price = day_price * (1 + random_factor)
                                future_prices.append(day_price)
                            # Tahmin grafiğini oluştur
                            try:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                # Geçmiş veri - Config'den history days kullan
                                history_days = ML_MODEL_PARAMS.get("chart_history_days", 30)
                                ax.plot(price_data.index[-history_days:], price_data['Close'].iloc[-history_days:], 
                                       label='Geçmiş Veri', color='blue')
                                # Gelecek tahmin
                                ax.plot(future_dates, future_prices, label='YZ Tahmini', 
                                       color='green' if target_price > current_price else 'red', 
                                       linestyle='--')
                                # Destek ve direnç çizgileri
                                ax.axhline(y=prediction_data['support_level'], color='green', linestyle=':', 
                                          label=f"Destek: {prediction_data['support_level']:.2f}")
                                ax.axhline(y=prediction_data['resistance_level'], color='red', linestyle=':', 
                                          label=f"Direnç: {prediction_data['resistance_level']:.2f}")
                                ax.set_title(f"{price_symbol} Yapay Zeka Fiyat Tahmini")
                                ax.set_xlabel('Tarih')
                                ax.set_ylabel('Fiyat (TL)')
                                ax.grid(True, alpha=0.3)
                                ax.legend()
                                # Grafiği göster
                                st.pyplot(fig)
                                # Memory leak'i önlemek için figür'ü kapat
                                plt.close(fig)
                            except Exception as chart_error:
                                log_exception(chart_error, "Tahmin grafiği oluşturulurken hata")
                                st.warning("Tahmin grafiği gösterilemedi, ancak analiz tamamlandı.")
                                # Hata durumunda da figür'ü kapat
                                try:
                                    plt.close(fig)
                                except:
                                    pass
                        # Ana grafiği göster
                        try:
                            fig = create_stock_chart(price_data_with_indicators, price_symbol)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as chart_error:
                            log_exception(chart_error, "Ana grafik oluşturulurken hata")
                            st.warning("Grafik gösterilemedi, ancak analiz tamamlandı.")
                    else:
                        st.error(f"{price_symbol} için veri alınamadı.")
                
                except Exception as e:
                    log_exception(e, "Fiyat tahmini sırasında hata")
                    st.error(f"Fiyat tahmini sırasında bir hata oluştu: {str(e)}")
    
    with ai_tabs[3]:
        st.subheader("Sektör Analizi")
        st.markdown("Seçtiğiniz hisse senedinin bulunduğu sektör için yapay zeka analizi yapar.")
        
        # Config'den default stock al
        default_sector_stock = ML_MODEL_PARAMS.get("default_stock", "THYAO")
        if not default_sector_stock.endswith('.IS'):
            default_sector_stock += '.IS'
            
        sector_symbol = st.text_input("Hisse Senedi Sembolü", value=default_sector_stock, key="ai_sector_symbol")
        sector_symbol = sector_symbol.upper()
        if not sector_symbol.endswith('.IS') and not sector_symbol == "":
            sector_symbol += '.IS'
        
        if st.button("Sektör Analizi", type="primary", key="sector_analysis"):
            with st.spinner(f"{sector_symbol} için sektör analizi yapılıyor..."):
                try:
                    # Analizi çalıştır
                    sector_result = ai_sector_analysis(gemini_pro, sector_symbol)
                    
                    # Return değerini kontrol et - tuple mu string mi?
                    if isinstance(sector_result, tuple):
                        sector_text, sector_data = sector_result
                    else:
                        sector_text = sector_result
                        sector_data = None
                        
                    # Boş result kontrolü
                    if not sector_text or sector_text.strip() == "":
                        sector_text = f"{sector_symbol} için sektör analizi tamamlanamadı."
                    
                    # Sonuçları göster
                    st.markdown(f"""
                    <div style="background-color:#f0f7ff; padding:15px; border-radius:10px; border-left:5px solid #0066ff;">
                    {sector_text}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Eğer fallback sonuçlar varsa (API bağlantısı yoksa), görselleri göster
                    if sector_data:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            outlook = sector_data['outlook']
                            outlook_color = "green" if outlook == "Olumlu" else ("red" if outlook == "Olumsuz" else "orange")
                            st.markdown(f"<h4 style='text-align: center; color: {outlook_color};'>{outlook}</h4>", unsafe_allow_html=True)
                            st.markdown("<p style='text-align: center;'>Sektör Görünümü</p>", unsafe_allow_html=True)
                        
                        with col2:
                            strength = sector_data['strength']
                            strength_color = "green" if strength > 70 else ("orange" if strength > 40 else "red")
                            st.markdown(f"<h4 style='text-align: center; color: {strength_color};'>%{strength}</h4>", unsafe_allow_html=True)
                            st.markdown("<p style='text-align: center;'>Sektör Gücü</p>", unsafe_allow_html=True)
                        
                        with col3:
                            trend = sector_data['trend']
                            trend_color = "green" if trend == "Yükseliş" else ("red" if trend == "Düşüş" else "orange")
                            st.markdown(f"<h4 style='text-align: center; color: {trend_color};'>{trend}</h4>", unsafe_allow_html=True)
                            st.markdown("<p style='text-align: center;'>Sektör Trendi</p>", unsafe_allow_html=True)
                except Exception as e:
                    log_exception(e, "Sektör analizi sırasında hata")
                    st.error(f"Sektör analizi sırasında bir hata oluştu: {str(e)}")
                    # Fallback sonuçları göster
                    st.markdown(f"""
                    <div style="background-color:#fff7f0; padding:15px; border-radius:10px; border-left:5px solid #ff9900;">
                    {sector_symbol} için sektör analizi şu anda yapılamıyor. Lütfen daha sonra tekrar deneyin.
                    </div>
                    """, unsafe_allow_html=True)
    
    with ai_tabs[4]:
        st.subheader("Portföy Önerileri")
        st.markdown("Yapay zeka, yatırım bütçenize ve risk profilinize göre portföy önerisi oluşturur.")
        
        # Config'den min/max budget değerleri al
        min_budget = ML_MODEL_PARAMS.get("min_budget", 1000)
        max_budget = ML_MODEL_PARAMS.get("max_budget", 10000000)
        default_budget = ML_MODEL_PARAMS.get("default_budget", 10000)
        
        budget = st.number_input("Yatırım Bütçesi (TL)", 
                                min_value=min_budget, 
                                max_value=max_budget, 
                                value=default_budget, 
                                step=1000)
        
        if st.button("Portföy Önerisi", type="primary", key="portfolio_recommendation"):
            with st.spinner("Portföy önerisi oluşturuluyor..."):
                try:
                    # Öneriyi çalıştır
                    portfolio_result = ai_portfolio_recommendation(gemini_pro, budget)
                    
                    # Return değerini kontrol et
                    if not portfolio_result or portfolio_result.strip() == "":
                        portfolio_result = f"{budget:,} TL bütçe için portföy önerisi tamamlanamadı."
                    
                    # Sonuçları göster
                    st.markdown(f"""
                    <div style="background-color:#f0f7ff; padding:15px; border-radius:10px; border-left:5px solid #0066ff;">
                    {portfolio_result}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.warning("Bu öneri bir yatırım tavsiyesi niteliği taşımaz, sadece eğitim amaçlıdır.")
                    
                except Exception as e:
                    log_exception(e, "Portföy önerisi oluşturulurken hata")
                    st.error(f"Portföy önerisi oluşturulurken bir hata oluştu: {str(e)}")
                    # Fallback önerisi göster
                    st.markdown(f"""
                    <div style="background-color:#fff7f0; padding:15px; border-radius:10px; border-left:5px solid #ff9900;">
                    {budget:,} TL bütçe için portföy önerisi şu anda oluşturulamıyor. Lütfen daha sonra tekrar deneyin.
                    </div>
                    """, unsafe_allow_html=True)
    
    with ai_tabs[5]:
        st.subheader("Teknik Analiz Yorumlama")
        st.markdown("Yapay zeka, teknik göstergeleri yorumlayarak alım/satım sinyallerini değerlendirir.")
        
        # Config'den default stock al
        default_ta_stock = ML_MODEL_PARAMS.get("default_stock", "THYAO")
        if not default_ta_stock.endswith('.IS'):
            default_ta_stock += '.IS'
            
        ta_symbol = st.text_input("Hisse Senedi Sembolü", value=default_ta_stock, key="ai_ta_symbol")
        ta_symbol = ta_symbol.upper()
        if not ta_symbol.endswith('.IS') and not ta_symbol == "":
            ta_symbol += '.IS'
        
        if st.button("Teknik Analiz", type="primary", key="technical_analysis"):
            with st.spinner(f"{ta_symbol} için teknik analiz yapılıyor..."):
                try:
                    # Hisse verilerini al - Config'den period kullan
                    data_period = FORECAST_PERIODS.get("6ay", {}).get("period", "6mo")
                    ta_data = get_stock_data(ta_symbol, period=data_period)
                    
                    if ta_data is not None and not ta_data.empty:
                        # Göstergeleri hesapla
                        try:
                            ta_data_with_indicators = calculate_indicators(ta_data)
                            # Analizi çalıştır
                            try:
                                interpretation = ai_technical_interpretation(gemini_pro, ta_symbol, ta_data_with_indicators)
                                # Return değerini kontrol et
                                if not interpretation or interpretation.strip() == "":
                                    interpretation = f"{ta_symbol} için teknik analiz yorumu tamamlanamadı."
                            except Exception as ai_error:
                                log_exception(ai_error, "AI teknik analiz yorumu sırasında hata")
                                interpretation = f"{ta_symbol} için teknik analiz yorumu şu anda yapılamıyor. Temel grafik aşağıda gösterilmektedir."
                        except Exception as indicator_error:
                            log_exception(indicator_error, "Teknik analiz göstergeleri hesaplanırken hata")
                            st.error(f"Göstergeler hesaplanırken hata: {str(indicator_error)}")
                            return
                        # Sonuçları göster
                        st.markdown(f"""
                        <div style="background-color:#f0f7ff; padding:15px; border-radius:10px; border-left:5px solid #0066ff;">
                        {interpretation}
                        </div>
                        """, unsafe_allow_html=True)
                        # Grafiği göster
                        try:
                            fig = create_stock_chart(ta_data_with_indicators, ta_symbol)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as chart_error:
                            log_exception(chart_error, "Teknik analiz grafiği oluşturulurken hata")
                            st.warning("Grafik gösterilemedi, ancak analiz tamamlandı.")
                    else:
                        st.error(f"{ta_symbol} için veri alınamadı.")
                
                except Exception as e:
                    log_exception(e, "Teknik analiz sırasında hata")
                    st.error(f"Teknik analiz sırasında bir hata oluştu: {str(e)}")