import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ana proje dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.stock_data import get_stock_data, get_all_bist_stocks, get_stock_list, get_company_info
from analysis.indicators import calculate_indicators, get_signals
from analysis.charts import detect_chart_patterns
from utils.analysis_utils import determine_trend, calculate_risk_level, calculate_recommendation
from ai.api import ai_stock_analysis, initialize_gemini_api
try:
    from ai.technical_indicators import calculate_advanced_indicators
except ImportError:
    def calculate_advanced_indicators(df):
        return df
try:
    from ai.predictions import predict_future_price
except ImportError:
    def predict_future_price(symbol, data):
        return None
from config import RISK_THRESHOLDS, RECOMMENDATION_THRESHOLDS, STOCK_ANALYSIS_WINDOWS

def render_comprehensive_scanner_tab():
    """Kapsamlı Hisse Tarayıcı sekmesi"""
    
    st.title("🔍 Kapsamlı Hisse Tarayıcı")
    st.markdown("""
    Bu sayfa hisse analizi sayfasındaki **TÜM** özellikleri kullanarak BIST hisselerini tarar:
    
    🧠 **Yapay Zeka Analizi** • 📈 **Trend Analizi** • ⚡ **Momentum Göstergeleri** • 📊 **Hacim Analizi** 
    • 🎯 **Formasyonlar** • 📐 **Fibonacci Seviyeleri** • 🎪 **Chart Patterns** • 💰 **Risk Analizi**
    """)
    
    # Tarama ayarları
    with st.expander("🎯 Tarama Ayarları", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            market_selection = st.selectbox(
                "Piyasa Seçimi",
                ["BIST 30", "BIST 50", "BIST 100", "Tüm BIST"],
                index=1,
                help="Hangi piyasa endeksini tarayacağınızı seçin"
            )
        
        with col2:
            min_score = st.slider(
                "Minimum Skor",
                min_value=10,
                max_value=80,
                value=25,
                step=5,
                help="Bu skorun üzerindeki hisseler gösterilir"
            )
        
        with col3:
            analysis_depth = st.selectbox(
                "Analiz Derinliği",
                ["Hızlı", "Orta", "Detaylı"],
                index=2,
                help="Detaylı: Yapay zeka + tüm özellikler, Orta: AI analizi, Hızlı: Temel analiz"
            )
    
    # Filtreleme kriterleri
    with st.expander("⚙️ Filtreleme Kriterleri"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📊 Analiz Modülleri**")
            include_trend = st.checkbox("📈 Trend Analizi", value=True)
            include_momentum = st.checkbox("⚡ Momentum Göstergeleri", value=True)
            include_volume = st.checkbox("📊 Hacim Analizi", value=True)
            include_ai = st.checkbox("🧠 Yapay Zeka Analizi", value=True, help="Orta ve Detaylı seviyelerde otomatik aktif")
        
        with col2:
            st.markdown("**💰 Risk & Volatilite**")
            max_volatility = st.slider("Maksimum Volatilite (%)", 10, 60, 40, 5)
            min_liquidity = st.selectbox("Minimum Likidite", ["Düşük", "Orta", "Yüksek"], index=1)
            include_patterns = st.checkbox("🎯 Formasyon Analizi", value=True, help="Chart patterns ve formasyonlar")
            
        with col3:
            st.markdown("**📐 Teknik Seviyeler**")
            min_volume_ratio = st.slider("Minimum Hacim Oranı", 0.3, 2.5, 0.6, 0.1)
            rsi_range = st.slider("RSI Aralığı", 10, 90, (15, 85))
            include_fibonacci = st.checkbox("📐 Fibonacci Seviyeleri", value=True, help="Fibonacci retracement seviyeleri")
    
    # Tarama butonu
    if st.button("🚀 Kapsamlı Tarama Başlat", type="primary"):
        run_comprehensive_scan(
            market_selection, min_score, analysis_depth,
            include_trend, include_momentum, include_volume, include_ai,
            include_patterns, include_fibonacci, max_volatility, 
            min_liquidity, min_volume_ratio, rsi_range
        )
    
    # Sonuçları göster
    if 'scan_results' in st.session_state and st.session_state.scan_results:
        display_scan_results()
    else:
        st.info("Tarama yapmak için yukarıdaki ayarları yapılandırın ve 'Kapsamlı Tarama Başlat' butonuna tıklayın.")

def run_comprehensive_scan(market_selection, min_score, analysis_depth,
                          include_trend, include_momentum, include_volume, include_ai,
                          include_patterns, include_fibonacci, max_volatility, 
                          min_liquidity, min_volume_ratio, rsi_range):
    """Kapsamlı tarama işlemini çalıştır"""
    
    # Taranacak hisseleri belirle
    stocks_to_scan = get_stocks_by_market(market_selection)
    
    if not stocks_to_scan:
        st.error("Taranacak hisse bulunamadı!")
        return
    
    st.info(f"🔍 {len(stocks_to_scan)} hisse taranıyor...")
    
    # Debug: Taranacak hisseleri göster
    with st.expander("🔍 Debug: Taranacak Hisseler"):
        st.write(f"Toplam hisse sayısı: {len(stocks_to_scan)}")
        st.write(f"İlk 10 hisse: {stocks_to_scan[:10]}")
        st.write(f"Seçilen piyasa: {market_selection}")
        st.write(f"Minimum skor: {min_score}")
        st.write(f"Analiz derinliği: {analysis_depth}")
        st.write(f"Tüm hisseler: {', '.join(stocks_to_scan)}")
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Sonuçları saklamak için liste
    results = []
    
    # Paralel tarama için thread pool
    max_workers = 5 if analysis_depth == "Hızlı" else (3 if analysis_depth == "Orta" else 2)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Tüm tarama işlemlerini başlat
        future_to_stock = {
            executor.submit(
                analyze_single_stock, 
                stock, 
                analysis_depth,
                include_trend, include_momentum, include_volume, include_ai,
                include_patterns, include_fibonacci, max_volatility, 
                min_liquidity, min_volume_ratio, rsi_range
            ): stock for stock in stocks_to_scan
        }
        
        completed = 0
        for future in as_completed(future_to_stock):
            stock = future_to_stock[future]
            try:
                result = future.result(timeout=30)  # 30 saniye timeout
                if result:  # Tüm sonuçları topla, daha sonra filtrele
                    results.append(result)
                
                completed += 1
                progress = completed / len(stocks_to_scan)
                progress_bar.progress(progress)
                status_text.text(f"İşlenen: {completed}/{len(stocks_to_scan)} - Son: {stock}")
                
            except Exception as e:
                print(f"Debug: {stock} analiz edilemedi: {str(e)}")  # Debug için
                completed += 1
                continue
    
    # Sonuçları skora göre sırala
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Minimum skora göre filtrele
    filtered_results = [r for r in results if r['score'] >= min_score]
    
    # Session state'e kaydet (filtrelenmiş sonuçları)
    st.session_state.scan_results = filtered_results
    st.session_state.all_scan_results = results  # Tüm sonuçları da sakla
    
    progress_bar.empty()
    status_text.empty()
    
    if filtered_results:
        st.success(f"✅ Tarama tamamlandı! {len(filtered_results)} adet potansiyel hisse bulundu (Toplam {len(results)} hisse analiz edildi).")
    else:
        st.warning(f"❌ {min_score} skor üzerinde hisse bulunamadı. Toplam {len(results)} hisse analiz edildi.")
        if results:
            best_score = max(r['score'] for r in results)
            st.info(f"💡 En yüksek skor: {best_score:.1f}. Minimum skoru {best_score:.0f}'a düşürmeyi deneyin.")
            
            # En iyi 5 sonucu göster
            st.write("**En İyi 5 Sonuç:**")
            for i, result in enumerate(results[:5], 1):
                st.write(f"{i}. {result['symbol']} - Skor: {result['score']:.1f}")
        else:
            st.error("Hiç hisse analiz edilemedi. Lütfen ayarları kontrol edin.")

def get_stocks_by_market(market_selection):
    """Piyasa seçimine göre hisse listesini döndür"""
    
    try:
        if market_selection == "BIST 30":
            bist_stocks = get_stock_list("BIST 30")
            stocks = [stock.replace('.IS', '') for stock in bist_stocks]
            print(f"Debug: BIST 30 - {len(stocks)} hisse bulundu")
        elif market_selection == "BIST 50":
            bist_stocks = get_stock_list("BIST 50")
            stocks = [stock.replace('.IS', '') for stock in bist_stocks]
            print(f"Debug: BIST 50 - {len(stocks)} hisse bulundu")
        elif market_selection == "BIST 100":
            bist_stocks = get_stock_list("BIST 100")
            stocks = [stock.replace('.IS', '') for stock in bist_stocks]
            print(f"Debug: BIST 100 - {len(stocks)} hisse bulundu")
        else:  # Tüm BIST
            bist_stocks = get_all_bist_stocks()
            stocks = bist_stocks
            print(f"Debug: Tüm BIST - {len(stocks)} hisse bulundu")
        
        print(f"Debug: {len(stocks)} hisse taranacak")
        return stocks
        
    except Exception as e:
        print(f"Debug: Hisse listesi alınırken hata: {str(e)}")
        # Fallback liste - BIST 30 benzeri
        fallback_stocks = [
            "AEFES", "AKBNK", "ARCLK", "ASELS", "BIMAS", "EREGL", "FROTO", 
            "GARAN", "HALKB", "ISCTR", "KCHOL", "KOZAL", "KOZAA", "MGROS",
            "OTKAR", "PGSUS", "SAHOL", "SASA", "SISE", "TCELL", "THYAO",
            "TKFEN", "TOASO", "TUPRS", "ULKER", "VAKBN", "YKBNK", "ZRGYO"
        ]
        return fallback_stocks

def analyze_single_stock(symbol, analysis_depth, include_trend, include_momentum, 
                        include_volume, include_ai, include_patterns, include_fibonacci,
                        max_volatility, min_liquidity, min_volume_ratio, rsi_range):
    """Tek bir hisseyi kapsamlı analiz et - Hisse analizi sayfasındaki TÜM özellikleri kullanır"""
    
    try:
        # Veri al
        period = "2y" if analysis_depth == "Detaylı" else ("1y" if analysis_depth == "Orta" else "6mo")
        stock_data = get_stock_data(f"{symbol}.IS", period=period)
        
        if stock_data is None or len(stock_data) < 50:
            return None
        
        # Teknik göstergeleri hesapla
        df_with_indicators = calculate_indicators(stock_data)
        if df_with_indicators is None or len(df_with_indicators) == 0:
            return None
        
        # Gelişmiş teknik göstergeleri hesapla
        try:
            df_with_indicators = calculate_advanced_indicators(df_with_indicators)
        except:
            pass  # Gelişmiş göstergeler başarısız olursa devam et
        
        # Sinyalleri hesapla
        signals = get_signals(df_with_indicators)
        
        # Formasyonları tespit et
        try:
            chart_patterns = detect_chart_patterns(df_with_indicators)
        except:
            chart_patterns = {}
        
        # Son veriyi al
        latest = df_with_indicators.iloc[-1]
        
        # Temel bilgiler
        current_price = latest['Close']
        volume = latest.get('Volume', 0)
        
        # Skorlama sistemi - 1000 puan üzerinden daha detaylı
        score = 0
        max_possible_score = 0
        analysis_details = {}
        
        # 1. Trend Analizi (200 puan) - Daha kapsamlı
        if include_trend:
            max_possible_score += 200
            sma_columns = ['SMA20', 'SMA50', 'SMA200']
            trend_info = determine_trend(df_with_indicators, sma_columns)
            
            # Kısa vadeli trend (80 puan)
            if trend_info['short_term'] == "Yükseliş":
                score += 80
            elif trend_info['short_term'] == "Nötr":
                score += 40
            else:
                score += 10
                
            # Orta vadeli trend (70 puan)
            if trend_info['medium_term'] == "Yükseliş":
                score += 70
            elif trend_info['medium_term'] == "Nötr":
                score += 35
            else:
                score += 5
                
            # Uzun vadeli trend (50 puan)
            if trend_info['long_term'] == "Yükseliş":
                score += 50
            elif trend_info['long_term'] == "Nötr":
                score += 25
            else:
                score += 5
            
            analysis_details['trend'] = trend_info['direction']
            analysis_details['trend_short'] = trend_info['short_term']
            analysis_details['trend_medium'] = trend_info['medium_term']
            analysis_details['trend_long'] = trend_info['long_term']
        
        # 2. Momentum Analizi (200 puan) - Tüm osilatörler
        if include_momentum:
            max_possible_score += 200
            
            # RSI (50 puan)
            rsi = latest.get('RSI', 50)
            if 30 <= rsi <= 70:
                score += 50
            elif 25 <= rsi <= 75:
                score += 35
            elif 20 <= rsi <= 80:
                score += 20
            else:
                score += 5
            
            # MACD (50 puan)
            macd = latest.get('MACD', 0)
            macd_signal = latest.get('MACD_Signal', 0)
            macd_histogram = latest.get('MACD_Histogram', 0)
            
            if macd > macd_signal and macd > 0 and macd_histogram > 0:
                score += 50
            elif macd > macd_signal and macd_histogram > 0:
                score += 35
            elif macd > macd_signal:
                score += 20
            else:
                score += 5
            
            # Stochastic (30 puan)
            stoch_k = latest.get('Stoch_%K', 50)
            stoch_d = latest.get('Stoch_%D', 50)
            if 20 <= stoch_k <= 80 and stoch_k > stoch_d:
                score += 30
            elif 20 <= stoch_k <= 80:
                score += 20
            else:
                score += 5
            
            # Williams %R (30 puan)
            williams_r = latest.get('Williams_%R', -50)
            if -80 <= williams_r <= -20:
                score += 30
            elif -90 <= williams_r <= -10:
                score += 20
            else:
                score += 5
            
            # CCI (40 puan)
            cci = latest.get('CCI', 0)
            if -100 <= cci <= 100:
                score += 40
            elif -200 <= cci <= 200:
                score += 25
            else:
                score += 5
            
            analysis_details['rsi'] = rsi
            analysis_details['macd_signal'] = "Pozitif" if macd > macd_signal else "Negatif"
            analysis_details['stoch_k'] = stoch_k
            analysis_details['williams_r'] = williams_r
            analysis_details['cci'] = cci
        
        # 3. Hacim Analizi (150 puan) - Gelişmiş hacim analizi
        if include_volume:
            max_possible_score += 150
            
            # Hacim oranı (70 puan)
            volume_sma = latest.get('Volume_SMA20', volume)
            if volume_sma > 0:
                volume_ratio = volume / volume_sma
                if volume_ratio >= 2.0:
                    score += 70
                elif volume_ratio >= 1.5:
                    score += 55
                elif volume_ratio >= 1.2:
                    score += 40
                elif volume_ratio >= 1.0:
                    score += 25
                else:
                    score += 10
            else:
                score += 15
            
            # OBV Trend (40 puan)
            obv = latest.get('OBV', 0)
            obv_sma = latest.get('OBV_SMA', obv)
            if obv > obv_sma:
                score += 40
            else:
                score += 10
            
            # Volume Price Trend (40 puan)
            vpt = latest.get('VPT', 0)
            vpt_prev = df_with_indicators['VPT'].iloc[-5:].mean() if 'VPT' in df_with_indicators.columns else vpt
            if vpt > vpt_prev:
                score += 40
            else:
                score += 10
            
            analysis_details['volume_ratio'] = volume / volume_sma if volume_sma > 0 else 1
            analysis_details['obv_trend'] = "Pozitif" if obv > obv_sma else "Negatif"
        
        # 4. Volatilite ve Risk (100 puan)
        max_possible_score += 100
        
        returns = df_with_indicators['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Volatilite skorlaması
        if volatility <= 10:
            score += 100
        elif volatility <= 15:
            score += 80
        elif volatility <= 20:
            score += 60
        elif volatility <= 25:
            score += 40
        elif volatility <= 30:
            score += 20
        else:
            score += 5
            
        analysis_details['volatility'] = volatility
        
        # 5. Destek/Direnç ve Bollinger Bantları (100 puan)
        max_possible_score += 100
        
        upper_band = latest.get('Upper_Band', current_price * 1.1)
        lower_band = latest.get('Lower_Band', current_price * 0.9)
        middle_band = latest.get('Middle_Band', current_price)
        
        # Bollinger Band pozisyonu (60 puan)
        if lower_band < current_price < middle_band:
            score += 60  # İdeal pozisyon
        elif middle_band < current_price < (middle_band + upper_band) / 2:
            score += 45
        elif current_price > lower_band * 1.02:
            score += 30
        else:
            score += 10
        
        # Fibonacci seviyeleri (40 puan) - Sadece eğer etkinleştirilmişse
        if include_fibonacci:
            try:
                high_52w = df_with_indicators['Close'].tail(252).max()
                low_52w = df_with_indicators['Close'].tail(252).min()
                fib_levels = {
                    '23.6': low_52w + 0.236 * (high_52w - low_52w),
                    '38.2': low_52w + 0.382 * (high_52w - low_52w),
                    '50.0': low_52w + 0.5 * (high_52w - low_52w),
                    '61.8': low_52w + 0.618 * (high_52w - low_52w)
                }
                
                # Fibonacci seviyelerine yakınlık kontrolü
                fib_found = False
                for level_name, level_value in fib_levels.items():
                    if abs(current_price - level_value) / current_price < 0.02:  # %2 tolerans
                        if level_name in ['23.6', '38.2']:  # Destek seviyeleri
                            score += 40
                            analysis_details['fibonacci_level'] = f"Destek: {level_name}%"
                            fib_found = True
                            break
                        elif level_name in ['50.0', '61.8']:  # Önemli seviyeler
                            score += 30
                            analysis_details['fibonacci_level'] = f"Seviye: {level_name}%"
                            fib_found = True
                            break
                
                if not fib_found:
                    score += 10  # Hiçbir seviyeye yakın değil
                    analysis_details['fibonacci_level'] = "Önemli seviyede değil"
                    
            except:
                score += 15  # Fibonacci hesaplanamadıysa orta puan
                analysis_details['fibonacci_level'] = "Hesaplanamadı"
        else:
            score += 20  # Fibonacci analizi kapalıysa orta puan
        
        # 6. Formasyonlar ve Chart Patterns (100 puan) - Sadece eğer etkinleştirilmişse
        if include_patterns:
            max_possible_score += 100
            
            pattern_score = 0
            detected_patterns = []
            
            if chart_patterns:
                for pattern_name, pattern_data in chart_patterns.items():
                    if pattern_data.get('signal') == 'BUY':
                        pattern_score += 30
                        detected_patterns.append(f"{pattern_name} (AL)")
                    elif pattern_data.get('signal') == 'HOLD':
                        pattern_score += 20
                        detected_patterns.append(f"{pattern_name} (TUT)")
                    elif pattern_data.get('signal') == 'SELL':
                        pattern_score += 5
                        detected_patterns.append(f"{pattern_name} (SAT)")
                    else:
                        pattern_score += 15
                        detected_patterns.append(pattern_name)
            
            score += min(pattern_score, 100)  # Maksimum 100 puan
            analysis_details['patterns'] = detected_patterns if detected_patterns else ["Önemli formasyon tespit edilmedi"]
        else:
            # Pattern analizi kapalıysa orta puan ver
            max_possible_score += 100
            score += 40
            analysis_details['patterns'] = ["Formasyon analizi kapalı"]
        
        # 7. Yapay Zeka Analizi (150 puan) - Eğer etkinleştirilmişse veya analiz derinliği gerektiriyorsa
        ai_enabled = include_ai or analysis_depth in ["Orta", "Detaylı"]
        
        if ai_enabled:
            max_possible_score += 150
            
            try:
                # AI analizi yap
                model = initialize_gemini_api()
                ai_analysis_text = ai_stock_analysis(model, symbol, df_with_indicators)
                
                # AI analiz metninden öneri çıkar
                ai_rec = "TUT"  # Varsayılan öneri
                if ai_analysis_text:
                    ai_text_upper = ai_analysis_text.upper()
                    if 'GÜÇLÜ AL' in ai_text_upper or 'STRONG BUY' in ai_text_upper or 'ALIM' in ai_text_upper:
                        ai_rec = "GÜÇLÜ AL"
                        score += 150
                    elif 'AL' in ai_text_upper and 'SAT' not in ai_text_upper:
                        ai_rec = "AL"
                        score += 120
                    elif 'TUT' in ai_text_upper or 'HOLD' in ai_text_upper:
                        ai_rec = "TUT"
                        score += 75
                    elif 'SAT' in ai_text_upper or 'SELL' in ai_text_upper:
                        ai_rec = "SAT"
                        score += 30
                    else:
                        ai_rec = "NÖTR"
                        score += 60
                    
                    analysis_details['ai_recommendation'] = ai_rec
                    analysis_details['ai_confidence'] = 'Orta'
                    analysis_details['ai_summary'] = ai_analysis_text[:150] + '...' if len(ai_analysis_text) > 150 else ai_analysis_text
                else:
                    score += 60  # AI analizi yapılamadıysa orta puan
                    analysis_details['ai_recommendation'] = "Analiz edilemedi"
                    analysis_details['ai_summary'] = "AI analizi başarısız"
                    
            except Exception as e:
                score += 60  # AI hatası durumunda orta puan
                analysis_details['ai_recommendation'] = "Hata"
                analysis_details['ai_summary'] = f"AI hatası: {str(e)[:50]}..."
        else:
            # AI analizi kapalıysa ortalama puan ver
            max_possible_score += 150
            score += 75
            analysis_details['ai_recommendation'] = "AI analizi kapalı"
        
        # Nihai skor hesapla (100 üzerinden normalize et)
        final_score = (score / max_possible_score * 100) if max_possible_score > 0 else 0
        
        # Şirket bilgilerini al
        try:
            company_info = get_company_info(f"{symbol}.IS")
            company_name = company_info.get('shortName', symbol)
        except:
            company_name = symbol
        
        # Risk seviyesi hesapla
        risk_level, _ = calculate_risk_level(volatility, RISK_THRESHOLDS)
        
        # Sonuç oluştur
        result = {
            'symbol': symbol,
            'company_name': company_name,
            'score': round(final_score, 1),
            'current_price': current_price,
            'volume': volume,
            'analysis_details': analysis_details,
            'recommendation': get_recommendation(final_score),
            'risk_level': risk_level,
            'last_update': datetime.now().strftime("%H:%M:%S"),
            'analysis_depth': analysis_depth,
            'max_possible_score': max_possible_score,
            'raw_score': score
        }
        
        return result
        
    except Exception as e:
        print(f"Debug: {symbol} analiz hatası: {str(e)}")
        return None

def get_recommendation(score):
    """Skora göre tavsiye ver"""
    if score >= 80:
        return "GÜÇLÜ AL"
    elif score >= 70:
        return "AL"
    elif score >= 60:
        return "ZAYIF AL"
    elif score >= 40:
        return "TUT"
    elif score >= 30:
        return "ZAYIF SAT"
    else:
        return "SAT"

def get_risk_level(volatility, max_volatility):
    """Risk seviyesini belirle"""
    if volatility <= max_volatility * 0.5:
        return "Düşük"
    elif volatility <= max_volatility * 0.75:
        return "Orta"
    else:
        return "Yüksek"

def display_scan_results():
    """Tarama sonuçlarını göster"""
    
    results = st.session_state.scan_results
    
    if not results:
        st.info("Görüntülenecek sonuç bulunamadı.")
        return
    
    st.subheader(f"📊 Tarama Sonuçları - {len(results)} Hisse Bulundu")
    
    # Özet istatistikler
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_score = np.mean([r['score'] for r in results])
        st.metric("Ortalama Skor", f"{avg_score:.1f}")
    
    with col2:
        strong_buy = len([r for r in results if r['recommendation'] == "GÜÇLÜ AL"])
        st.metric("Güçlü Al", strong_buy)
    
    with col3:
        buy_signals = len([r for r in results if r['recommendation'] in ["GÜÇLÜ AL", "AL"]])
        st.metric("Al Sinyali", buy_signals)
    
    with col4:
        low_risk = len([r for r in results if r['risk_level'] == "Düşük"])
        st.metric("Düşük Risk", low_risk)
    
    # Sonuç tabloları
    tab1, tab2, tab3 = st.tabs(["📈 En İyi Fırsatlar", "📊 Tüm Sonuçlar", "🔍 Detaylı Analiz"])
    
    with tab1:
        st.markdown("### 🏆 En Yüksek Skorlu Hisseler")
        display_top_opportunities(results[:10])
    
    with tab2:
        st.markdown("### 📋 Tüm Tarama Sonuçları")
        display_all_results(results)
    
    with tab3:
        st.markdown("### 🔬 Detaylı Analiz")
        display_detailed_analysis(results)

def display_top_opportunities(results):
    """En iyi fırsatları göster"""
    
    for i, result in enumerate(results, 1):
        symbol = result['symbol']
        score = result['score']
        recommendation = result['recommendation']
        price = result['current_price']
        risk = result['risk_level']
        
        # Renk kodlaması
        rec_color = {
            "GÜÇLÜ AL": "darkgreen",
            "AL": "green", 
            "ZAYIF AL": "lightgreen",
            "TUT": "orange",
            "ZAYIF SAT": "coral",
            "SAT": "red"
        }.get(recommendation, "gray")
        
        risk_color = {
            "Düşük": "green",
            "Orta": "orange", 
            "Yüksek": "red"
        }.get(risk, "gray")
        
        st.markdown(f"""
        <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; margin-bottom: 10px; background-color: #f9f9f9;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4 style="margin: 0; color: #333;">#{i} {symbol} - {result['company_name']}</h4>
                    <p style="margin: 5px 0; color: #666;">Fiyat: {price:.2f} ₺</p>
                </div>
                <div style="text-align: right;">
                    <div style="background-color: {rec_color}; color: white; padding: 5px 10px; border-radius: 5px; margin-bottom: 5px;">
                        <strong>{recommendation}</strong>
                    </div>
                    <div style="font-size: 24px; font-weight: bold; color: #333;">
                        {score:.1f}
                    </div>
                    <div style="color: {risk_color}; font-size: 12px;">
                        Risk: {risk}
                    </div>
                </div>
            </div>
            <div style="margin-top: 10px; font-size: 12px; color: #666;">
                Son güncelleme: {result['last_update']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Favorilere ekle butonu
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button(f"⭐ Favorilere Ekle", key=f"fav_{symbol}_{i}"):
                if symbol not in st.session_state.get('favorite_stocks', []):
                    if 'favorite_stocks' not in st.session_state:
                        st.session_state.favorite_stocks = []
                    st.session_state.favorite_stocks.append(symbol)
                    st.success(f"{symbol} favorilere eklendi!")
                else:
                    st.info(f"{symbol} zaten favorilerde.")
        
        with col2:
            if st.button(f"📊 Detaylı Analiz", key=f"detail_{symbol}_{i}"):
                st.session_state.selected_stock_for_analysis = symbol
                st.info(f"{symbol} için Hisse Analizi sayfasına gidin.")

def display_all_results(results):
    """Tüm sonuçları tablo formatında göster"""
    
    # DataFrame oluştur
    df_data = []
    for result in results:
        details = result['analysis_details']
        df_data.append({
            'Hisse': result['symbol'],
            'Şirket': result['company_name'],
            'Skor': result['score'],
            'Tavsiye': result['recommendation'],
            'Fiyat (₺)': result['current_price'],
            'Risk': result['risk_level'],
            'RSI': details.get('rsi', '-'),
            'MACD': details.get('macd_signal', '-'),
            'Volatilite (%)': f"{details.get('volatility', 0):.1f}",
            'Hacim Oranı': f"{details.get('volume_ratio', 1):.2f}",
            'Trend': details.get('trend', '-')
        })
    
    df = pd.DataFrame(df_data)
    
    # DataFrame'i göster
    st.dataframe(df, use_container_width=True)
    
    # CSV indirme butonu
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 Sonuçları CSV olarak İndir",
        data=csv,
        file_name=f'hisse_tarama_sonuclari_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
        mime='text/csv'
    )

def display_detailed_analysis(results):
    """Detaylı analiz grafiklerini göster"""
    
    if not results:
        return
    
    # Skor dağılımı
    scores = [r['score'] for r in results]
    
    fig_hist = px.histogram(
        x=scores,
        nbins=20,
        title="Skor Dağılımı",
        labels={'x': 'Skor', 'y': 'Hisse Sayısı'}
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Risk vs Skor scatter plot
    risk_scores = []
    risk_labels = []
    colors = []
    
    for result in results:
        risk_scores.append(result['score'])
        risk_labels.append(result['risk_level'])
        colors.append(result['analysis_details'].get('volatility', 0))
    
    fig_scatter = px.scatter(
        x=risk_scores,
        y=risk_labels,
        color=colors,
        title="Risk Seviyesi vs Skor",
        labels={'x': 'Skor', 'y': 'Risk Seviyesi', 'color': 'Volatilite (%)'},
        hover_data=[result['symbol'] for result in results]
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Tavsiye dağılımı
    recommendations = [r['recommendation'] for r in results]
    rec_counts = pd.Series(recommendations).value_counts()
    
    fig_pie = px.pie(
        values=rec_counts.values,
        names=rec_counts.index,
        title="Tavsiye Dağılımı"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

if __name__ == "__main__":
    render_comprehensive_scanner_tab()
