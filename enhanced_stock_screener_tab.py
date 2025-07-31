import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import math

from data.stock_data import get_stock_data, get_company_info, get_stock_news, get_all_bist_stocks, get_stock_list
from analysis.indicators import calculate_indicators
from data.db_utils import save_analysis_result
from data.utils import load_analysis_results

def render_enhanced_stock_screener_tab():
    """
    Gelişmiş hisse tarama sekmesi - Tüm BIST hisselerini analiz eder ve filtreler
    """
    st.header("🔍 Gelişmiş Hisse Tarayıcısı", divider="rainbow")
    st.markdown("""
    Bu araç, Borsa İstanbul'daki tüm hisseleri veya seçili endeksleri analiz eder ve belirlediğiniz kriterlere göre filtreler.
    """)
    
    # Sayfa ayarları
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("⚙️ Tarama Ayarları")
        
        # Tarama kapsamı
        tarama_kapsamı = st.selectbox(
            "Tarama Kapsamı",
            ["BIST 30", "BIST 50", "BIST 100", "Tüm BIST Hisseleri", "Özel Liste"],
            index=2
        )
        
        # Özel hisse listesi
        if tarama_kapsamı == "Özel Liste":
            ozel_hisseler = st.text_area(
                "Hisse Kodları (Her satıra bir kod)",
                placeholder="THYAO\nASELS\nGARAN\nEREGL"
            )
        
        # Analiz periyodu
        analiz_periyodu = st.selectbox(
            "Analiz Periyodu",
            ["1 Hafta", "2 Hafta", "1 Ay", "3 Ay", "6 Ay", "1 Yıl"],
            index=3
        )
        
        # Teknik göstergeler
        st.subheader("📊 Teknik Göstergeler")
        rsi_aktif = st.checkbox("RSI Analizi", value=True)
        macd_aktif = st.checkbox("MACD Analizi", value=True)
        bollinger_aktif = st.checkbox("Bollinger Bantları", value=True)
        ma_aktif = st.checkbox("Hareketli Ortalamalar", value=True)
        volume_aktif = st.checkbox("Hacim Analizi", value=True)
        
        # Filtreleme kriterleri
        st.subheader("🎯 Filtreleme Kriterleri")
        
        # Fiyat aralığı
        price_filter = st.checkbox("Fiyat Aralığı")
        if price_filter:
            price_min, price_max = st.slider("Fiyat Aralığı (TL)", 0.1, 1000.0, (1.0, 500.0))
        
        # Günlük değişim
        change_filter = st.checkbox("Günlük Değişim %")
        if change_filter:
            change_min, change_max = st.slider("Değişim Aralığı (%)", -20.0, 20.0, (-5.0, 10.0))
        
        # Hacim filtresi
        volume_filter = st.checkbox("Minimum Hacim")
        if volume_filter:
            min_volume = st.number_input("Minimum Günlük Hacim", value=100000, step=50000)
        
        # RSI filtresi
        rsi_filter = st.checkbox("RSI Filtresi")
        if rsi_filter:
            rsi_min, rsi_max = st.slider("RSI Aralığı", 0, 100, (30, 70))
        
        # Sinyal filtresi
        signal_filter = st.checkbox("Teknik Sinyal Filtresi")
        if signal_filter:
            min_signals = st.slider("Minimum Sinyal Sayısı", 0, 10, 2)
        
        # Tarama butonu
        start_scanning = st.button("🚀 Taramayı Başlat", type="primary")
    
    with col1:
        if start_scanning:
            # Hisse listesini hazırla
            if tarama_kapsamı == "Özel Liste":
                if ozel_hisseler:
                    symbols = [s.strip().upper() for s in ozel_hisseler.split('\n') if s.strip()]
                else:
                    st.error("Özel hisse listesi boş!")
                    return
            elif tarama_kapsamı == "Tüm BIST Hisseleri":
                symbols = get_all_bist_stocks()
                st.info(f"Toplam {len(symbols)} hisse taranacak...")
            else:
                stock_list = get_stock_list(tarama_kapsamı)
                symbols = [s.replace('.IS', '') for s in stock_list]
            
            # Periyodu çevir
            period_map = {
                "1 Hafta": "1w", "2 Hafta": "2w", "1 Ay": "1mo",
                "3 Ay": "3mo", "6 Ay": "6mo", "1 Yıl": "1y"
            }
            period = period_map[analiz_periyodu]
            
            # Tarama işlemini başlat
            results = perform_stock_screening(
                symbols, period, {
                    'rsi': rsi_aktif, 'macd': macd_aktif, 'bollinger': bollinger_aktif,
                    'ma': ma_aktif, 'volume': volume_aktif
                }, {
                    'price_filter': price_filter,
                    'price_range': (price_min, price_max) if price_filter else None,
                    'change_filter': change_filter,
                    'change_range': (change_min, change_max) if change_filter else None,
                    'volume_filter': volume_filter,
                    'min_volume': min_volume if volume_filter else None,
                    'rsi_filter': rsi_filter,
                    'rsi_range': (rsi_min, rsi_max) if rsi_filter else None,
                    'signal_filter': signal_filter,
                    'min_signals': min_signals if signal_filter else None
                }
            )
            
            # Sonuçları göster
            display_screening_results(results)
        else:
            st.info("Tarama yapmak için sol taraftaki ayarları yapılandırın ve 'Taramayı Başlat' butonuna tıklayın.")
            
            # Örnek gösterge açıklamaları
            st.subheader("📖 Teknik Gösterge Açıklamaları")
            
            with st.expander("RSI (Relative Strength Index)"):
                st.write("""
                **RSI (Göreli Güç Endeksi):**
                - 0-100 arasında değişir
                - 30'un altı: Aşırı satım (alış fırsatı)
                - 70'in üstü: Aşırı alım (satış fırsatı)
                - 50: Nötr bölge
                """)
            
            with st.expander("MACD (Moving Average Convergence Divergence)"):
                st.write("""
                **MACD:**
                - Trend değişimlerini yakalar
                - MACD çizgisi sinyal çizgisini yukarı keserse: Alış sinyali
                - MACD çizgisi sinyal çizgisini aşağı keserse: Satış sinyali
                - Histogram: Momentum gücünü gösterir
                """)
            
            with st.expander("Bollinger Bantları"):
                st.write("""
                **Bollinger Bantları:**
                - Üst Bant: Dirença yakın
                - Alt Bant: Desteke yakın
                - Fiyat alt bandı aşağıdan yukarı keserse: Alış sinyali
                - Bantlar daraldığında: Büyük hareket beklentisi
                """)

def perform_stock_screening(symbols, period, indicators, filters):
    """
    Hisse tarama işlemini gerçekleştirir
    """
    results = []
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    total_symbols = len(symbols)
    processed = 0
    
    for i, symbol in enumerate(symbols):
        progress_text.text(f"Analiz ediliyor: {symbol} ({i+1}/{total_symbols})")
        progress_bar.progress((i + 1) / total_symbols)
        
        try:
            # Hisse verisini al
            df = get_stock_data(symbol, period)
            
            if len(df) < 20:  # Minimum veri kontrolü
                continue
                
            # Şirket bilgilerini al
            company_info = get_company_info(symbol)
            company_name = company_info.get('name', symbol)
            
            # Teknik göstergeleri hesapla
            df_with_indicators = calculate_indicators(df)
            
            # Son değerleri al
            latest = df_with_indicators.iloc[-1]
            prev = df_with_indicators.iloc[-2] if len(df_with_indicators) > 1 else latest
            
            # Temel bilgileri hazırla
            current_price = latest['Close']
            daily_change = ((latest['Close'] - prev['Close']) / prev['Close'] * 100) if 'Close' in prev else 0
            volume = latest['Volume'] if 'Volume' in latest else 0
            
            # Filtreleri uygula
            if not apply_filters(latest, prev, current_price, daily_change, volume, filters):
                continue
            
            # Sinyalleri analiz et
            signals = analyze_signals(latest, prev, indicators)
            signal_count = len(signals)
            
            # Sinyal filtresini kontrol et
            if filters.get('signal_filter') and signal_count < filters.get('min_signals', 0):
                continue
            
            # Sonucu kaydet
            result = {
                'Symbol': symbol,
                'Name': company_name,
                'Price': round(current_price, 2),
                'Change_%': round(daily_change, 2),
                'Volume': int(volume),
                'Signal_Count': signal_count,
                'Signals': ', '.join(signals),
                'RSI': round(latest.get('RSI', 0), 2) if 'RSI' in latest else None,
                'MACD': round(latest.get('MACD', 0), 4) if 'MACD' in latest else None,
                'Score': calculate_score(latest, signals)
            }
            
            results.append(result)
            processed += 1
            
        except Exception as e:
            st.warning(f"{symbol} analiz edilirken hata: {str(e)}")
            continue
    
    progress_text.text(f"Tarama tamamlandı! {processed} hisse analiz edildi.")
    time.sleep(1)
    progress_bar.empty()
    progress_text.empty()
    
    return results

def apply_filters(latest, prev, current_price, daily_change, volume, filters):
    """
    Filtreleme kriterlerini uygular
    """
    # Fiyat filtresi
    if filters.get('price_filter') and filters.get('price_range'):
        min_price, max_price = filters['price_range']
        if not (min_price <= current_price <= max_price):
            return False
    
    # Değişim filtresi
    if filters.get('change_filter') and filters.get('change_range'):
        min_change, max_change = filters['change_range']
        if not (min_change <= daily_change <= max_change):
            return False
    
    # Hacim filtresi
    if filters.get('volume_filter') and filters.get('min_volume'):
        if volume < filters['min_volume']:
            return False
    
    # RSI filtresi
    if filters.get('rsi_filter') and filters.get('rsi_range') and 'RSI' in latest:
        min_rsi, max_rsi = filters['rsi_range']
        if not (min_rsi <= latest['RSI'] <= max_rsi):
            return False
    
    return True

def analyze_signals(latest, prev, indicators):
    """
    Teknik sinyalleri analiz eder
    """
    signals = []
    
    # RSI Sinyalleri
    if indicators.get('rsi') and 'RSI' in latest and 'RSI' in prev:
        try:
            if prev['RSI'] < 30 and latest['RSI'] >= 30:
                signals.append("RSI Aşırı Satım Çıkış")
            elif latest['RSI'] < 35 and latest['RSI'] > 25:
                signals.append("RSI Aşırı Satım Yakın")
            elif prev['RSI'] > 70 and latest['RSI'] <= 70:
                signals.append("RSI Aşırı Alım Çıkış")
        except:
            pass
    
    # MACD Sinyalleri
    if indicators.get('macd') and all(col in latest for col in ['MACD', 'MACD_Signal']):
        try:
            if prev['MACD'] < prev['MACD_Signal'] and latest['MACD'] > latest['MACD_Signal']:
                signals.append("MACD Alış Sinyali")
            elif latest['MACD'] > 0 and latest['MACD'] > latest['MACD_Signal']:
                signals.append("MACD Pozitif")
        except:
            pass
    
    # Bollinger Sinyalleri
    if indicators.get('bollinger') and all(col in latest for col in ['Lower_Band', 'Middle_Band', 'Upper_Band']):
        try:
            if prev['Close'] < prev['Lower_Band'] and latest['Close'] > latest['Lower_Band']:
                signals.append("Bollinger Alt Bant Kırılım")
            elif latest['Close'] > latest['Middle_Band']:
                signals.append("Bollinger Orta Üstü")
        except:
            pass
    
    # Hareketli Ortalama Sinyalleri
    if indicators.get('ma'):
        try:
            if all(col in latest for col in ['SMA5', 'SMA20']):
                if prev['SMA5'] < prev['SMA20'] and latest['SMA5'] > latest['SMA20']:
                    signals.append("Golden Cross (5/20)")
                elif latest['Close'] > latest['SMA5'] > latest['SMA20']:
                    signals.append("MA Yükseliş Trendi")
            
            if all(col in latest for col in ['EMA5', 'EMA20']):
                if prev['EMA5'] < prev['EMA20'] and latest['EMA5'] > latest['EMA20']:
                    signals.append("EMA Golden Cross")
        except:
            pass
    
    # Hacim Sinyalleri
    if indicators.get('volume'):
        try:
            # Son 5 günün ortalama hacmi
            if 'Volume' in latest and 'Volume' in prev:
                if latest['Volume'] > prev['Volume'] * 1.5:
                    signals.append("Yüksek Hacim")
        except:
            pass
    
    return signals

def calculate_score(latest, signals):
    """
    Hisse için toplam skor hesaplar
    """
    score = len(signals) * 10  # Her sinyal 10 puan
    
    # RSI bonus
    if 'RSI' in latest:
        if 25 <= latest['RSI'] <= 35:  # Aşırı satım yakın
            score += 15
        elif 35 <= latest['RSI'] <= 45:  # Alış bölgesi
            score += 10
        elif 55 <= latest['RSI'] <= 65:  # Güçlü trend
            score += 5
    
    # MACD bonus
    if 'MACD' in latest and 'MACD_Signal' in latest:
        if latest['MACD'] > latest['MACD_Signal'] and latest['MACD'] > 0:
            score += 10
    
    return min(score, 100)  # Max 100 puan

def display_screening_results(results):
    """
    Tarama sonuçlarını görüntüler
    """
    if not results:
        st.warning("Filtreleme kriterlerinize uygun hisse bulunamadı.")
        return
    
    st.success(f"🎯 Toplam {len(results)} hisse filtreleme kriterlerinizi karşılıyor!")
    
    # Sonuçları DataFrame'e çevir
    df_results = pd.DataFrame(results)
    
    # Sıralama seçenekleri
    col1, col2, col3 = st.columns(3)
    with col1:
        sort_by = st.selectbox("Sırala:", ["Score", "Signal_Count", "Change_%", "Price", "Volume"])
    with col2:
        sort_order = st.selectbox("Düzen:", ["Azalan", "Artan"])
    with col3:
        show_count = st.selectbox("Göster:", [20, 50, 100, len(results)], index=1)
    
    # Sıralama uygula
    ascending = (sort_order == "Artan")
    df_sorted = df_results.sort_values(sort_by, ascending=ascending).head(show_count)
    
    # Sonuç tablosunu göster
    st.subheader(f"📊 En İyi {len(df_sorted)} Hisse")
    
    # Renklendirilmiş tablo
    def color_change(val):
        color = 'red' if val < 0 else 'green'
        return f'color: {color}'
    
    def color_score(val):
        if val >= 70:
            return 'background-color: #d4edda'  # Yeşil
        elif val >= 40:
            return 'background-color: #fff3cd'  # Sarı
        else:
            return 'background-color: #f8d7da'  # Kırmızı
    
    styled_df = df_sorted.style.applymap(color_change, subset=['Change_%']) \
                               .applymap(color_score, subset=['Score']) \
                               .format({
                                   'Price': '{:.2f} ₺',
                                   'Change_%': '{:.2f}%',
                                   'Volume': '{:,}',
                                   'RSI': '{:.1f}',
                                   'MACD': '{:.4f}'
                               })
    
    st.dataframe(styled_df, use_container_width=True, height=600)
    
    # Detay görünümü için hisse seçimi
    st.subheader("🔍 Detaylı Analiz")
    selected_symbol = st.selectbox("Hisse Seç:", df_sorted['Symbol'].tolist())
    
    if selected_symbol:
        show_detailed_analysis(selected_symbol, results)
    
    # Özet istatistikler
    st.subheader("📈 Özet İstatistikler")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_change = df_sorted['Change_%'].mean()
        st.metric("Ortalama Değişim", f"{avg_change:.2f}%")
    
    with col2:
        avg_score = df_sorted['Score'].mean()
        st.metric("Ortalama Skor", f"{avg_score:.1f}")
    
    with col3:
        high_signal_count = len(df_sorted[df_sorted['Signal_Count'] >= 3])
        st.metric("3+ Sinyal", f"{high_signal_count} hisse")
    
    with col4:
        positive_change = len(df_sorted[df_sorted['Change_%'] > 0])
        st.metric("Pozitif Getiri", f"{positive_change} hisse")

def show_detailed_analysis(symbol, results):
    """
    Seçilen hisse için detaylı analiz gösterir
    """
    # Seçilen hissenin verilerini bul
    stock_data = next((r for r in results if r['Symbol'] == symbol), None)
    
    if not stock_data:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"📊 {symbol} - {stock_data['Name']}")
        
        # Temel bilgiler
        st.metric("Güncel Fiyat", f"{stock_data['Price']} ₺", f"{stock_data['Change_%']:.2f}%")
        st.metric("Toplam Skor", stock_data['Score'])
        st.metric("Sinyal Sayısı", stock_data['Signal_Count'])
        
        # Sinyaller
        if stock_data['Signals']:
            st.write("**Teknik Sinyaller:**")
            for signal in stock_data['Signals'].split(', '):
                st.write(f"• {signal}")
    
    with col2:
        # Grafik gösterimi
        try:
            df = get_stock_data(symbol, "3mo")
            
            if len(df) > 0:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='lines',
                    name='Fiyat',
                    line=dict(color='blue', width=2)
                ))
                
                fig.update_layout(
                    title=f"{symbol} - Son 3 Ay",
                    xaxis_title="Tarih",
                    yaxis_title="Fiyat (₺)",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
        except:
            st.write("Grafik gösterilemiyor.") 