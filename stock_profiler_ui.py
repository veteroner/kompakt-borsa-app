import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os

# Ana proje dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.stock_profiler import StockProfiler

def render_stock_profiler_tab():
    """Hisse Profil Analizi sekmesi"""
    
    st.title("🎯 Hisse Profil Analizi")
    st.markdown("Her hisse senedi için özelleştirilmiş analiz profilleri oluşturun ve yönetin")
    
    # Tabs oluştur
    profile_tab, analysis_tab, signals_tab = st.tabs([
        "📊 Profil Oluştur", 
        "🔍 Profil Analizi", 
        "📡 Kişisel Sinyaller"
    ])
    
    # Profiler instance'ı oluştur
    profiler = StockProfiler()
    
    with profile_tab:
        render_profile_creation(profiler)
    
    with analysis_tab:
        render_profile_analysis(profiler)
    
    with signals_tab:
        render_personalized_signals(profiler)

def render_profile_creation(profiler):
    """Profil oluşturma sekmesi"""
    
    st.header("📊 Yeni Profil Oluştur")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.text_input(
            "Hisse Sembolü",
            value="THYAO",
            help="Örnek: THYAO, ASELS, BIST"
        ).upper()
        
        period = st.selectbox(
            "Analiz Periyodu",
            options=["1y", "2y", "3y", "5y"],
            index=1,
            help="Ne kadar geçmiş veri kullanılacak"
        )
    
    with col2:
        threshold = st.slider(
            "Önemli Hareket Eşiği (%)",
            min_value=1,
            max_value=10,
            value=5,
            help="Yüksek/düşük hareket olarak kabul edilecek minimum değişim"
        )
        
        auto_update = st.checkbox(
            "Otomatik Güncelleme",
            value=False,
            help="Profili düzenli olarak güncelle"
        )
    
    if st.button("🚀 Profil Oluştur", type="primary"):
        if symbol:
            with st.spinner(f"{symbol} için profil oluşturuluyor..."):
                try:
                    profile = profiler.create_stock_profile(symbol, period)
                    
                    if profile:
                        st.success(f"✅ {symbol} profili başarıyla oluşturuldu!")
                        
                        # Özet bilgileri göster
                        display_profile_summary(profile)
                        
                    else:
                        st.error(f"❌ {symbol} için profil oluşturulamadı. Hisse sembolünü kontrol edin.")
                
                except Exception as e:
                    st.error(f"❌ Hata: {str(e)}")
        else:
            st.warning("⚠️ Lütfen hisse sembolü girin")

def display_profile_summary(profile):
    """Profil özetini görüntüle"""
    
    st.subheader(f"📈 {profile['symbol']} Profil Özeti")
    
    # Ana metrikler
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Analiz Edilen Gün",
            profile['total_days']
        )
    
    with col2:
        st.metric(
            "Test Edilen Gösterge",
            profile['analysis_summary']['total_indicators']
        )
    
    with col3:
        st.metric(
            "Ortalama Doğruluk",
            f"{profile['analysis_summary']['average_accuracy']:.1%}"
        )
    
    with col4:
        st.metric(
            "En İyi Doğruluk",
            f"{profile['analysis_summary']['best_accuracy']:.1%}"
        )
    
    # En iyi göstergeler
    st.subheader("🏆 En Etkili Göstergeler")
    
    top_indicators = list(profile['indicator_rankings'].keys())[:10]
    indicator_data = []
    
    for indicator in top_indicators:
        info = profile['indicator_rankings'][indicator]
        indicator_data.append({
            'Gösterge': indicator,
            'Skor': info['score'],
            'Doğruluk': info['accuracy'],
            'Hassasiyet': info['precision'],
            'Geri Çağırma': info['recall']
        })
    
    df_indicators = pd.DataFrame(indicator_data)
    st.dataframe(df_indicators, use_container_width=True)
    
    # Görselleştirme
    fig = px.bar(
        df_indicators.head(5),
        x='Gösterge',
        y='Skor',
        title="En İyi 5 Gösterge Performansı",
        color='Doğruluk',
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig, use_container_width=True)

def render_profile_analysis(profiler):
    """Profil analizi sekmesi"""
    
    st.header("🔍 Mevcut Profilleri İncele")
    
    # Mevcut profilleri listele
    profiles = list_available_profiles(profiler)
    
    if not profiles:
        st.info("📂 Henüz oluşturulmuş profil yok. Önce 'Profil Oluştur' sekmesinden profil oluşturun.")
        return
    
    selected_symbol = st.selectbox(
        "Analiz Edilecek Hisse",
        options=profiles,
        help="İncelemek istediğiniz hisse senedini seçin"
    )
    
    if selected_symbol:
        profile = profiler.load_profile(selected_symbol)
        
        if profile:
            display_detailed_profile_analysis(profile)
        else:
            st.error("Profil yüklenemedi")

def display_detailed_profile_analysis(profile):
    """Detaylı profil analizi görüntüle"""
    
    symbol = profile['symbol']
    st.subheader(f"📊 {symbol} Detaylı Analiz")
    
    # Ticaret karakteristikleri
    st.markdown("### 💼 Ticaret Karakteristikleri")
    
    trading_chars = profile['trading_characteristics']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Günlük Ortalama Getiri",
            f"{trading_chars['average_daily_return']:.2%}",
            delta=f"Vol: {trading_chars['volatility']:.2%}"
        )
    
    with col2:
        st.metric(
            "Maksimum Düşüş",
            f"{trading_chars['max_drawdown']:.2%}"
        )
    
    with col3:
        st.metric(
            "Trend Gücü",
            f"{trading_chars['trend_strength']:.3f}"
        )
    
    with col4:
        st.metric(
            "Kırılma Sıklığı",
            f"{trading_chars['breakout_frequency']:.2%}"
        )
    
    # Hacim profili
    st.markdown("### 📊 Hacim Profili")
    
    volume_profile = profile['volume_profile']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Ortalama Hacim",
            f"{volume_profile['average_volume']:,.0f}"
        )
        st.metric(
            "Hacim-Fiyat Korelasyonu",
            f"{volume_profile['volume_price_correlation']:.3f}"
        )
    
    with col2:
        st.metric(
            "Hacim Volatilitesi",
            f"{volume_profile['volume_volatility']:.2%}"
        )
        st.metric(
            "Yüksek Hacim Performansı",
            f"{volume_profile['high_volume_performance']:.2%}"
        )
    
    # Başarı oranları
    st.markdown("### 🎯 Başarı Oranları")
    
    success_rates = profile['success_rates']
    
    success_df = pd.DataFrame([
        {
            'Periyot': '1 Gün',
            'Pozitif Getiri': f"{success_rates.get('positive_return_1d', 0):.1%}",
            'Önemli Kazanç (>%3)': f"{success_rates.get('significant_gain_1d', 0):.1%}",
            'Önemli Kayıp (<%5)': f"{success_rates.get('significant_loss_1d', 0):.1%}"
        },
        {
            'Periyot': '3 Gün',
            'Pozitif Getiri': f"{success_rates.get('positive_return_3d', 0):.1%}",
            'Önemli Kazanç (>%3)': f"{success_rates.get('significant_gain_3d', 0):.1%}",
            'Önemli Kayıp (<%5)': f"{success_rates.get('significant_loss_3d', 0):.1%}"
        },
        {
            'Periyot': '5 Gün',
            'Pozitif Getiri': f"{success_rates.get('positive_return_5d', 0):.1%}",
            'Önemli Kazanç (>%3)': f"{success_rates.get('significant_gain_5d', 0):.1%}",
            'Önemli Kayıp (<%5)': f"{success_rates.get('significant_loss_5d', 0):.1%}"
        }
    ])
    
    st.dataframe(success_df, use_container_width=True)

def render_personalized_signals(profiler):
    """Kişiselleştirilmiş sinyaller sekmesi"""
    
    st.header("📡 Kişiselleştirilmiş Sinyaller")
    
    # Mevcut profilleri listele
    profiles = list_available_profiles(profiler)
    
    if not profiles:
        st.info("📂 Henüz oluşturulmuş profil yok. Önce profil oluşturun.")
        return
    
    selected_symbol = st.selectbox(
        "Sinyal Analizi İçin Hisse",
        options=profiles,
        help="Kişiselleştirilmiş sinyal alacağınız hisse senedini seçin"
    )
    
    if selected_symbol:
        # Güncel veriyi çek
        try:
            ticker = yf.Ticker(f"{selected_symbol}.IS")
            current_data = ticker.history(period="3mo")  # 3 aylık veri
            
            if len(current_data) > 0:
                # Kişiselleştirilmiş sinyalleri al
                signals = profiler.get_personalized_signals(selected_symbol, current_data)
                
                if signals:
                    display_personalized_signals(signals, current_data)
                else:
                    st.error("Sinyaller oluşturulamadı")
            else:
                st.error("Güncel veri alınamadı")
                
        except Exception as e:
            st.error(f"Veri çekme hatası: {str(e)}")

def display_personalized_signals(signals, current_data):
    """Kişiselleştirilmiş sinyalleri görüntüle"""
    
    symbol = signals['symbol']
    
    # Ana sinyal özeti
    st.subheader(f"🎯 {symbol} - Kişisel Sinyal Analizi")
    
    # Genel değerlendirme
    col1, col2, col3 = st.columns(3)
    
    recommendation = signals['recommendation']
    overall_score = signals['overall_score']
    confidence = signals['confidence']
    
    with col1:
        color = "green" if recommendation == "BUY" else "red" if recommendation == "SELL" else "orange"
        st.markdown(f"## :{color}[{recommendation}]")
        st.markdown(f"**Öneri:** {recommendation}")
    
    with col2:
        st.metric(
            "Sinyal Gücü",
            f"{overall_score:.3f}",
            delta=f"Güven: %{confidence:.0f}"
        )
    
    with col3:
        latest_price = current_data['Close'].iloc[-1]
        st.metric(
            "Güncel Fiyat",
            f"{latest_price:.2f} TL"
        )
    
    # Detay sinyaller
    st.subheader("📊 Gösterge Bazında Sinyaller")
    
    signal_data = []
    for signal in signals['signals']:
        signal_emoji = "🟢" if signal['signal'] == 'buy' else "🔴" if signal['signal'] == 'sell' else "🟡"
        
        signal_data.append({
            'Durum': signal_emoji,
            'Gösterge': signal['indicator'],
            'Değer': f"{signal['value']:.3f}",
            'Sinyal': signal['signal'].upper(),
            'Güç': f"{signal['strength']:.3f}",
            'Doğruluk': f"{signal['accuracy']:.1%}",
            'Alım Eşiği': f"{signal['buy_threshold']:.3f}" if signal['buy_threshold'] else "N/A",
            'Satım Eşiği': f"{signal['sell_threshold']:.3f}" if signal['sell_threshold'] else "N/A"
        })
    
    df_signals = pd.DataFrame(signal_data)
    st.dataframe(df_signals, use_container_width=True)
    
    # Sinyal trendini göster
    create_signal_chart(current_data, signals)

def create_signal_chart(data, signals):
    """Sinyal grafiği oluştur"""
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxis=True,
        vertical_spacing=0.1,
        subplot_titles=('Fiyat Hareketi', 'Sinyal Gücü'),
        row_heights=[0.7, 0.3]
    )
    
    # Fiyat grafiği
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Fiyat"
        ),
        row=1, col=1
    )
    
    # Sinyal göstergesi
    signal_strengths = [s['strength'] for s in signals['signals']]
    signal_dates = [signals['date']] * len(signal_strengths)
    
    fig.add_trace(
        go.Scatter(
            x=signal_dates,
            y=signal_strengths,
            mode='markers',
            marker=dict(
                size=15,
                color=signal_strengths,
                colorscale='RdYlGn',
                showscale=True
            ),
            name="Sinyal Gücü"
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f"{signals['symbol']} - Kişiselleştirilmiş Sinyal Analizi",
        xaxis_title="Tarih",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

def list_available_profiles(profiler):
    """Mevcut profilleri listele"""
    try:
        if os.path.exists(profiler.profile_dir):
            files = os.listdir(profiler.profile_dir)
            profiles = [f.replace('_profile.json', '') for f in files if f.endswith('_profile.json')]
            return sorted(profiles)
        return []
    except:
        return []

if __name__ == "__main__":
    render_stock_profiler_tab() 