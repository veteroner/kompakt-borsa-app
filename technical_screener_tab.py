import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from data.stock_data import get_stock_data, get_company_info, get_stock_news, get_all_bist_stocks
from analysis.indicators import calculate_indicators
from data.db_utils import save_analysis_result
from data.utils import load_analysis_results

def render_technical_screener_tab():
    """
    Teknik analiz tarama sekmesini oluşturur ve yükselme potansiyeli olan hisseleri listeler
    """
    st.header("Teknik Gösterge Tarayıcı")
    
    # Layout
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Hisse listesi - Borsa İstanbul'daki tüm hisseler veya belirli endekslerdekiler
        hisse_listesi = st.multiselect(
            "Taramaya Dahil Edilecek Hisseler",
            ["BIST 30", "BIST 50", "BIST 100", "Tüm Hisseler"],
            default=["BIST 100"]
        )
        
        # Özel hisse ekle
        ozel_hisseler = st.text_input("Özel Hisse Senedi Kodları (virgülle ayırın, örn: THYAO,GARAN,ASELS)")
        
    with col2:
        # Gösterge seçimleri
        gostergeler = st.multiselect(
            "Kullanılacak Göstergeler",
            ["RSI", "MACD", "Bollinger", "SMA", "EMA", "Stokastik", "ADX", "Hacim Analizi", "Formasyonlar"],
            default=["RSI", "MACD", "Bollinger"]
        )
    
    with col3:
        # Tarama periyodu
        periyot = st.selectbox(
            "Veri Periyodu",
            ["1 Hafta", "2 Hafta", "1 Ay", "3 Ay", "6 Ay", "1 Yıl"],
            index=3  # Default olarak 3 ay
        )
        
        # Tarama butonu
        tara_btn = st.button("Hisseleri Tara", key="teknik_tara_button")
    
    # Bilgi notu
    st.info("Bu araç, seçilen teknik göstergelere göre yükseliş potansiyeli taşıyan hisseleri tarar. " 
           "Sonuçlar sadece teknik analize dayanır ve yatırım tavsiyesi değildir.")
    
    # Tarama işlemi
    if tara_btn:
        with st.spinner("Teknik analiz taraması yapılıyor. Bu işlem biraz zaman alabilir..."):
            # Periyodu gün sayısına çevir
            periyot_gun = {
                "1 Hafta": "1w", 
                "2 Hafta": "2w", 
                "1 Ay": "1mo", 
                "3 Ay": "3mo",
                "6 Ay": "6mo",
                "1 Yıl": "1y"
            }.get(periyot, "3mo")
            
            # Hisse listesini oluştur
            semboller = []
            
            # Özel hisseler
            if ozel_hisseler:
                ozel_liste = [hisse.strip().upper() for hisse in ozel_hisseler.split(",")]
                semboller.extend(ozel_liste)
            
            # Hazır listeler (burada mock veriler kullanıyoruz, gerçek uygulama için veri kaynağı eklenmelidir)
            if "BIST 30" in hisse_listesi:
                # Mock BIST-30 listemiz (gerçek listede 30 hisse olmalı)
                bist30 = ["THYAO", "ASELS", "GARAN", "EREGL", "AKBNK", "KCHOL", "TUPRS", "SISE", "YKBNK", "SAHOL"]
                semboller.extend([h for h in bist30 if h not in semboller])
            
            if "BIST 50" in hisse_listesi:
                # Mock BIST-50 listemiz 
                bist50 = ["THYAO", "ASELS", "GARAN", "EREGL", "AKBNK", "KCHOL", "TUPRS", "SISE", "YKBNK", "SAHOL", 
                         "PGSUS", "TAVHL", "TOASO", "VESTL", "BIMAS", "FROTO", "ARCLK", "PETKM", "TCELL", "EKGYO"]
                semboller.extend([h for h in bist50 if h not in semboller])
            
            if "BIST 100" in hisse_listesi:
                # Mock BIST-100 (tam liste için veri kaynağı eklenmelidir)
                bist100 = ["THYAO", "ASELS", "GARAN", "EREGL", "AKBNK", "KCHOL", "TUPRS", "SISE", "YKBNK", "SAHOL", 
                          "PGSUS", "TAVHL", "TOASO", "VESTL", "BIMAS", "FROTO", "ARCLK", "PETKM", "TCELL", "EKGYO",
                          "SOKM", "TTKOM", "DOHOL", "KRDMD", "KOZAL", "ULKER", "MGROS", "HEKTS", "ALARK", "TTRAK"]
                semboller.extend([h for h in bist100 if h not in semboller])
            
            if "Tüm Hisseler" in hisse_listesi:
                # BIST'teki tüm hisseleri al
                tum_hisseler = get_all_bist_stocks()
                st.info(f"Toplam {len(tum_hisseler)} hisse taranıyor.")
                semboller = list(set(semboller + tum_hisseler))  # Önceki hisseler de dahil, tekrarları kaldır
            
            # Hisse sayısı kontrolü
            if len(semboller) == 0:
                st.error("Taranacak hisse bulunamadı. Lütfen hisse listesi veya özel hisseler ekleyin.")
                return
            
            # Progress bar
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            # Sonuçları tutacak liste
            results = []
            
            # Her hisse için teknik analiz yap
            for i, sembol in enumerate(semboller):
                progress_text.text(f"{sembol} analiz ediliyor... ({i+1}/{len(semboller)})")
                progress_bar.progress((i) / len(semboller))
                
                try:
                    # Hisse verisini al
                    df = get_stock_data(sembol, periyot_gun)
                    
                    if len(df) < 20:  # Minimum veri gereksinimi
                        continue
                    
                    # Teknik göstergeleri hesapla
                    df_with_indicators = calculate_indicators(df)
                    
                    # Son değerleri al
                    latest = df_with_indicators.iloc[-1]
                    prev = df_with_indicators.iloc[-2]
                    
                    # Sinyalleri kontrol et
                    sinyaller = []
                    sinyal_sayisi = 0
                    
                    # RSI Kontrol (Aşırı satım bölgesinden çıkış sinyali)
                    if "RSI" in gostergeler:
                        try:
                            if 'RSI' in latest and 'RSI' in prev:
                                if prev['RSI'] < 30 and latest['RSI'] >= 30:
                                    sinyaller.append("RSI aşırı satım bölgesinden çıkış")
                                    sinyal_sayisi += 1
                                elif 35 > latest['RSI'] > 30:
                                    sinyaller.append("RSI aşırı satım bölgesine yakın")
                                    sinyal_sayisi += 0.5
                        except:
                            pass
                    
                    # MACD Kontrol (MACD Sinyal çizgisini yukarı yönde kesiyor)
                    if "MACD" in gostergeler:
                        try:
                            if all(col in latest for col in ['MACD', 'MACD_Signal']) and all(col in prev for col in ['MACD', 'MACD_Signal']):
                                if prev['MACD'] < prev['MACD_Signal'] and latest['MACD'] > latest['MACD_Signal']:
                                    sinyaller.append("MACD sinyal çizgisini yukarı kesti (Alış)")
                                    sinyal_sayisi += 1
                                elif latest['MACD'] > 0 and latest['MACD'] > latest['MACD_Signal'] and latest['MACD'] > prev['MACD']:
                                    sinyaller.append("MACD pozitif ve yükseliyor")
                                    sinyal_sayisi += 0.5
                        except:
                            pass
                    
                    # Bollinger Bant Kontrolü (Fiyat alt bandı aşağıdan yukarı kesti)
                    if "Bollinger" in gostergeler:
                        try:
                            if all(col in latest for col in ['Lower_Band', 'Middle_Band', 'Upper_Band']):
                                if prev['Close'] < prev['Lower_Band'] and latest['Close'] > latest['Lower_Band']:
                                    sinyaller.append("Fiyat Bollinger alt bandını yukarı kesti")
                                    sinyal_sayisi += 1
                                elif latest['Close'] < latest['Middle_Band'] and latest['Close'] > latest['Lower_Band']:
                                    band_width = (latest['Upper_Band'] - latest['Lower_Band']) / latest['Middle_Band']
                                    if band_width < 0.1:  # Bantlar sıkışıyorsa
                                        sinyaller.append("Bollinger bantları sıkışıyor - olası kırılma")
                                        sinyal_sayisi += 0.5
                        except:
                            pass
                    
                    # SMA Kontrol (Kısa dönem SMA uzun dönem SMA'yı yukarı kesiyor - Golden Cross)
                    if "SMA" in gostergeler:
                        try:
                            if all(col in latest for col in ['SMA5', 'SMA20']):
                                if prev['SMA5'] < prev['SMA20'] and latest['SMA5'] > latest['SMA20']:
                                    sinyaller.append("Golden Cross: 5-günlük SMA 20-günlük SMA'yı yukarı kesti")
                                    sinyal_sayisi += 1
                                elif latest['SMA5'] > latest['SMA20'] and latest['SMA5'] > prev['SMA5']:
                                    sinyaller.append("Kısa vadeli SMA yükseliş trendinde")
                                    sinyal_sayisi += 0.5
                        except:
                            pass
                    
                    # EMA Kontrol
                    if "EMA" in gostergeler:
                        try:
                            if all(col in latest for col in ['EMA5', 'EMA20']):
                                if prev['EMA5'] < prev['EMA20'] and latest['EMA5'] > latest['EMA20']:
                                    sinyaller.append("EMA Golden Cross: 5-günlük EMA 20-günlük EMA'yı yukarı kesti")
                                    sinyal_sayisi += 1
                                elif latest['EMA5'] > latest['EMA20'] and latest['EMA5'] > prev['EMA5']:
                                    sinyaller.append("Kısa vadeli EMA yükseliş trendinde")
                                    sinyal_sayisi += 0.5
                        except:
                            pass
                    
                    # Stokastik Osilatör Kontrol
                    if "Stokastik" in gostergeler:
                        try:
                            if all(col in latest for col in ['Stoch_%K', 'Stoch_%D']):
                                if prev['Stoch_%K'] < 20 and latest['Stoch_%K'] > 20 and latest['Stoch_%K'] > latest['Stoch_%D']:
                                    sinyaller.append("Stokastik aşırı satım bölgesinden çıkış sinyali")
                                    sinyal_sayisi += 1
                                elif prev['Stoch_%K'] < prev['Stoch_%D'] and latest['Stoch_%K'] > latest['Stoch_%D']:
                                    sinyaller.append("Stokastik %K, %D'yi yukarı kesti")
                                    sinyal_sayisi += 0.5
                        except:
                            pass
                    
                    # ADX Kontrol (Trend gücü)
                    if "ADX" in gostergeler:
                        try:
                            if 'ADX' in latest:
                                if latest['ADX'] > 25:
                                    # Güçlü trend varsa yön kontrolü yap
                                    if 'SMA5' in latest and 'SMA20' in latest and latest['SMA5'] > latest['SMA20']:
                                        sinyaller.append(f"Güçlü yükseliş trendi (ADX: {latest['ADX']:.1f})")
                                        sinyal_sayisi += 0.5
                        except:
                            pass
                    
                    # Hacim Analizi
                    if "Hacim Analizi" in gostergeler:
                        try:
                            # Ortalama hacim hesapla
                            avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1]
                            
                            if latest['Volume'] > 1.5 * avg_volume and latest['Close'] > prev['Close']:
                                sinyaller.append("Yüksek hacimle yükseliş")
                                sinyal_sayisi += 1
                            elif latest['Volume'] > 1.2 * avg_volume and latest['Close'] > prev['Close']:
                                sinyaller.append("Ortalamanın üzerinde hacim")
                                sinyal_sayisi += 0.5
                        except:
                            pass
                    
                    # Temel Formasyon Analizi
                    if "Formasyonlar" in gostergeler:
                        try:
                            # Çift Dip Formasyonu Kontrolü (Basit tespit)
                            if len(df) >= 20:
                                recent_lows = df['Low'].rolling(window=3).min().iloc[-20:]
                                min_indices = recent_lows[recent_lows == recent_lows.min()].index
                                
                                if len(min_indices) >= 2 and (min_indices[-1] - min_indices[0]).days >= 5:
                                    price_diff_pct = abs(df.loc[min_indices[0]]['Low'] - df.loc[min_indices[-1]]['Low']) / df.loc[min_indices[0]]['Low']
                                    
                                    if price_diff_pct <= 0.03:  # %3 içinde benzer diplerle
                                        sinyaller.append("Olası Çift Dip Formasyonu")
                                        sinyal_sayisi += 1
                        except:
                            pass
                    
                    # En az bir sinyal varsa listeye ekle
                    if sinyal_sayisi > 0:
                        # Son fiyat değişim yüzdesi
                        son_fiyat = latest['Close']
                        bir_hafta_once = df['Close'][-5] if len(df) >= 5 else df['Close'][0]
                        
                        degisim_yuzde = ((son_fiyat - bir_hafta_once) / bir_hafta_once) * 100
                        
                        results.append({
                            'Sembol': sembol,
                            'Son Fiyat': son_fiyat,
                            'Değişim (%)': f"{degisim_yuzde:.2f}%",
                            'Sinyal Sayısı': sinyal_sayisi,
                            'Sinyaller': ", ".join(sinyaller),
                            'Hacim': latest['Volume'],
                        })
                
                except Exception as e:
                    st.error(f"{sembol} analizi sırasında hata: {str(e)}")
                    continue
            
            progress_bar.progress(1.0)
            progress_text.text("Analiz tamamlandı!")
            
            # Sonuçları göster
            if not results:
                st.warning("Yükseliş potansiyeli gösteren hisse bulunamadı.")
            else:
                # Sonuçları sinyal sayısına göre sırala
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values('Sinyal Sayısı', ascending=False)
                
                # Fiyat değişimine göre renklendirme fonksiyonu
                def color_negative_red(val):
                    try:
                        if '%' in val:
                            # Değişim değeri
                            val_num = float(val.replace('%', ''))
                            return 'color: red' if val_num < 0 else 'color: green'
                    except:
                        pass
                    return ''
                
                st.subheader(f"Yükseliş Potansiyeli Olan Hisseler ({len(results_df)})")
                st.dataframe(results_df.style.applymap(color_negative_red, subset=['Değişim (%)']), use_container_width=True)
                
                # Potansiyel en yüksek hisseler
                if len(results_df) > 0:
                    st.subheader("En Güçlü Yükseliş Sinyalleri")
                    
                    # En yüksek sinyal sayısına sahip hisseler
                    top_stocks = results_df.head(3)
                    
                    for i, (idx, row) in enumerate(top_stocks.iterrows()):
                        col1, col2 = st.columns([1, 3])
                        
                        with col1:
                            st.subheader(f"{row['Sembol']}")
                            st.metric(
                                label="Son Fiyat", 
                                value=f"{row['Son Fiyat']:.2f} TL",
                                delta=row['Değişim (%)']
                            )
                        
                        with col2:
                            st.markdown(f"**Teknik Sinyaller ({row['Sinyal Sayısı']}):**")
                            st.markdown(f"{row['Sinyaller']}")
                            
                            # Hisse detayına yönlendirme
                            if st.button(f"{row['Sembol']} Analiz Et", key=f"analiz_{i}"):
                                # Analiz sayfasına yönlendir
                                st.session_state.selected_stock_for_analysis = row['Sembol']
                                st.success(f"{row['Sembol']} analiz sayfasına yönlendiriliyorsunuz...")
                                # Not: Bu kısım ana uygulama yapısına göre değiştirilmelidir.
    
    # Açıklama kısmı
    with st.expander("Teknik Göstergeler Hakkında Bilgi"):
        st.markdown("""
        **Teknik Göstergeler ve Anlamları:**
        
        * **RSI (Göreceli Güç İndeksi):** 
          - 30 altı: Aşırı satım bölgesi (alım fırsatı)
          - 70 üstü: Aşırı alım bölgesi (satım fırsatı)
        
        * **MACD (Hareketli Ortalama Yakınsama/Iraksama):**
          - MACD çizgisinin sinyal çizgisini yukarı kesmesi: Alış sinyali
          - MACD çizgisinin sinyal çizgisini aşağı kesmesi: Satış sinyali
        
        * **Bollinger Bantları:**
          - Fiyatın alt bandı yukarı kesmesi: Olası yükseliş
          - Fiyatın üst bandı aşağı kesmesi: Olası düşüş
          - Bantların daralması: Yakın zamanda sert hareket beklentisi
        
        * **SMA (Basit Hareketli Ortalama):**
          - Kısa dönem SMA'nın uzun dönem SMA'yı yukarı kesmesi (Golden Cross): Güçlü alış sinyali
          - Kısa dönem SMA'nın uzun dönem SMA'yı aşağı kesmesi (Death Cross): Güçlü satış sinyali
        
        * **Stokastik Osilatör:**
          - K çizgisinin 20 altından yukarı kesmesi: Alış sinyali
          - K çizgisinin 80 üstünden aşağı kesmesi: Satış sinyali
        
        * **ADX (Ortalama Yön Endeksi):**
          - 25 üzeri: Güçlü trend
          - 20 altı: Zayıf trend
        
        * **Hacim Analizi:**
          - Yükselen fiyatın artan hacimle desteklenmesi: Trend güçlü
          - Düşen fiyatın azalan hacimle gerçekleşmesi: Düşüş zayıflıyor olabilir
        
        * **Formasyonlar:**
          - Çift Dip: Güçlü dönüş sinyali
          - Üçgen Formasyonu: Fiyatın sıkışması ve ardından güçlü hareket
          - Baş-Omuz: Trend dönüşümü
        """) 