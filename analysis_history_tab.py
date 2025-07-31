import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

from data.db_utils import (
    get_detailed_analysis_history,
    export_analysis_results,
    compare_analysis_results,
    delete_analysis_result,
    update_analysis_price
)

def render_analysis_history_tab():
    """
    Analiz geçmişini gösterir ve analiz sonuçlarını yönetir
    """
    st.header("Analiz Geçmişi ve Karşılaştırma", divider="rainbow")
    
    # Filtreler için üst bölüm
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Hisse senedi filtresi
        stock_symbol = st.text_input("Hisse Senedi Kodu", "", placeholder="Örn: THYAO (Boş bırakabilirsiniz)")
    
    with col2:
        # Analiz tipi filtresi
        analysis_type = st.selectbox(
            "Analiz Tipi", 
            ["Tümü", "teknik", "temel", "ml", "sentiment"], 
            index=0
        )
        if analysis_type == "Tümü":
            analysis_type = None
    
    with col3:
        # Tarih filtresi
        date_option = st.selectbox(
            "Tarih Aralığı",
            ["Son 7 Gün", "Son 30 Gün", "Son 90 Gün", "Tüm Zamanlar", "Özel Aralık"],
            index=1
        )
    
    # Özel tarih aralığı seçildiyse
    start_date = None
    end_date = None
    
    if date_option == "Özel Aralık":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Başlangıç Tarihi", value=datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("Bitiş Tarihi", value=datetime.now())
        
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")
    else:
        # Diğer tarih seçenekleri için hesaplama
        end_date = datetime.now().strftime("%Y-%m-%d")
        if date_option == "Son 7 Gün":
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        elif date_option == "Son 30 Gün":
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        elif date_option == "Son 90 Gün":
            start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        elif date_option == "Tüm Zamanlar":
            start_date = None
            end_date = None
    
    # Filtreleme ve sonuçları getirme
    apply_filter = st.button("Sonuçları Getir")
    
    if apply_filter or 'history_tab_initialized' not in st.session_state:
        st.session_state.history_tab_initialized = True
        
        with st.spinner("Analiz sonuçları getiriliyor..."):
            # Analiz geçmişini getir
            results = get_detailed_analysis_history(
                symbol=stock_symbol if stock_symbol else None,
                analysis_type=analysis_type,
                start_date=start_date,
                end_date=end_date,
                limit=100  # En fazla 100 sonuç getir
            )
            
            if results:
                st.success(f"{len(results)} analiz sonucu bulundu.")
                
                # Fiyat değerlerini güncelleme butonu
                if st.button("Fiyat Değerlerini Güncelle"):
                    ml_analyses = [r for r in results if r["analysis_type"] == "ml"]
                    updated_count = 0
                    
                    for analysis in ml_analyses:
                        if analysis["price"] == 0 and "result_data" in analysis:
                            # result_data içinden fiyat bilgisini çek
                            result_data = analysis["result_data"]
                            price = result_data.get("last_price", None)
                            
                            if price is None:
                                price = result_data.get("current_price", 0)
                            
                            if price > 0:
                                # Veritabanında güncelle
                                if update_analysis_price(analysis["id"], price):
                                    updated_count += 1
                    
                    if updated_count > 0:
                        st.success(f"{updated_count} analizin fiyat değeri güncellendi. Sayfayı yenilemek için 'Sonuçları Getir' butonuna tıklayın.")
                    else:
                        st.info("Güncellenecek fiyat değeri bulunamadı.")
                
                # Sonuçları session state'e kaydet
                st.session_state.analysis_history_results = results
                
                # Temel sonuçlar için veri çerçevesi
                basic_results = []
                for r in results:
                    row = {
                        "ID": r["id"],
                        "Sembol": r["symbol"],
                        "Analiz Tipi": r["analysis_type"],
                        "Tarih": r["analysis_date"],
                        "Fiyat": r["price"]
                    }
                    
                    # Analiz tipine göre önemli bilgileri ekle
                    if r["analysis_type"] == "teknik":
                        trend = r["result_data"].get("trend", "")
                        rec = r["result_data"].get("recommendation", "")
                        row["Trend/Sonuç"] = trend
                        row["Tavsiye"] = rec
                    elif r["analysis_type"] == "ml":
                        trend = r["result_data"].get("trend", "")
                        rec = r["result_data"].get("recommendation", "")
                        conf = r["result_data"].get("confidence", 0)
                        row["Trend/Sonuç"] = trend
                        row["Tavsiye"] = rec
                        row["Güven"] = f"%{conf:.1f}" if isinstance(conf, (int, float)) else conf
                    elif r["analysis_type"] == "sentiment":
                        sentiment = r["result_data"].get("sentiment_score", 0)
                        row["Trend/Sonuç"] = f"Duyarlılık: {sentiment:.2f}" if isinstance(sentiment, (int, float)) else sentiment
                    
                    # Not ekle
                    row["Not"] = r["notes"] if r["notes"] else ""
                    
                    basic_results.append(row)
                
                # Sonuçları görüntüle
                df = pd.DataFrame(basic_results)
                st.dataframe(df, hide_index=True, use_container_width=True)
                
                # Aksiyon butonları
                st.subheader("Analiz Sonuçları İşlemleri")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Seçilen analizin detaylarını görüntüleme
                    selected_id = st.selectbox(
                        "Detaylarını Görmek İstediğiniz Analiz ID",
                        options=[r["id"] for r in results],
                        index=0,
                        format_func=lambda x: f"ID: {x} - {next((r['symbol'] + ' (' + r['analysis_type'] + ')' for r in results if r['id'] == x))}"
                    )
                
                with col2:
                    # Dışa aktarma formatı
                    export_format = st.selectbox(
                        "Dışa Aktarma Formatı",
                        options=["CSV", "JSON"],
                        index=0
                    )
                
                with col3:
                    # Karşılaştırma için ikinci analiz
                    compare_id = st.selectbox(
                        "Karşılaştırılacak Analiz ID",
                        options=[0] + [r["id"] for r in results if r["id"] != selected_id],
                        index=0,
                        format_func=lambda x: "Karşılaştırma Yok" if x == 0 else f"ID: {x} - {next((r['symbol'] + ' (' + r['analysis_type'] + ')' for r in results if r['id'] == x), '')}"
                    )
                
                # Detayları görüntüle butonu
                if st.button("Seçilen Analiz Detaylarını Göster"):
                    selected_result = next((r for r in results if r["id"] == selected_id), None)
                    
                    if selected_result:
                        st.subheader(f"{selected_result['symbol']} - {selected_result['analysis_type'].capitalize()} Analiz Detayları")
                        
                        # Temel bilgiler
                        st.markdown(f"**Analiz Tarihi:** {selected_result['analysis_date']}")
                        st.markdown(f"**Fiyat:** {selected_result['price']} TL")
                        
                        # Notlar
                        if selected_result["notes"]:
                            st.markdown(f"**Notlar:** {selected_result['notes']}")
                        
                        # Sonuçlar ve göstergeler için iki sekme oluştur
                        tab1, tab2 = st.tabs(["Analiz Sonuçları", "Teknik Göstergeler"])
                        
                        with tab1:
                            # Sonuç verilerini görüntüle
                            st.json(selected_result["result_data"])
                            
                            # Görsel gösterim (analiz tipine göre)
                            if "recommendation" in selected_result["result_data"]:
                                rec = selected_result["result_data"]["recommendation"]
                                rec_color = "green" if rec in ["AL", "GÜÇLÜ AL"] else ("red" if rec in ["SAT", "GÜÇLÜ SAT"] else "orange")
                                
                                st.markdown(f"<h3 style='text-align: center; color: {rec_color};'>{rec}</h3>", unsafe_allow_html=True)
                        
                        with tab2:
                            # Teknik göstergeleri görüntüle
                            if selected_result["indicators"]:
                                st.json(selected_result["indicators"])
                            else:
                                st.info("Bu analiz için teknik gösterge verisi bulunmamaktadır.")
                
                # Karşılaştırma butonu
                if compare_id != 0 and st.button("Analizleri Karşılaştır"):
                    with st.spinner("Analizler karşılaştırılıyor..."):
                        comparison = compare_analysis_results(selected_id, compare_id)
                        
                        if "error" in comparison:
                            st.error(f"Karşılaştırma hatası: {comparison['error']}")
                        else:
                            st.subheader("Analiz Karşılaştırması")
                            
                            # Temel bilgiler
                            st.markdown("### Temel Bilgiler")
                            basic_info = comparison["basic_info"]
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Sembol:** {basic_info['symbol'][0]} → {basic_info['symbol'][1]}")
                                st.markdown(f"**Analiz Tipi:** {basic_info['analysis_type'][0]} → {basic_info['analysis_type'][1]}")
                            
                            with col2:
                                st.markdown(f"**Tarih:** {basic_info['analysis_date'][0]} → {basic_info['analysis_date'][1]}")
                                
                                price1, price2 = basic_info['price']
                                price_change = basic_info['price_change']
                                price_color = "green" if price_change > 0 else "red"
                                
                                st.markdown(f"**Fiyat:** {price1} TL → {price2} TL <span style='color: {price_color};'>({price_change}%)</span>", unsafe_allow_html=True)
                            
                            # Sonuç verileri karşılaştırması
                            if comparison["result_data"]:
                                st.markdown("### Analiz Sonuçları Karşılaştırması")
                                
                                result_data = []
                                for key, values in comparison["result_data"].items():
                                    if not key.endswith("_pct_change"):
                                        row = {"Parametre": key}
                                        
                                        if isinstance(values[0], (int, float)) and isinstance(values[1], (int, float)):
                                            row["1. Analiz"] = f"{values[0]}"
                                            row["2. Analiz"] = f"{values[1]}"
                                            
                                            pct_key = f"{key}_pct_change"
                                            if pct_key in comparison["result_data"]:
                                                row["Değişim"] = comparison["result_data"][pct_key]
                                        else:
                                            row["1. Analiz"] = f"{values[0]}"
                                            row["2. Analiz"] = f"{values[1]}"
                                            row["Değişim"] = "N/A"
                                        
                                        result_data.append(row)
                                
                                st.table(pd.DataFrame(result_data))
                            
                            # Göstergeler karşılaştırması
                            if comparison["indicators"]:
                                st.markdown("### Teknik Göstergeler Karşılaştırması")
                                
                                indicators_data = []
                                for key, values in comparison["indicators"].items():
                                    if not key.endswith("_pct_change"):
                                        row = {"Gösterge": key}
                                        
                                        if isinstance(values[0], (int, float)) and isinstance(values[1], (int, float)):
                                            row["1. Analiz"] = f"{values[0]:.4f}" if abs(values[0]) < 10 else f"{values[0]:.2f}"
                                            row["2. Analiz"] = f"{values[1]:.4f}" if abs(values[1]) < 10 else f"{values[1]:.2f}"
                                            
                                            pct_key = f"{key}_pct_change"
                                            if pct_key in comparison["indicators"]:
                                                row["Değişim"] = comparison["indicators"][pct_key]
                                        else:
                                            row["1. Analiz"] = f"{values[0]}"
                                            row["2. Analiz"] = f"{values[1]}"
                                            row["Değişim"] = "N/A"
                                        
                                        indicators_data.append(row)
                                
                                st.table(pd.DataFrame(indicators_data))
                
                # Dışa aktarma butonu
                if st.button("Analiz Sonuçlarını Dışa Aktar"):
                    symbol_for_export = stock_symbol if stock_symbol else "TUM"
                    export_data = export_analysis_results(
                        symbol=stock_symbol if stock_symbol else None,
                        format=export_format.lower(),
                        analysis_type=analysis_type,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if export_data:
                        # Dosya adı oluştur
                        file_ext = ".csv" if export_format.lower() == "csv" else ".json"
                        file_name = f"{symbol_for_export}_analiz_sonuclari_{datetime.now().strftime('%Y%m%d')}{file_ext}"
                        
                        # Dosyayı indirmek için link oluştur
                        st.download_button(
                            label=f"{file_name} Dosyasını İndir",
                            data=export_data,
                            file_name=file_name,
                            mime="text/csv" if export_format.lower() == "csv" else "application/json"
                        )
                    else:
                        st.error("Dışa aktarılacak veri bulunamadı.")
                
                # Silme işlemi
                with st.expander("Analiz Sonucu Silme (Dikkatli Kullanın)"):
                    delete_id = st.selectbox(
                        "Silinecek Analiz ID",
                        options=[r["id"] for r in results],
                        index=0,
                        format_func=lambda x: f"ID: {x} - {next((r['symbol'] + ' (' + r['analysis_date'] + ')') for r in results if r['id'] == x)}"
                    )
                    
                    delete_confirm = st.checkbox("Silme işlemini onaylıyorum")
                    
                    if st.button("Seçilen Analizi Sil", disabled=not delete_confirm):
                        # Silme fonksiyonunu çağır (db_utils.py içinde tanımlanmalı)
                        if delete_analysis_result(delete_id):
                            st.success(f"ID: {delete_id} analiz sonucu başarıyla silindi.")
                        else:
                            st.error(f"ID: {delete_id} analiz sonucu silinirken bir hata oluştu.")
            else:
                st.warning("Belirtilen kriterlere uygun analiz sonucu bulunamadı.")
    
    # İstatistikler bölümü
    st.header("Analiz İstatistikleri", divider="rainbow")
    
    # İstatistiklerin hesaplanması
    if 'analysis_history_results' in st.session_state and st.session_state.analysis_history_results:
        results = st.session_state.analysis_history_results
        
        # İstatistiksel bilgileri hesapla
        total_count = len(results)
        
        # Analiz tipine göre sayı
        analysis_types = {}
        for r in results:
            analysis_type = r["analysis_type"]
            if analysis_type in analysis_types:
                analysis_types[analysis_type] += 1
            else:
                analysis_types[analysis_type] = 1
        
        # Hisse başına analiz sayısı
        symbols = {}
        for r in results:
            symbol = r["symbol"]
            if symbol in symbols:
                symbols[symbol] += 1
            else:
                symbols[symbol] = 1
        
        # En çok analiz edilen hisseler (ilk 5)
        top_symbols = dict(sorted(symbols.items(), key=lambda x: x[1], reverse=True)[:5])
        
        # Zaman içindeki analiz sayısı
        dates = {}
        for r in results:
            date = r["analysis_date"].split(" ")[0]  # Sadece tarih kısmı (saat olmadan)
            if date in dates:
                dates[date] += 1
            else:
                dates[date] = 1
        
        # Verileri göster
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Analiz Tipi Dağılımı")
            
            # Pasta grafik
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.pie(
                analysis_types.values(), 
                labels=analysis_types.keys(), 
                autopct='%1.1f%%',
                startangle=90,
                shadow=False
            )
            ax.axis('equal')
            st.pyplot(fig)
        
        with col2:
            st.subheader("En Çok Analiz Edilen Hisseler")
            
            # Çubuk grafik
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(list(top_symbols.keys()), list(top_symbols.values()))
            ax.set_xlabel("Analiz Sayısı")
            ax.invert_yaxis()  # En büyük değeri en üstte göster
            st.pyplot(fig)
        
        # Zaman serisi grafiği
        st.subheader("Zamana Göre Analiz Sayısı")
        
        # Tarihleri sırala
        sorted_dates = sorted(dates.items())
        dates_list = [item[0] for item in sorted_dates]
        counts_list = [item[1] for item in sorted_dates]
        
        # Plotly ile çizgi grafik
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dates_list,
                y=counts_list,
                mode='lines+markers',
                name='Analiz Sayısı',
                line=dict(color='royalblue', width=2),
                marker=dict(size=8)
            )
        )
        
        fig.update_layout(
            title='Günlük Analiz Sayısı Trendi',
            xaxis_title='Tarih',
            yaxis_title='Analiz Sayısı',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Özet metrik gösterimi
        st.subheader("Özet İstatistikler")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Toplam Analiz", total_count)
        
        with metric_col2:
            # Teknik analiz sayısı
            teknik_count = analysis_types.get("teknik", 0)
            st.metric("Teknik Analiz", teknik_count)
        
        with metric_col3:
            # ML analiz sayısı
            ml_count = analysis_types.get("ml", 0)
            st.metric("ML Analiz", ml_count)
        
        with metric_col4:
            # Analiz sıklığı (gün başına)
            if len(dates) > 1:
                avg_per_day = total_count / len(dates)
                st.metric("Gün Başına Analiz", f"{avg_per_day:.1f}")
            else:
                st.metric("Gün Başına Analiz", "N/A")
    else:
        st.info("İstatistikler için önce analiz sonuçlarını getirin.") 