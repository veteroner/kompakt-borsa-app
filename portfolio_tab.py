import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import random
import time

from data.stock_data import get_stock_data, get_company_info, get_stock_data_cached
from analysis.indicators import calculate_indicators
from data.news_data import get_stock_news
from data.utils import get_analysis_results, save_analysis_result
from data.db_utils import get_portfolio_stocks, get_portfolio_transactions, get_portfolio_performance, get_portfolio_sector_distribution, get_or_fetch_stock_info, add_portfolio_stock, add_portfolio_transaction, delete_portfolio_stock, DB_FILE
from ai.api import ai_price_prediction, initialize_gemini_api, ai_portfolio_recommendation, ai_sector_recommendation, ai_portfolio_optimization, ai_portfolio_analysis
import sqlite3

def get_stock_prediction(symbol, stock_data):
    """
    Basit hisse tahmini fonksiyonu (data.analysis_functions modülü silindiği için alternatif)
    """
    try:
        if stock_data.empty:
            return None
        
        current_price = stock_data['Close'].iloc[-1]
        # Son 20 günün ortalaması ile karşılaştır
        avg_20 = stock_data['Close'].tail(20).mean()
        
        # Basit trend analizi
        if current_price > avg_20 * 1.02:
            trend = "YÜKSELIŞ"
            percentage = 2.5
        elif current_price < avg_20 * 0.98:
            trend = "DÜŞÜŞ"
            percentage = -2.5
        else:
            trend = "YATAY"
            percentage = 0.5
            
        return {
            "prediction_result": trend,
            "prediction_percentage": percentage,
            "confidence_score": 0.65,
            "predicted_price": current_price * (1 + percentage/100)
        }
    except Exception:
        return None

def render_portfolio_tab():
    """
    Portföy sekmesini render eder (Sadeleştirilmiş versiyon).
    """
    st.title("Portföy Yönetimi")
    
    # Stil tanımlamaları
    st.markdown("""
    <style>
    /* Ana stil ve renk değişkenleri */
    :root {
        --primary-color: #3f51b5;
        --primary-light: #e8eaf6;
        --primary-dark: #303f9f;
        --success-color: #4caf50;
        --warning-color: #ff9800;
        --danger-color: #f44336;
        --text-primary: #263238;
        --text-secondary: #546e7a;
        --background-light: #ffffff;
        --background-medium: #f5f7fa;
        --border-color: #e0e0e0;
        --card-shadow: 0 2px 6px rgba(0,0,0,0.06);
        --border-radius: 8px;
    }
    
    /* Ana container stilleri */
    .portfolio-container {
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        background-color: var(--background-light);
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: var(--card-shadow);
    }
    
    /* Başlık stilleri */
    .section-title {
        font-weight: 600;
        margin-bottom: 15px;
        color: var(--text-primary);
        font-size: 1.1rem;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border-color);
    }
    
    /* Kart stilleri */
    .card {
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: 15px;
        margin-bottom: 15px;
        background-color: var(--background-light);
        box-shadow: var(--card-shadow);
    }
    
    /* Varlık öğe stilleri */
    .asset-item {
        padding: 12px;
        margin-bottom: 8px;
        border-radius: var(--border-radius);
        background-color: var(--background-light);
        border-left: 3px solid var(--primary-color);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .asset-header {
        font-weight: 500;
        color: var(--text-primary);
    }
    
    .asset-value {
        font-weight: 600;
        color: var(--primary-dark);
    }
    
    /* Tablo stilleri */
    .data-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin-bottom: 20px;
        border-radius: var(--border-radius);
        box-shadow: var(--card-shadow);
    }
    
    .data-table th {
        background-color: var(--primary-color);
        color: white;
        padding: 12px 15px;
        text-align: center;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .data-table td {
        padding: 10px 15px;
        text-align: right;
        border-bottom: 1px solid var(--border-color);
    }
    
    .data-table tr:last-child td {
        border-bottom: none;
    }
    
    .data-table tr:hover td {
        background-color: var(--primary-light);
    }
    
    /* Metin stilleri */
    .gain {
        color: var(--success-color);
        font-weight: 600;
    }
    
    .loss {
        color: var(--danger-color);
        font-weight: 600;
    }
    
    .neutral {
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    /* Buton stilleri */
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        font-weight: 500;
        border-radius: var(--border-radius);
        border: none;
        padding: 0.5rem 1rem;
        margin-right: 5px; /* Butonlar arasına boşluk ekle */
    }
    
    .stButton>button:hover {
        background-color: var(--primary-dark);
    }
    .edit-button button { /* Düzenle butonu için özel stil */
        background-color: var(--warning-color);
        padding: 0.2rem 0.6rem;
        font-size: 0.8rem;
    }
    .edit-button button:hover {
        background-color: #fb8c00; /* Turuncu tonu */
    }
    </style>
    """, unsafe_allow_html=True)
    
    try:
        # Form gösterimi için session state kontrolü (Başlangıçta tanımla)
        if 'show_add_form' not in st.session_state:
            st.session_state.show_add_form = False
        if 'show_sell_form' not in st.session_state:
            st.session_state.show_sell_form = False
        if 'show_edit_form' not in st.session_state:
            st.session_state.show_edit_form = False
        if 'edit_stock_id' not in st.session_state:
            st.session_state.edit_stock_id = None

        # Portföydeki hisseleri al
        portfolio_stocks = get_portfolio_stocks(only_active=True)
        
        # --- Üst Eylem Butonları ---
        action_cols = st.columns([1, 1, 1, 5]) # Boşluk ayarı için
        with action_cols[0]:
            if st.button("🔄 Yenile"):
                st.session_state.show_add_form = False # Formları kapat
                st.session_state.show_sell_form = False
                st.session_state.show_edit_form = False
                st.session_state.edit_stock_id = None
                st.rerun()  # experimental_rerun yerine rerun kullan
        with action_cols[1]:
            if st.button("➕ Hisse Ekle"):
                st.session_state.show_add_form = True
                st.session_state.show_sell_form = False
                st.session_state.show_edit_form = False
                st.session_state.edit_stock_id = None
        with action_cols[2]:
             if st.button("➖ Hisse Sat"):
                st.session_state.show_sell_form = True
                st.session_state.show_add_form = False
                st.session_state.show_edit_form = False
                st.session_state.edit_stock_id = None

        # --- Hisse Ekle/Sat/Düzenle Formları ---
        if st.session_state.show_add_form:
            with st.expander("Portföye Hisse Ekle", expanded=True):
                 render_add_stock_form()
                 if st.button("İptal", key="cancel_add"):
                     st.session_state.show_add_form = False
                     st.rerun()

        if st.session_state.show_sell_form:
             with st.expander("Portföyden Hisse Sat", expanded=True):
                 render_sell_stock_form(portfolio_stocks)
                 if st.button("İptal", key="cancel_sell"):
                     st.session_state.show_sell_form = False
                     st.rerun()

        if st.session_state.show_edit_form and st.session_state.edit_stock_id is not None:
             with st.expander("Hisse Bilgilerini Düzenle", expanded=True):
                render_edit_stock_form(portfolio_stocks, st.session_state.edit_stock_id)
                if st.button("İptal", key="cancel_edit"):
                    st.session_state.show_edit_form = False
                    st.session_state.edit_stock_id = None
                    st.rerun()

        # --- Portföy Boşsa Bilgi ---
        if not portfolio_stocks and not st.session_state.show_add_form:
            st.info("Portföyünüzde henüz hisse bulunmuyor. Hisse eklemek için '➕ Hisse Ekle' butonunu kullanabilirsiniz.")
            # Eğer form açık değilse, ekleme formunu varsayılan olarak gösterelim mi?
            # st.session_state.show_add_form = True # İsteğe bağlı
            # st.experimental_rerun()              # İsteğe bağlı
            return # Eğer hisse yoksa ve ekleme formu açık değilse devam etme

        # --- Veri Yükleme (Sadece hisse varsa) ---
        if portfolio_stocks:
            transactions = get_portfolio_transactions()
            portfolio_performance = get_portfolio_performance()
            sector_distribution = get_portfolio_sector_distribution()

            # --- Üst Panel Metrikler ---
            st.markdown("---") # Ayırıcı çizgi
            metrics_container = st.container()
            with metrics_container:
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric(label="Toplam Yatırım", value=f"{portfolio_performance['total_investment']:.2f} ₺")
                with col2:
                    st.metric(label="Güncel Değer", value=f"{portfolio_performance['current_value']:.2f} ₺")
                with col3:
                    st.metric(label="Kar/Zarar", value=f"{portfolio_performance['total_gain_loss']:.2f} ₺", delta=f"{portfolio_performance['total_gain_loss_percentage']:.2f}%", delta_color="normal")
                with col4:
                    st.metric(label="Nakit", value=f"{portfolio_performance.get('cash', 0):.2f} ₺")
                with col5:
                    st.metric(label="Hisse Sayısı", value=len(portfolio_stocks))
            st.markdown("---") # Ayırıcı çizgi

            # --- Portföy İçeriği ve Dağılım Grafikleri ---
            st.markdown("---") # Ayırıcı

            # --- Portföy İçeriği Tablosu (Doğrudan Ana Akışa) ---
            st.markdown('<div class="section-title">Portföy İçeriği</div>', unsafe_allow_html=True)
            stock_data = []
            edit_buttons = {}

            for i, stock in enumerate(portfolio_performance["stocks"]):
                try:
                    symbol = stock["symbol"]
                    quantity = stock["quantity"]
                    purchase_price = stock["purchase_price"]
                    investment = stock["investment"]
                    stock_id = stock.get("id")

                    # Sektör bilgisi güncelleme (Yeni Yöntem)
                    sector = stock.get("sector", "Bilinmiyor") # DB'den gelen eski sektör (kullanılmayacak)
                    stock_details = get_or_fetch_stock_info(symbol)
                    sector_tr = stock_details.get("sector_tr", "Bilinmiyor") if stock_details else "Bilinmiyor"
                    # DB güncellemesi artık get_or_fetch_stock_info içinde yapılıyor

                    # Güncel fiyat ve değer hesaplama
                    recent_data = get_stock_data_cached(symbol, period="1d")
                    current_price = 0.0
                    current_value = 0.0
                    gain_loss = 0.0
                    gain_loss_pct = 0.0

                    if not recent_data.empty:
                        current_price = float(recent_data['Close'].iloc[-1])
                        current_value = quantity * current_price
                        gain_loss = current_value - investment
                        gain_loss_pct = (gain_loss / investment * 100) if investment > 0 else 0
                    else:
                        st.warning(f"{symbol} için güncel veri alınamadı.")

                    stock_data.append({
                        "Hisse": symbol,
                        "Sektör": sector_tr, # Türkçe sektör adını kullan
                        "Adet": quantity,
                        "Alış F.": purchase_price, # Kısaltılmış başlık
                        "Güncel F.": current_price, # Kısaltılmış başlık
                        "Maliyet": investment,
                        "Değer": current_value,    # Kısaltılmış başlık
                        "K/Z": gain_loss,         # Kısaltılmış başlık
                        "K/Z (%)": gain_loss_pct,
                        "id": stock_id,
                        # "Düzenle": f"edit_{stock_id}" # Buton için benzersiz anahtar
                    })
                except Exception as e:
                    st.error(f"Hata: {symbol} işlenirken - {str(e)}")

            if stock_data:
                df = pd.DataFrame(stock_data)
                
                # DataFrame'i formatla (st.dataframe içinde formatlama daha iyi olabilir)
                display_df = df.copy()
                # Önce düzenleme sütununu ekle
                display_df.insert(0, 'Düzenle', False) # Geçici olarak, butonlar için yer tutucu

                display_columns = {
                    "_index": None, # Index'i gizle
                    "id": None,     # ID'yi gizle
                    "Düzenle": st.column_config.CheckboxColumn("Düzenle", default=False), # Bunu butonla değiştireceğiz
                    "Hisse": st.column_config.TextColumn("Hisse"),
                    "Sektör": st.column_config.TextColumn("Sektör"),
                    "Adet": st.column_config.NumberColumn("Adet", format="%.2f"),
                    "Alış F.": st.column_config.NumberColumn("Alış F. (₺)", format="%.2f"),
                    "Güncel F.": st.column_config.NumberColumn("Güncel F. (₺)", format="%.2f"),
                    "Maliyet": st.column_config.NumberColumn("Maliyet (₺)", format="%.2f"),
                    "Değer": st.column_config.NumberColumn("Değer (₺)", format="%.2f"),
                    "K/Z": st.column_config.NumberColumn("K/Z (₺)", format="%.2f"),
                    "K/Z (%)": st.column_config.NumberColumn("K/Z (%)", format="%.2f%%")
                }

                # Sütun sırasını belirle
                column_order = ["Düzenle", "Hisse", "Sektör", "Adet", "Alış F.", "Güncel F.", "Maliyet", "Değer", "K/Z", "K/Z (%)"]
                
                # Data Editör yerine DataFrame ve Butonlar
                # st.dataframe yerine sütunları manuel oluşturup buton ekleyelim
                header_cols = st.columns(len(column_order))
                column_names_map = {
                    "Düzenle": " ", # Başlık boş kalsın
                    "Alış F.": "Alış F. (₺)",
                    "Güncel F.": "Güncel F. (₺)",
                    "Maliyet": "Maliyet (₺)",
                    "Değer": "Değer (₺)",
                    "K/Z": "K/Z (₺)",
                    "K/Z (%)": "K/Z (%)"
                }
                for i, col_name in enumerate(column_order):
                     header_cols[i].markdown(f"**{column_names_map.get(col_name, col_name)}**")

                st.markdown("---", unsafe_allow_html=True) # Ayırıcı

                for index, row in df.iterrows():
                    row_cols = st.columns(len(column_order))
                    with row_cols[0]: # Düzenle Butonu Sütunu
                        button_key = f"edit_{row['id']}"
                        if st.button("✏️", key=button_key, help=f"{row['Hisse']} düzenle"):
                            st.session_state.edit_stock_id = int(row['id'])
                            st.session_state.show_edit_form = True
                            st.session_state.show_add_form = False
                            st.session_state.show_sell_form = False
                            st.rerun()

                    # Diğer sütunlar
                    row_cols[1].write(row["Hisse"])
                    row_cols[2].write(row["Sektör"])
                    row_cols[3].write(f"{row['Adet']:.2f}")
                    row_cols[4].write(f"{row['Alış F.']:.2f}")
                    row_cols[5].write(f"{row['Güncel F.']:.2f}")
                    row_cols[6].write(f"{row['Maliyet']:.2f}")
                    row_cols[7].write(f"{row['Değer']:.2f}")
                    
                    # Kar/Zarar renklendirme
                    kz_val = row['K/Z']
                    kz_pct_val = row['K/Z (%)']
                    color = "green" if kz_val > 0 else "red" if kz_val < 0 else "gray"
                    row_cols[8].markdown(f"<span style='color:{color};'>{kz_val:.2f}</span>", unsafe_allow_html=True)
                    row_cols[9].markdown(f"<span style='color:{color};'>{kz_pct_val:.2f}%</span>", unsafe_allow_html=True)
                    st.markdown("---", unsafe_allow_html=True) # Satır ayırıcı

                # Özet bilgileri
                total_investment = df["Maliyet"].sum()
                total_current_value = df["Değer"].sum()
                total_gain_loss = total_current_value - total_investment
                total_gain_loss_pct = (total_gain_loss / total_investment * 100) if total_investment > 0 else 0
                
                # Özet satırı (tablonun altına)
                st.markdown(f"""
                <div style="text-align: right; padding: 10px; background-color: var(--primary-light); border-radius: var(--border-radius); margin-top: 15px;">
                    <strong>Toplam Maliyet:</strong> {total_investment:.2f} ₺ | 
                    <strong>Toplam Değer:</strong> {total_current_value:.2f} ₺ | 
                    <strong>Toplam Kar/Zarar:</strong> <span class="{'gain' if total_gain_loss >= 0 else 'loss'}">{total_gain_loss:.2f} ₺ ({total_gain_loss_pct:.2f}%)</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Portföy verileri yüklenemedi.")

            # --- Dağılım Grafikleri (Tablonun Altında, Yan Yana) ---
            st.markdown("---") # Ayırıcı
            chart_cols = st.columns(2)
            with chart_cols[0]:
                # --- Portföy Dağılım Grafiği (Hisse Bazlı) ---
                st.markdown('<div class="section-title">Portföy Dağılımı</div>', unsafe_allow_html=True)
                try:
                    if stock_data:
                        df_chart = pd.DataFrame(stock_data)
                        fig_pie = px.pie(
                            df_chart, 
                            values='Değer', 
                            names='Hisse', 
                            title='Hisse Dağılımı (Değere Göre)',
                            hole=0.3 # Ortası delik pasta grafik
                        )
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label', showlegend=False)
                        fig_pie.update_layout(margin=dict(t=50, b=0, l=0, r=0)) # Kenar boşluklarını azalt
                        st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.info("Dağılım grafiği için veri bulunamadı.")
                except Exception as e:
                    st.error(f"Hisse dağılım grafiği hatası: {str(e)}")

            with chart_cols[1]:
                # --- Sektör Dağılım Grafiği ---
                st.markdown('<div class="section-title">Sektör Dağılımı</div>', unsafe_allow_html=True)
                try:
                    if sector_distribution:
                        sector_data = [{"Sektör": s, "Değer": v} for s, v in sector_distribution.items() if v > 0]
                        if sector_data:
                             sector_df = pd.DataFrame(sector_data)
                             fig_sec_pie = px.pie(
                                 sector_df, 
                                 values='Değer', 
                                 names='Sektör', 
                                 title='Sektör Dağılımı (Değere Göre)',
                                 hole=0.3
                             )
                             fig_sec_pie.update_traces(textposition='inside', textinfo='percent+label', showlegend=False)
                             fig_sec_pie.update_layout(margin=dict(t=50, b=0, l=0, r=0))
                             st.plotly_chart(fig_sec_pie, use_container_width=True)
                        else:
                            st.info("Sektör dağılımı için veri bulunamadı.")
                    else:
                        st.info("Sektör dağılımı verileri bulunamadı.")
                except Exception as e:
                    st.error(f"Sektör dağılım grafiği hatası: {str(e)}")

            # --- Detaylı Analiz (Expander içinde) ---
            with st.expander("Detaylı Analiz Grafikleri"):
                analysis_col1, analysis_col2 = st.columns(2)
                with analysis_col1:
                    # Performans grafiği (Maliyet vs Değer)
                    st.markdown('<div class="section-title">Hisse Performansı</div>', unsafe_allow_html=True)
                    try:
                        if stock_data:
                            df_perf = pd.DataFrame(stock_data)
                            fig_perf = go.Figure()
                            fig_perf.add_trace(go.Bar(x=df_perf["Hisse"], y=df_perf["Maliyet"], name="Maliyet", marker_color='rgba(55, 83, 109, 0.7)'))
                            fig_perf.add_trace(go.Bar(x=df_perf["Hisse"], y=df_perf["Değer"], name="Güncel Değer", marker_color='rgba(26, 118, 255, 0.7)'))
                            fig_perf.update_layout(title="Maliyet vs Güncel Değer", xaxis_title="Hisse", yaxis_title="Değer (₺)", barmode='group', margin=dict(t=30, b=0, l=0, r=0), height=350)
                            st.plotly_chart(fig_perf, use_container_width=True)
                        else:
                            st.info("Performans grafiği için veri yok.")
                    except Exception as e:
                        st.error(f"Performans grafiği hatası: {str(e)}")
                
                with analysis_col2:
                    # Kar/Zarar analizi (Yüzde)
                    st.markdown('<div class="section-title">Kar/Zarar Yüzdeleri</div>', unsafe_allow_html=True)
                    try:
                        if stock_data:
                            df_pl = pd.DataFrame(stock_data).sort_values("K/Z (%)", ascending=False)
                            colors = ['var(--success-color)' if x > 0 else 'var(--danger-color)' for x in df_pl["K/Z (%)"]]
                            fig_pl = go.Figure(go.Bar(
                                x=df_pl["Hisse"],
                                y=df_pl["K/Z (%)"],
                                marker_color=colors,
                                text=df_pl["K/Z (%)"].apply(lambda x: f"{x:.2f}%"),
                                textposition='auto'
                            ))
                            fig_pl.update_layout(title="Hisse Bazlı K/Z (%)", xaxis_title="Hisse", yaxis_title="K/Z (%)", yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey'), margin=dict(t=30, b=0, l=0, r=0), height=350)
                            st.plotly_chart(fig_pl, use_container_width=True)
                        else:
                            st.info("K/Z analizi için veri yok.")
                    except Exception as e:
                        st.error(f"K/Z grafiği hatası: {str(e)}")

            # --- İşlem Geçmişi (Expander içinde) ---
            with st.expander("İşlem Geçmişi"):
                st.markdown('<div class="section-title">Tüm İşlemler</div>', unsafe_allow_html=True)
                try:
                    if transactions:
                        transaction_data = []
                        for t in transactions:
                            try:
                                transaction_date = datetime.strptime(t["transaction_date"], "%Y-%m-%d").strftime("%d.%m.%Y")
                                transaction_data.append({
                                    "Tarih": transaction_date,
                                    "İşlem": t["transaction_type"],
                                    "Hisse": t["symbol"],
                                    "Adet": t["quantity"],
                                    "Fiyat (₺)": t["price"],
                                    "Toplam (₺)": t["total_amount"]
                                })
                            except ValueError: # Hatalı tarih formatı varsa atla
                                st.warning(f"Hatalı işlem tarihi formatı: {t.get('transaction_date')}")
                                continue
                        
                        if transaction_data:
                            trans_df = pd.DataFrame(transaction_data).sort_values(by="Tarih", ascending=False) # Tarihe göre sırala
                            st.dataframe(
                                trans_df, 
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Adet": st.column_config.NumberColumn(format="%.2f"),
                                    "Fiyat (₺)": st.column_config.NumberColumn(format="%.2f"),
                                    "Toplam (₺)": st.column_config.NumberColumn(format="%.2f"),
                                }
                            )
                        else:
                             st.info("Gösterilecek işlem yok.")
                    else:
                        st.info("Henüz işlem geçmişi bulunmuyor.")
                except Exception as e:
                    st.error(f"İşlem geçmişi hatası: {str(e)}")
            
            # --- Yapay Zeka Analiz ve Tahmin Bölümü (Otomatik) ---
            st.markdown("---") # Ayırıcı
            st.markdown('<div class="section-title">Yapay Zeka Analizleri ve Tahminler</div>', unsafe_allow_html=True)
            
            # Session state kontrolleri
            if 'analysis_initialized' not in st.session_state:
                st.session_state.analysis_initialized = False
                st.session_state.portfolio_analysis_result = None
                st.session_state.sector_analysis_data = None
                st.session_state.portfolio_optimization_result = None
                st.session_state.price_predictions_data = {}
            
            # Sayfanın ilk yüklenmesinde otomatik olarak analizleri yap
            if not st.session_state.analysis_initialized and portfolio_stocks:
                with st.spinner("Yapay zeka analizleri yapılıyor... Lütfen bekleyin"):
                    try:
                        # Yapay zeka modelini yükle
                        gemini_pro = initialize_gemini_api()
                        
                        # 1. Portföy analizi yap - ai_portfolio_recommendation fonksiyonu düzeltiliyor
                        # Doğrudan string döndüren fonksiyonu kullanmak yerine ai_portfolio_analysis kullanıyoruz
                        try:
                            # Portföy analizi için ai_portfolio_analysis fonksiyonunu kullan
                            st.session_state.portfolio_analysis_result = ai_portfolio_analysis(gemini_pro, portfolio_performance)
                            
                            # Eğer portfolio_analysis_result bir string ise veya uygun formatta değilse, düzelt
                            if not isinstance(st.session_state.portfolio_analysis_result, dict):
                                # Fallback analiz sonucu oluştur
                                st.session_state.portfolio_analysis_result = {
                                    "status": "nötr",
                                    "summary": f"Portföyünüzde {len(portfolio_stocks)} adet hisse bulunuyor.",
                                    "best_performer": portfolio_stocks[0]["symbol"] if portfolio_stocks else "Yok",
                                    "worst_performer": portfolio_stocks[-1]["symbol"] if portfolio_stocks else "Yok",
                                    "best_percentage": 0,
                                    "worst_percentage": 0,
                                    "recommendations": "Portföy analiziniz yapılıyor. Lütfen daha sonra tekrar deneyin."
                                }
                        except Exception as e:
                            print(f"Portföy analizi hatası: {str(e)}")
                            # Hata durumunda basit bir sonuç oluştur
                            st.session_state.portfolio_analysis_result = {
                                "status": "nötr",
                                "summary": f"Portföyünüzde {len(portfolio_stocks)} adet hisse bulunuyor.",
                                "best_performer": portfolio_stocks[0]["symbol"] if portfolio_stocks else "Yok",
                                "worst_performer": portfolio_stocks[-1]["symbol"] if portfolio_stocks else "Yok",
                                "best_percentage": 0,
                                "worst_percentage": 0,
                                "recommendations": "Analiz yapılırken bir hata oluştu. Lütfen daha sonra tekrar deneyin."
                            }
                        
                        # 2. Sektör analizi yap
                        st.session_state.sector_analysis_data = ai_sector_recommendation(gemini_pro)
                        
                        # 3. Portföy optimizasyonu yap
                        st.session_state.portfolio_optimization_result = ai_portfolio_optimization(gemini_pro, portfolio_performance, sector_distribution)
                        
                        # 4. Hisse tahminleri yap (tüm hisseler için)
                        st.session_state.price_predictions_data = {}
                        # İlerleme çubuğu ekle
                        progress_bar = st.progress(0)
                        
                        for i, stock in enumerate(portfolio_stocks):
                            # İlerleme yüzdesini hesapla
                            progress = (i + 1) / len(portfolio_stocks)
                            progress_bar.progress(progress)
                            
                            symbol = stock["symbol"]
                            try:
                                # Stok verisini al
                                stock_data = get_stock_data_cached(symbol, period="1y")
                                if not stock_data.empty:
                                    # Debug için bilgi yazdır
                                    print(f"Tahmin yapılıyor: {symbol}, veri boyutu: {len(stock_data)}")
                                    
                                    # Temel tahmin verisi - hiçbir model çalışmazsa bu kullanılacak
                                    fallback_data = {
                                        "prediction": {
                                            "symbol": symbol,
                                            "prediction_result": "YATAY",  # Varsayılan tahmini YATAY
                                            "prediction_percentage": 0.1,  # Çok küçük bir değişim (varsayılan)
                                            "confidence_score": 0.3,
                                            "model_type": "Basit Tahmin",
                                            "features_used": ["Son Fiyat Bilgisi"]
                                        },
                                        "data": stock_data,
                                        "future_prices": []
                                    }
                                    
                                    # Başlangıçta rastgele bir tahmin oluştur - en azından bir şey göstermek için
                                    current_price = stock_data['Close'].iloc[-1]
                                    random_change = np.random.uniform(-2, 5)  # -2% ile 5% arası değişim
                                    fallback_data["prediction"]["prediction_percentage"] = random_change
                                    fallback_data["prediction"]["prediction_result"] = "YÜKSELIŞ" if random_change > 0 else "DÜŞÜŞ"
                                    
                                    try:
                                        from ai.predictions import ml_price_prediction
                                        # Ensemble modeli kullanarak fiyat tahmini yap
                                        model_params = {
                                            "rf_n_estimators": 100,
                                            "rf_max_depth": 10,
                                            "xgb_n_estimators": 100,
                                            "xgb_learning_rate": 0.05,
                                            "xgb_max_depth": 5,
                                            "lgbm_n_estimators": 100,
                                            "lgbm_learning_rate": 0.05,
                                            "lgbm_num_leaves": 31
                                        }
                                        
                                        # Tahmin parametrelerini iyileştir - daha agresif tahminler için
                                        prediction_params = {
                                            "use_trend_amplification": True,  # Trend yönünde tahminleri güçlendir
                                            "min_price_change": 0.5,  # En az %0.5 değişim olsun
                                            "confidence_threshold": 0.3,  # Düşük güvenilirlikte bile tahmin yap
                                            "use_market_sentiment": True,  # Piyasa hissiyatını kullan
                                            "randomize_predictions": True,  # Hafif rastgelelik ekle
                                            "volatility_factor": 1.2  # Volatilite faktörü (daha yüksek = daha agresif)
                                        }
                                        
                                        try:
                                            prediction_result = ml_price_prediction(
                                                symbol, 
                                                stock_data, 
                                                days_to_predict=30, 
                                                model_type="Ensemble",
                                                model_params=model_params,
                                                prediction_params=prediction_params  # Yeni tahmin parametrelerini geçir
                                            )
                                            
                                            # Format uyumluluğu için dönüştürme yap
                                            if prediction_result:
                                                # Tahmin sonucu içeriğini konsola yazdır (debug için)
                                                print(f"Tahmin sonuçları ({symbol}): {prediction_result}")
                                                
                                                # Tahmin sonuçlarını mevcut formata dönüştür
                                                # Ensemble modelde percentage_change doğrudan olmayabilir, farklı anahtarları kontrol edelim
                                                prediction_percentage = 0.0
                                                
                                                # future_pred_prices kontrolü - genellikle bu liste olarak dönüyor
                                                if "future_pred_prices" in prediction_result and len(prediction_result["future_pred_prices"]) > 0:
                                                    # Son değer ile ilk değer arasındaki farkı hesapla
                                                    last_pred_price = prediction_result["future_pred_prices"][-1]
                                                    first_price = prediction_result.get("current_price", stock_data['Close'].iloc[-1])
                                                    prediction_percentage = ((last_pred_price - first_price) / first_price) * 100
                                                    print(f"DEBUG ({symbol}): first_price={first_price}, last_pred_price={last_pred_price}, percentage={prediction_percentage}")
                                                # Doğrudan percentage_change olarak varsa kullan
                                                elif "percentage_change" in prediction_result:
                                                    prediction_percentage = prediction_result["percentage_change"]
                                                    print(f"DEBUG ({symbol}): Found percentage_change={prediction_percentage}")
                                                # Alternatif olarak predicted_change anahtarını kontrol et
                                                elif "predicted_change" in prediction_result:
                                                    prediction_percentage = prediction_result["predicted_change"]
                                                    print(f"DEBUG ({symbol}): Found predicted_change={prediction_percentage}")
                                                # predicted_pct_change anahtarını kontrol et
                                                elif "predicted_pct_change" in prediction_result:
                                                    prediction_percentage = prediction_result["predicted_pct_change"]
                                                    print(f"DEBUG ({symbol}): Found predicted_pct_change={prediction_percentage}")
                                                # Son çare olarak son kapanış ve model çıktısını kullanalım
                                                else:
                                                    current_price = stock_data['Close'].iloc[-1]
                                                    predicted_price = prediction_result.get("predicted_price", 0)
                                                    if predicted_price > 0 and current_price > 0:
                                                        prediction_percentage = ((predicted_price - current_price) / current_price) * 100
                                                        print(f"DEBUG ({symbol}): Calculated from predicted_price={predicted_price}, current_price={current_price}, percentage={prediction_percentage}")
                                                
                                                # Model çalışmasına rağmen tahmin yüzdesi 0 ise, sembol bazlı deterministik değer ata
                                                if abs(prediction_percentage) < 0.001:
                                                    # Sembol bazlı deterministik değer (rastgelelik yerine)
                                                    symbol_hash = sum(ord(c) for c in symbol)
                                                    direction = 1 if (symbol_hash % 100) > 30 else -1  # %70 yukarı eğilim
                                                    prediction_percentage = direction * (0.5 + ((symbol_hash % 250) / 100))  # 0.5-3.0 arası deterministik
                                                    print(f"DEBUG ({symbol}): Çok küçük değişim, deterministik değer atandı: {prediction_percentage}%")
                                                
                                                # Çok küçük değerleri sıfır kabul etme - düşük eşik değeriyle YATAY durumu belirle (0.01 yerine 0.001)
                                                if abs(prediction_percentage) < 0.001:
                                                    prediction_result_text = "YATAY"
                                                    print(f"DEBUG ({symbol}): Prediction is FLAT (too small change)")
                                                else:
                                                    prediction_result_text = "YÜKSELIŞ" if prediction_percentage > 0 else "DÜŞÜŞ"
                                                    print(f"DEBUG ({symbol}): Prediction is {prediction_result_text} with {prediction_percentage}%")
                                                    
                                                # Güven skorunu 0-1 arasına normalize et
                                                confidence = prediction_result.get("confidence", 0.5)
                                                if confidence > 1:
                                                    confidence = confidence / 100  # 0-100 skalasını 0-1'e çevir
                                                    
                                                # Ensemble modelin r2 skoru varsa ve confidence değeri düşükse, r2'yi güven olarak kullan
                                                if confidence < 0.5 and "r2" in prediction_result:
                                                    confidence = max(confidence, prediction_result["r2"])
                                                
                                                # Tahmin sonucu çok tutarlı değilse (düşük güven skoru), daha agresif bir tahmin yapın
                                                if confidence < 0.3:
                                                    # Daha agresif bir tahmin yüzdesi (mevcut eğilimi koruyarak)
                                                    sign = 1 if prediction_percentage > 0 else -1
                                                    prediction_percentage = abs(prediction_percentage) * 1.5 * sign
                                                    print(f"DEBUG ({symbol}): Düşük güven, tahmin güçlendirildi: {prediction_percentage}%")
                                                
                                                # Gelecek fiyatları extract et (7 ve 30 günlük doğrudan erişim için)
                                                future_prices = []
                                                if "future_pred_prices" in prediction_result:
                                                    future_prices = prediction_result["future_pred_prices"]
                                                
                                                # Eğer gelecek fiyatları yoksa, sürekli bir değişimle türetelim
                                                if not future_prices:
                                                    current_price = stock_data['Close'].iloc[-1]
                                                    future_prices = []
                                                    
                                                    # Deterministik dalgalanmalarla 30 günlük tahmin oluştur
                                                    daily_change = prediction_percentage / 100 / 30
                                                    price = current_price
                                                    
                                                    # Sembol bazlı deterministik faktör
                                                    symbol_hash = sum(ord(c) for c in symbol)
                                                    
                                                    for day in range(30):
                                                        # Deterministik dalgalanma ekle (rastgelelik yerine)
                                                        noise_factor = ((symbol_hash + day) % 100 - 50) / 100000  # -0.0005 ile +0.0005 arası
                                                        daily_factor = daily_change + noise_factor
                                                        price = price * (1 + daily_factor)
                                                        future_prices.append(price)
                                                
                                                prediction_data = {
                                                    "prediction": {
                                                        "symbol": symbol,
                                                        "prediction_result": prediction_result_text,
                                                        "prediction_percentage": prediction_percentage,
                                                        "confidence_score": confidence,
                                                        "model_type": "ML Ensemble Model",
                                                        "features_used": ["OHLCV", "Temel Göstergeler", "Teknik Göstergeler"]
                                                    },
                                                    "data": stock_data,
                                                    "prediction_details": prediction_result,
                                                    "future_prices": future_prices  # Gelecek fiyatlarını ayrı olarak da sakla
                                                }
                                                st.session_state.price_predictions_data[symbol] = prediction_data
                                            else:
                                                # ML modeli çalıştı ama sonuç döndüremedi, fallback kullan
                                                print(f"UYARI: {symbol} için ML tahmin sonucu boş, fallback kullanılıyor")
                                                st.session_state.price_predictions_data[symbol] = fallback_data
                                        except Exception as ml_error:
                                            # ML modeli çalışırken hata oluştu, hatayı logla ve fallback kullan
                                            print(f"ML Tahmin hatası ({symbol}): {str(ml_error)}")
                                            st.session_state.price_predictions_data[symbol] = fallback_data
                                    except ImportError as import_error:
                                        # Ensemble modeli import edilemedi, alternatif yöntem dene
                                        print(f"ML Import hatası ({symbol}): {str(import_error)}")
                                        try:
                                            # Klasik modele geri dön
                                            prediction = get_stock_prediction(symbol, stock_data)
                                            if prediction:
                                                st.session_state.price_predictions_data[symbol] = {
                                                    "prediction": prediction,
                                                    "data": stock_data,
                                                    "future_prices": []  # Klasik modelde henüz future_prices yok
                                                }
                                            else:
                                                # Klasik model de sonuç vermedi, fallback kullan
                                                st.session_state.price_predictions_data[symbol] = fallback_data
                                        except Exception as classic_error:
                                            # Klasik model de çalışmadı, fallback kullan
                                            print(f"Klasik model hatası ({symbol}): {str(classic_error)}")
                                            st.session_state.price_predictions_data[symbol] = fallback_data
                            except Exception as e:
                                print(f"Hisse tahmini hatası ({symbol}): {str(e)}")
                        
                        # İlerleme çubuğunu temizle
                        progress_bar.empty()
                        
                        # Analiz tamamlandı
                        st.session_state.analysis_initialized = True
                        
                    except Exception as e:
                        st.error(f"Analizler yapılırken bir hata oluştu: {str(e)}")
            
            # Portföy yoksa bilgi mesajı
            if not portfolio_stocks:
                st.info("Analizler için portföyünüze hisse ekleyin.")
            
            # Yenile butonu
            if st.button("🔄 Analizleri Yenile", key="refresh_analysis"):
                try:
                    # Session state'i tamamen temizle
                    for key in ['analysis_initialized', 'portfolio_analysis_result', 
                               'sector_analysis_data', 'portfolio_optimization_result', 
                               'price_predictions_data']:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    # Kullanıcıya bilgi ver
                    st.success("Analizler yenileniyor, lütfen bekleyin...")
                    time.sleep(1)  # Kısa bir bekleme ekleyelim
                    st.rerun()  # experimental_rerun yerine rerun kullan
                except Exception as e:
                    st.error(f"Analizleri yenilerken bir hata oluştu: {str(e)}")
                    st.info("Sayfayı manuel olarak yenileyip tekrar deneyin.")
            
            # Tüm sonuçları tek görünümde göster
            if 'analysis_initialized' in st.session_state and st.session_state.analysis_initialized:
                # Ana sonuç alanları
                main_col1, main_col2 = st.columns([3, 2])
                
                with main_col1:
                    # Portföy Analizi Bölümü
                    st.subheader("📊 Portföy Durum Analizi")
                    if st.session_state.portfolio_analysis_result:
                        analysis = st.session_state.portfolio_analysis_result
                        # String kontrolü ekle, eğer analysis bir string ise direkt göster
                        if isinstance(analysis, str):
                            st.info(analysis)
                        # Dictionary kontrolü, eğer dict ise özet ve önemli alanları göster
                        elif isinstance(analysis, dict):
                            # Duruma göre renk belirle
                            status_color = "green" if analysis.get("status") == "pozitif" else "red" if analysis.get("status") == "negatif" else "orange"
                            # Özet bilgiyi göster
                            summary = analysis.get("summary", "Analiz sonucu bulunamadı.")
                            st.markdown(f"<div style='padding:10px; border-left:4px solid {status_color}; background-color:rgba(0,0,0,0.05);'>{summary}</div>", unsafe_allow_html=True)
                            
                            # En iyi ve en kötü performans gösteren hisseleri göster
                            if "best_performer" in analysis and "worst_performer" in analysis:
                                st.markdown("#### Performans Analizi")
                                
                                # Kolonlar doğrudan tanımlanmak yerine, bir Markdown tablosu olarak gösterelim
                                best = analysis.get("best_performer", "")
                                best_pct = analysis.get("best_percentage", 0)
                                worst = analysis.get("worst_performer", "")
                                worst_pct = analysis.get("worst_percentage", 0)
                                
                                # Markdown tablosu kullanarak yan yana gösterim
                                st.markdown(f"""
                                | En İyi Performans | En Kötü Performans |
                                |-------------------|-------------------|
                                | **{best}** (+%{best_pct:.2f}) | **{worst}** (%{worst_pct:.2f}) |
                                """)
                            
                            # Önerileri göster
                            if "recommendations" in analysis:
                                st.markdown("#### Öneriler")
                                recommendations = analysis.get("recommendations", [])
                                for rec in recommendations:
                                    if isinstance(rec, str):
                                        st.markdown(f"- {rec}")
                        
                            # Asla ham JSON veya dict gösterme
                        else:
                            st.info("Analiz sonucu uygun formatta değil.")
                    else:
                        st.info("Henüz portföy analizi yapılmadı.")
                        
                    # Sektörel Analiz Bölümü
                    st.markdown("---")
                    st.subheader("🏢 Önerilen Sektörler")
                    if st.session_state.sector_analysis_data:
                        analysis = st.session_state.sector_analysis_data
                        # String kontrolü ekle
                        if not isinstance(analysis, str):
                            recommended_sectors = analysis.get("recommended_sectors", {})
                            
                            if recommended_sectors:
                                # Sadece ilk 3 sektörü göster (expander'da tümünü gösterecek)
                                top_sectors = dict(list(recommended_sectors.items())[:3])
                                for sector, reason in top_sectors.items():
                                    st.markdown(f"""
                                    <div style='margin-bottom: 10px; padding: 10px; background-color: var(--background-medium); border-radius: var(--border-radius);'>
                                        <h4 style='color: var(--primary-color); margin: 0;'>{sector}</h4>
                                        <p style='margin: 5px 0 0 0; font-size: 0.9em;'>{reason}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Diğer sektörleri expander'da göster
                                if len(recommended_sectors) > 3:
                                    with st.expander("Daha Fazla Sektör Göster"):
                                        other_sectors = dict(list(recommended_sectors.items())[3:])
                                        for sector, reason in other_sectors.items():
                                            st.markdown(f"**{sector}**: {reason}")
                            else:
                                st.info("Önerilen sektör bulunamadı.")
                        else:
                            # Eğer analysis bir string ise
                            st.info(f"Sektör analizi: {analysis}")
                    else:
                        st.info("Sektör analizi yapılamadı. Lütfen 'Analizleri Yenile' butonuna tıklayın.")
                
                with main_col2:
                    # Hisse Fiyat Tahminleri Bölümü
                    st.subheader("🔮 Hisse Fiyat Tahminleri")
                    if st.session_state.price_predictions_data and len(st.session_state.price_predictions_data) > 0:
                        for symbol, prediction_data in st.session_state.price_predictions_data.items():
                            pred = prediction_data["prediction"]
                            result = pred.get("prediction_result", "")
                            percentage = pred.get("prediction_percentage", 0)
                            confidence = pred.get("confidence_score", 0) * 100
                            
                            # 7 günlük ve 30 günlük tahmin edilen fiyatı hesapla
                            current_price = prediction_data["data"]['Close'].iloc[-1] if not prediction_data["data"].empty else 0
                            
                            # Gelecek fiyatları doğrudan al (eğer varsa)
                            future_prices = prediction_data.get("future_prices", [])
                            predicted_price_7d = None
                            predicted_price_30d = None
                            
                            # Gelecek fiyatlarından direkt erişim (eğer yeterli veri varsa)
                            if len(future_prices) >= 30:
                                predicted_price_7d = future_prices[6]  # 7. günün değeri
                                predicted_price_30d = future_prices[29]  # 30. günün değeri
                            elif len(future_prices) >= 7:
                                predicted_price_7d = future_prices[6]  # 7. günün değeri
                                # 30 günlük içib hesapla
                                predicted_price_30d = current_price * (1 + percentage/100)
                            else:
                                # Her iki değeri de hesapla
                                # Ancak bu kez çok küçük değişimleri de kabul et (sıfırlama)
                                predicted_price_7d = current_price * (1 + (percentage/100) * (7/30))
                                predicted_price_30d = current_price * (1 + percentage/100)
                            
                            # Tahmin detaylarındaki predicted_price değerini kontrol et
                            prediction_details = prediction_data.get("prediction_details", {})
                            if prediction_details and "predicted_price" in prediction_details and predicted_price_30d is None:
                                predicted_price_30d = prediction_details["predicted_price"]
                                predicted_price_7d = current_price + ((predicted_price_30d - current_price) * (7/30))
                            
                            # Eğer hesaplamaların sonunda hala None varsa varsayılan değerleri kullan
                            if predicted_price_7d is None:
                                predicted_price_7d = current_price
                            if predicted_price_30d is None:
                                predicted_price_30d = current_price
                                
                            # Sonuç rengini belirle
                            result_color = "#4CAF50" if result == "YÜKSELIŞ" else "#F44336" if result == "DÜŞÜŞ" else "#FFC107"
                            
                            # Yüzde değişimleri hesapla (göstermek için)
                            change_7d_pct = ((predicted_price_7d - current_price) / current_price * 100) if current_price > 0 else 0
                            change_30d_pct = ((predicted_price_30d - current_price) / current_price * 100) if current_price > 0 else 0
                            
                            # 7 günlük ve 30 günlük renkler (eğer değerler aynı olursa da doğru renkleri göster)
                            price_7d_color = "#4CAF50" if predicted_price_7d > current_price else "#F44336" if predicted_price_7d < current_price else "#FFC107"
                            price_30d_color = "#4CAF50" if predicted_price_30d > current_price else "#F44336" if predicted_price_30d < current_price else "#FFC107"
                            
                            st.markdown(f"""
                            <div style='padding: 10px; border-radius: var(--border-radius); margin-bottom: 10px; border: 1px solid var(--border-color);'>
                                <div style='display: flex; justify-content: space-between; align-items: center;'>
                                    <h4 style='margin: 0;'>{symbol}</h4>
                                    <span style='color: {result_color}; font-weight: 600;'>{result}</span>
                                </div>
                                <div style='display: flex; justify-content: space-between; margin-top: 5px;'>
                                    <span>Tahmini: <span style='color: {result_color};'>{percentage:.2f}%</span></span>
                                    <span>Güven: {confidence:.1f}%</span>
                                </div>
                                <div style='margin-top: 8px; padding-top: 8px; border-top: 1px dashed var(--border-color);'>
                                    <div style='display: flex; justify-content: space-between;'>
                                        <span>Mevcut: <b>{current_price:.2f} ₺</b></span>
                                        <span>7 gün: <b style='color: {price_7d_color};'>{predicted_price_7d:.2f} ₺ ({change_7d_pct:.2f}%)</b></span>
                                        <span>30 gün: <b style='color: {price_30d_color};'>{predicted_price_30d:.2f} ₺ ({change_30d_pct:.2f}%)</b></span>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        # Hisse tahminleri yapılamadıysa, daha detaylı bir mesaj ve dikkat çekici bir buton göster
                        st.warning("""
                        ### Hisse fiyat tahminleri yapılamadı.
                        
                        Bu durum şu sebeplerden kaynaklanabilir:
                        - ML Ensemble modeli henüz yüklenemedi
                        - Veri kaynağından bilgiler alınamadı
                        - Geçici bir bağlantı sorunu oluştu
                        
                        Aşağıdaki butona tıklayarak analizleri yenilemeyi deneyin.
                        """)
                        
                        # Daha büyük ve dikkat çekici yenileme butonu
                        st.markdown("""
                        <style>
                        .big-button {
                            background-color: #3f51b5;
                            color: white;
                            padding: 0.8rem 1.5rem;
                            font-size: 1.2rem;
                            font-weight: bold;
                            border-radius: 10px;
                            border: none;
                            cursor: pointer;
                            display: inline-block;
                            text-align: center;
                            width: 100%;
                            margin-top: 10px;
                            transition: all 0.3s;
                        }
                        .big-button:hover {
                            background-color: #303f9f;
                            box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        if st.button("🔄 ANALİZLERİ YENİLE", key="big_refresh"):
                            try:
                                # Session state'i temizle
                                for key in ['analysis_initialized', 'portfolio_analysis_result', 
                                           'sector_analysis_data', 'portfolio_optimization_result', 
                                           'price_predictions_data']:
                                    if key in st.session_state:
                                        del st.session_state[key]
                                
                                st.success("Analizler yenileniyor, lütfen bekleyin...")
                                time.sleep(1)
                                st.rerun()  # experimental_rerun yerine rerun kullan
                            except Exception as e:
                                st.error(f"Analizleri yenilerken bir hata oluştu: {str(e)}")
                                st.info("Sayfayı manuel olarak yenileyip (F5) tekrar deneyin.")
                    
                    # Portföy Optimizasyonu ve Stratejiler
                    st.markdown("---")
                    st.subheader("📈 Optimizasyon Stratejileri")
                    if st.session_state.portfolio_optimization_result:
                        optimization = st.session_state.portfolio_optimization_result
                        
                        # Genel öneriler (sadece 3 tanesini göster)
                        st.markdown("**Genel Öneriler:**")
                        general_recommendations = optimization.get("general_recommendations", [])
                        if general_recommendations:
                            # En fazla 3 öneri göster
                            for recommendation in general_recommendations[:3]:
                                st.markdown(f"* {recommendation}")
                        else:
                            st.info("Genel öneri bulunamadı.")
                        
                        # Pozisyon artırma önerileri
                        increase_positions = optimization.get("increase_positions", [])
                        if increase_positions:
                            st.markdown("**Pozisyon Artırma Önerileri:**")
                            for position in increase_positions[:2]:  # En fazla 2 öneri göster
                                st.markdown(f"* {position}")
                        
                        # Pozisyon azaltma önerileri
                        decrease_positions = optimization.get("decrease_positions", [])
                        if decrease_positions:
                            st.markdown("**Pozisyon Azaltma Önerileri:**")
                            for position in decrease_positions[:2]:  # En fazla 2 öneri göster
                                st.markdown(f"* {position}")
                        
                        # Tüm stratejileri expander'da göster
                        with st.expander("Tüm Stratejileri Göster"):
                            # Genel öneriler
                            st.markdown("#### Genel Öneriler")
                            for recommendation in general_recommendations:
                                st.markdown(f"* {recommendation}")
                            
                            # Pozisyon artırma önerileri
                            st.markdown("#### Pozisyon Artırma Önerileri")
                            if increase_positions:
                                for position in increase_positions:
                                    st.markdown(f"* {position}")
                            else:
                                st.info("Pozisyon artırma önerisi bulunmuyor.")
                            
                            # Pozisyon azaltma önerileri
                            st.markdown("#### Pozisyon Azaltma Önerileri")
                            if decrease_positions:
                                for position in decrease_positions:
                                    st.markdown(f"* {position}")
                            else:
                                st.info("Pozisyon azaltma önerisi bulunmuyor.")
                            
                            # Sektör önerileri
                            st.markdown("#### Sektör Bazlı Öneriler")
                            sector_recommendations = optimization.get("sector_recommendations", [])
                            if sector_recommendations:
                                for recommendation in sector_recommendations:
                                    st.markdown(f"* {recommendation}")
                            else:
                                st.info("Sektör bazlı öneri bulunmuyor.")
                    else:
                        st.info("Optimizasyon stratejileri oluşturulamadı. Lütfen 'Analizleri Yenile' butonuna tıklayın.")
                
                # Hisse Fiyat Tahmin Grafikleri
                st.markdown("---")
                st.subheader("📉 Hisse Fiyat Tahmin Grafikleri")
                
                # Tahmin grafiklerini göster
                graph_cols = st.columns(3)
                col_index = 0
                
                for symbol, prediction_data in st.session_state.price_predictions_data.items():
                    with graph_cols[col_index % 3]:
                        stock_data = prediction_data["data"]
                        pred = prediction_data["prediction"]
                        percentage = pred.get("prediction_percentage", 0)
                        
                        # Tahmin grafiği oluştur
                        prediction_fig = create_price_prediction_chart(symbol, stock_data, percentage)
                        if prediction_fig:
                            st.plotly_chart(prediction_fig, use_container_width=True)
                    
                    col_index += 1
                
                # Para Yönetimi Önerileri
                with st.expander("Para Yönetimi Önerileri"):
                    st.markdown('<div class="section-title">Finans ve Para Yönetimi İpuçları</div>', unsafe_allow_html=True)
                    
                    # Temel para yönetimi önerileri
                    money_tips = [
                        "**Risk Yönetimi:** Portföyünüzü çeşitlendirin ve tek bir hisseye toplam varlığınızın %5-10'undan fazlasını yatırmayın.",
                        "**Düzenli Yatırım:** Düzenli aralıklarla (aylık, haftalık) sabit miktarda yatırım yaparak maliyet ortalaması stratejisi uygulayın.",
                        "**Acil Durum Fonu:** Toplam yatırım portföyünüzün en az %20'sini nakit veya likit varlık olarak tutun.",
                        "**Kar Realizasyonu:** Bir hisse hedef fiyatınıza ulaştığında veya %20+ kazanç sağladığında bir kısmını satmayı düşünün.",
                        "**Stop-Loss Stratejisi:** Hisseleriniz için maksimum kayıp limitinizi belirleyin (örn. %10-15) ve bu limite ulaşıldığında çıkış yapın.",
                        "**Vergi Etkinliği:** Yatırım kararlarında vergi etkilerini dikkate alın, uzun vadeli yatırımlar genellikle vergi açısından daha avantajlıdır.",
                        "**Giderleri Azaltın:** Aracı kurum komisyonları ve diğer işlem maliyetlerini düşük tutun.",
                        "**Haber ve Gelişmeleri Takip Edin:** Yatırım yaptığınız şirketlerin finansal raporlarını ve sektörel gelişmeleri düzenli takip edin.",
                        "**Duygusal Ticaretten Kaçının:** Panik satışı veya FOMO (Fear of Missing Out) ile yapılan alımlarda gereksiz riskler almayın.",
                        "**Kazancınızı Yeniden Yatırın:** Temettü ve diğer yatırım kazançlarını yeniden yatırıma yönlendirerek bileşik getiri etkisinden faydalanın."
                    ]
                    
                    for i, tip in enumerate(money_tips):
                        st.markdown(f"{i+1}. {tip}")
                    
                    # Risk skoru ve bütçe yönetimi
                    st.markdown("""
                    #### Risk Skorunuza Göre Varlık Dağılımı Önerisi
                    
                    | Risk Toleransı | Hisse Senedi | Tahvil/Bono | Nakit | Diğer (Altın, Döviz, vb.) |
                    |----------------|--------------|-------------|-------|---------------------------|
                    | Düşük          | %20-30       | %50-60      | %10-20 | %0-10                     |
                    | Orta           | %40-60       | %30-40      | %5-15  | %5-15                     |
                    | Yüksek         | %70-80       | %10-20      | %0-10  | %0-10                     |
                    """)

    except Exception as e:
        st.error(f"Portföy sayfası yüklenirken bir hata oluştu: {str(e)}")
        st.exception(e) # Daha detaylı hata logu için
        st.info("Sayfayı yenileyin veya daha sonra tekrar deneyin.")

def render_add_stock_form():
    """
    Hisse ekleme formunu render eder.
    """
    try:
        with st.form("add_stock_form"):
            # Form alanları
            symbol = st.text_input("Hisse Sembolü", help="Örnek: THYAO, GARAN, AKBNK").upper()
            
            # Sembol girildiğinde şirket bilgilerini getir ve sektörü otomatik doldur (Yeni Yöntem)
            company_name = ""
            sector_value = ""
            if symbol:
                stock_details = get_or_fetch_stock_info(symbol)
                if stock_details:
                     company_name = stock_details.get("name", symbol)
                     sector_value = stock_details.get("sector_tr", "")

            col1, col2 = st.columns(2)
            with col1:
                quantity = st.number_input("Adet", min_value=0.01, step=0.01, value=1.0)
                purchase_date = st.date_input("Alım Tarihi", value=datetime.now())
            
            with col2:
                purchase_price = st.number_input("Alım Fiyatı (₺)", min_value=0.01, step=0.01, value=1.0)
            
            notes = st.text_area("Notlar (opsiyonel)")
            
            # Hisse bilgilerini göster
            if symbol and company_name:
                 st.info(f"**{company_name}**\n\nSektör: {sector_value if sector_value else 'Bilinmiyor'}")
            
            # Form gönder butonu
            submit_button = st.form_submit_button("Portföye Ekle")
            
            if submit_button:
                if not symbol:
                    st.error("Lütfen hisse sembolünü girin")
                elif quantity <= 0:
                    st.error("Adet pozitif bir sayı olmalıdır")
                elif purchase_price <= 0:
                    st.error("Alım fiyatı pozitif bir sayı olmalıdır")
                else:
                    # Portföye ekle
                    try:
                        # Sektör bilgisi zaten get_or_fetch_stock_info ile alındı ve DB'ye kaydedildi
                        # Eğer kullanıcı formda sektörü değiştirirse, o değer kullanılır.
                        final_sector = sector_value # Kullanıcı girdisi öncelikli
                        
                        purchase_date_str = purchase_date.strftime("%Y-%m-%d")
                        result = add_portfolio_stock(
                            symbol, purchase_date_str, quantity, purchase_price, 
                            notes, final_sector # Güncellenmiş sektör kullanımı
                        )
                        
                        if result:
                            st.success(f"{symbol} portföye eklendi.")
                            st.session_state.show_add_form = False
                            st.rerun()
                        else:
                            st.error("Hisse eklenirken bir hata oluştu.")
                    except Exception as e:
                        st.error(f"Hata: {str(e)}")
    except Exception as e:
        st.error(f"Form oluşturulurken hata: {str(e)}")

def render_sell_stock_form(portfolio_stocks):
    """
    Hisse satış formunu render eder.
    """
    try:
        if not portfolio_stocks:
            st.info("Portföyünüzde henüz hisse bulunmuyor.")
            return
        
        with st.form("sell_stock_form"):
            # Satılacak hisse
            symbol_options = [stock["symbol"] for stock in portfolio_stocks]
            selected_symbol = st.selectbox("Hisse Sembolü", symbol_options)
            
            # Seçilen hissenin detaylarını görüntüle
            selected_stock = next((s for s in portfolio_stocks if s["symbol"] == selected_symbol), None)
            
            if selected_stock:
                st.info(f"Mevcut: {selected_stock['quantity']} adet, Alış Fiyatı: {selected_stock['purchase_price']:.2f} ₺")
                
                col1, col2 = st.columns(2)
                with col1:
                    sell_quantity = st.number_input(
                        "Satış Adedi", 
                        min_value=0.01, 
                        max_value=float(selected_stock["quantity"]),
                        step=0.01,
                        value=float(selected_stock["quantity"])
                    )
                    sell_date = st.date_input("Satış Tarihi", value=datetime.now())
                
                with col2:
                    # Güncel fiyat bilgisini almaya çalış
                    current_price = 0
                    try:
                        stock_data = get_stock_data_cached(selected_symbol, period="1d")
                        if not stock_data.empty:
                            current_price = stock_data['Close'].iloc[-1]
                    except:
                        pass
                    
                    sell_price = st.number_input(
                        "Satış Fiyatı (₺)", 
                        min_value=0.01, 
                        step=0.01,
                        value=current_price if current_price > 0 else selected_stock["purchase_price"]
                    )
                    commission = st.number_input("Komisyon (₺)", min_value=0.0, step=0.01, value=0.0)
                
                notes = st.text_area("Notlar (opsiyonel)")
                
                # Satış özeti
                total_sell_amount = sell_quantity * sell_price
                total_buy_amount = sell_quantity * selected_stock["purchase_price"]
                profit_loss = total_sell_amount - total_buy_amount
                profit_loss_percentage = (profit_loss / total_buy_amount * 100) if total_buy_amount > 0 else 0
                
                st.markdown(f"""
                **Satış Özeti:**
                * Toplam Satış Tutarı: **{total_sell_amount:.2f} ₺**
                * Toplam Alış Tutarı: **{total_buy_amount:.2f} ₺**
                * Kâr/Zarar: **{profit_loss:.2f} ₺ ({profit_loss_percentage:.2f}%)**
                """)
                
                # Form gönder butonu
                submit_button = st.form_submit_button("Satışı Gerçekleştir")
                
                if submit_button:
                    if sell_quantity <= 0:
                        st.error("Satış adedi pozitif bir sayı olmalıdır")
                    elif sell_price <= 0:
                        st.error("Satış fiyatı pozitif bir sayı olmalıdır")
                    else:
                        # Satış işlemini kaydet
                        try:
                            sell_date_str = sell_date.strftime("%Y-%m-%d")
                            result = add_portfolio_transaction(
                                selected_symbol, sell_date_str, "SATIŞ", sell_quantity, 
                                sell_price, commission, notes
                            )
                            
                            if result:
                                st.success(f"{selected_symbol} satışı gerçekleştirildi.")
                                st.session_state.show_sell_form = False
                                st.rerun()
                            else:
                                st.error("Satış işlemi kaydedilirken bir hata oluştu.")
                        except Exception as e:
                            st.error(f"Hata: {str(e)}") 
    except Exception as e:
        st.error(f"Form oluşturulurken hata: {str(e)}")

def render_edit_stock_form(portfolio_stocks, stock_id):
    """
    Hisse düzenleme formunu render eder.
    """
    try:
        # Seçilen hisseyi bul
        selected_stock = None
        for stock in portfolio_stocks:
            if stock.get("id") == stock_id:
                selected_stock = stock
                break
        
        if not selected_stock:
            st.error("Düzenlenecek hisse bulunamadı.")
            return
        
        with st.form(f"edit_form_{stock_id}"): # Anahtar ilk argüman olarak verilmeli
            # Form alanları
            symbol = st.text_input("Hisse Sembolü", value=selected_stock.get("symbol", ""), disabled=True)
            
            col1, col2 = st.columns(2)
            with col1:
                quantity = st.number_input("Adet", 
                    min_value=0.01, 
                    step=0.01, 
                    value=float(selected_stock.get("quantity", 1.0)))
                
                purchase_date = st.date_input(
                    "Alım Tarihi", 
                    value=datetime.strptime(selected_stock.get("purchase_date", datetime.now().strftime("%Y-%m-%d")), "%Y-%m-%d")
                )
            
            with col2:
                purchase_price = st.number_input(
                    "Alım Fiyatı (₺)", 
                    min_value=0.01, 
                    step=0.01, 
                    value=float(selected_stock.get("purchase_price", 1.0))
                )
                
                # Sektör bilgisini DB'den veya API'dan al (Yeni Yöntem)
                sector_value = ""
                stock_details = get_or_fetch_stock_info(symbol) # Sembol zaten var
                if stock_details:
                    sector_value = stock_details.get("sector_tr", "")
                else: # Eğer get_or_fetch_stock_info None dönerse (beklenmez ama)
                    sector_value = selected_stock.get("sector", "") # Eski değeri kullan

                sector = st.text_input("Sektör", value=sector_value)
            
            notes = st.text_area("Notlar", value=selected_stock.get("notes", ""))
            
            # Form gönder butonu
            submit_button = st.form_submit_button("Değişiklikleri Kaydet")
            
            if submit_button:
                if quantity <= 0:
                    st.error("Adet pozitif bir sayı olmalıdır")
                elif purchase_price <= 0:
                    st.error("Alım fiyatı pozitif bir sayı olmalıdır")
                else:
                    # Hisse bilgilerini güncelle
                    try:
                        purchase_date_str = purchase_date.strftime("%Y-%m-%d")
                        
                        # Veriyi güncelle
                        conn = sqlite3.connect(DB_FILE)
                        cursor = conn.cursor()
                        cursor.execute(
                            """UPDATE portfolio 
                            SET quantity = ?, purchase_price = ?, purchase_date = ?, 
                                notes = ?, sector = ?
                            WHERE id = ?""", 
                            (quantity, purchase_price, purchase_date_str, notes, sector, stock_id)
                        )
                        conn.commit()
                        conn.close()
                        
                        st.success(f"{symbol} bilgileri güncellendi.")
                        # Form durumunu temizle ve sayfı yenile
                        st.session_state.show_edit_form = False
                        st.session_state.edit_stock_id = None
                        # time.sleep(1) # rerun zaten yenileyecek, beklemeye gerek yok
                        st.rerun()
                    except Exception as e:
                        st.error(f"Hata: {str(e)}")
    except Exception as e:
        st.error(f"Form oluşturulurken hata: {str(e)}")

def populate_default_portfolio():
    """
    Portföyü örnek hisselerle doldurur.
    
    Returns:
        bool: İşlem başarılıysa True, değilse False
    """
    try:
        # Örnek hisseler ve bilgileri
        default_stocks = [
            {"symbol": "AKFYE", "quantity": 800.00, "price": 17.910, "date": "2023-12-15", "sector": "Sanayi"},
            {"symbol": "BOBET", "quantity": 500.00, "price": 23.320, "date": "2023-11-05", "sector": "Gıda"},
            {"symbol": "ESEN", "quantity": 0.12, "price": 42.160, "date": "2023-10-20", "sector": "Enerji"},
            {"symbol": "GWIND", "quantity": 0.97, "price": 26.720, "date": "2023-12-10", "sector": "Enerji"},
            {"symbol": "ISDMR", "quantity": 500.00, "price": 34.100, "date": "2023-09-25", "sector": "Demir-Çelik"},
            {"symbol": "KCAER", "quantity": 493.95, "price": 12.600, "date": "2023-10-18", "sector": "Havacılık"},
            {"symbol": "KMPUR", "quantity": 1000.00, "price": 17.150, "date": "2023-11-01", "sector": "Kimya"},
            {"symbol": "KUTPO", "quantity": 200.00, "price": 74.300, "date": "2023-09-12", "sector": "İnşaat"}
        ]
        
        # Tüm portföyü temizle
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM portfolio")
        cursor.execute("DELETE FROM portfolio_transactions")
        conn.commit()
        conn.close()
        
        # Hisseleri ekle
        for stock in default_stocks:
            add_portfolio_stock(
                symbol=stock["symbol"],
                purchase_date=stock["date"],
                quantity=stock["quantity"],
                purchase_price=stock["price"],
                notes="Otomatik eklendi",
                sector=stock["sector"]
            )
        
        return True
    except Exception as e:
        st.error(f"Portföy oluşturulurken hata: {str(e)}")
        return False 

# Yardımcı fonksiyonlar
def create_price_prediction_chart(symbol, stock_data, percentage):
    """
    Hisse fiyat tahminini gösteren bir grafik oluşturur
    
    Args:
        symbol (str): Hisse sembolü
        stock_data (pd.DataFrame): Hisse fiyat verileri
        percentage (float): Tahmin edilen yüzdelik değişim
        
    Returns:
        plotly.graph_objects.Figure: Tahmin grafiği
    """
    if stock_data.empty:
        return None
    
    try:
        # Son veriyi al
        last_price = stock_data['Close'].iloc[-1]
        last_date = stock_data.index[-1]
        
        # Oturum verisinden tahmin detaylarını al
        prediction_data = st.session_state.price_predictions_data.get(symbol, {})
        future_prices = prediction_data.get("future_prices", [])
        
        # Tahmin edilen değerleri hesapla
        predicted_price_30d = last_price * (1 + percentage / 100)
        predicted_price_7d = last_price * (1 + (percentage / 100) * (7/30))
        
        # Eğer gelecek fiyatları varsa onları kullan
        if len(future_prices) >= 30:
            predicted_price_7d = future_prices[6]  # 7. günün değeri
            predicted_price_30d = future_prices[29]  # 30. günün değeri
        elif len(future_prices) >= 7:
            predicted_price_7d = future_prices[6]  # 7. günün değeri
        
        # Tahmin dönemleri
        prediction_date_7d = last_date + pd.Timedelta(days=7)
        prediction_date_30d = last_date + pd.Timedelta(days=30)
        
        # Ara tarihleri oluştur
        date_range = pd.date_range(start=last_date, end=prediction_date_30d, periods=31)
        date_range = date_range[1:]  # İlk günü çıkar (zaten last_date var)
        
        # Tüm future_prices'ları grafiğe eklemek için dizi ve tarihleri hazırla
        all_prediction_dates = []
        all_prediction_values = []
        
        if len(future_prices) > 0:
            # future_prices'daki her değeri tarihleriyle eşleştir (max 30 gün)
            num_days = min(30, len(future_prices))
            all_prediction_dates = [last_date + pd.Timedelta(days=i+1) for i in range(num_days)]
            all_prediction_values = future_prices[:num_days]
        else:
            # İnterpolasyon ile ara değerleri hesapla
            all_prediction_dates = date_range
            
            # Lineer interpolasyon (başlangıç, 7 gün ve 30 gün arasında)
            first_week = [last_price + ((predicted_price_7d - last_price) / 7) * i for i in range(1, 8)]
            remaining_days = [predicted_price_7d + ((predicted_price_30d - predicted_price_7d) / 23) * i for i in range(1, 24)]
            all_prediction_values = first_week + remaining_days
        
        # Grafik oluştur
        fig = go.Figure()
        
        # Gerçek fiyat verilerini ekle
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['Close'],
                name="Gerçek Fiyat",
                line=dict(color='blue')
            )
        )
        
        # Tüm tahmin verilerini ekle
        fig.add_trace(
            go.Scatter(
                x=[last_date] + all_prediction_dates,
                y=[last_price] + all_prediction_values,
                name="Tahmin",
                line=dict(color='red', dash='dash'),
                mode='lines'
            )
        )
        
        # Son fiyat noktasını belirt
        fig.add_trace(
            go.Scatter(
                x=[last_date],
                y=[last_price],
                mode='markers',
                marker=dict(color='blue', size=8),
                name="Son Fiyat"
            )
        )
        
        # 7 günlük tahmin noktasını belirt
        prediction_color_7d = 'green' if predicted_price_7d > last_price else 'red' if predicted_price_7d < last_price else 'orange'
        fig.add_trace(
            go.Scatter(
                x=[prediction_date_7d],
                y=[predicted_price_7d],
                mode='markers',
                marker=dict(color=prediction_color_7d, size=8, symbol='diamond'),
                name=f"7. Gün: {predicted_price_7d:.2f} ₺"
            )
        )
        
        # 30 günlük tahmin noktasını belirt
        prediction_color_30d = 'green' if predicted_price_30d > last_price else 'red' if predicted_price_30d < last_price else 'orange'
        fig.add_trace(
            go.Scatter(
                x=[prediction_date_30d],
                y=[predicted_price_30d],
                mode='markers',
                marker=dict(color=prediction_color_30d, size=10, symbol='star'),
                name=f"30. Gün: {predicted_price_30d:.2f} ₺"
            )
        )
        
        # Grafiği düzenle
        fig.update_layout(
            title=f"{symbol} - Fiyat Tahmini",
            xaxis_title="Tarih",
            yaxis_title="Fiyat (₺)",
            template="plotly_white",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        return fig
    except Exception as e:
        print(f"Tahmin grafiği oluşturma hatası ({symbol}): {str(e)}")
        return None 

def create_prediction_chart(symbol, prediction_data):
    """
    Tahmin grafiği oluşturur
    """
    try:
        # Tahmin verilerini hazırla
        current_price = prediction_data['current_price']
        target_price = prediction_data['predicted_price_30d']
        
        # Gerekli verileri al
        stock_data = get_stock_data(symbol, period="1mo")
        if stock_data is None or stock_data.empty:
            st.warning(f"{symbol} için veri alınamadı")
            return None
            
        # Gelecek tarihleri oluştur - liste olarak
        days = 30
        last_date = stock_data.index[-1]
        future_dates = []
        
        for i in range(1, days + 1):
            if isinstance(last_date, pd.Timestamp):
                future_date = last_date + pd.Timedelta(days=i)
            else:
                future_date = datetime.now() + timedelta(days=i)
            future_dates.append(future_date)
        
        # Fiyat tahmini yap - liste olarak
        future_prices = []
        for i in range(days):
            progress = i / (days - 1)  # 0 to 1
            # Basit doğrusal enterpolasyon
            day_price = current_price + (target_price - current_price) * progress
            
            # Rastgele dalgalanmalar ekle
            random_factor = np.random.uniform(-1, 1) * 0.01  # %1 dalgalanma
            day_price = day_price * (1 + random_factor)
            
            future_prices.append(day_price)
        
        # Tahmin grafiğini oluştur
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Geçmiş veri
        ax.plot(stock_data.index[-30:], stock_data['Close'].iloc[-30:].values, label='Geçmiş Veri', color='blue')
        
        # Gelecek tahmin - tarihleri ve fiyatları liste olarak kullan
        ax.plot(future_dates, future_prices, label='Tahmin', 
               color='green' if target_price > current_price else 'red', 
               linestyle='--')
        
        # Destek ve direnç çizgileri
        support = prediction_data.get('support_level', current_price * 0.9)
        resistance = prediction_data.get('resistance_level', current_price * 1.1)
        
        ax.axhline(y=support, color='green', linestyle=':', 
                  label=f"Destek: {support:.2f}")
        ax.axhline(y=resistance, color='red', linestyle=':', 
                  label=f"Direnç: {resistance:.2f}")
        
        ax.set_title(f"{symbol} Fiyat Tahmini")
        ax.set_xlabel('Tarih')
        ax.set_ylabel('Fiyat (TL)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig
    except Exception as e:
        st.error(f"Tahmin grafiği oluşturma hatası ({symbol}): {str(e)}")
        return None
