import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import sqlite3
import sys
import os
from typing import Dict, List, Tuple

# Ana proje dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db_utils import get_ml_predictions, get_ml_prediction_stats, DB_FILE
from data.stock_data import get_stock_data

def render_ml_backtest_tab():
    """ML Backtest sekmesi"""
    
    st.title("🎯 ML Model Backtest Analizi")
    st.markdown("""
    Bu sekme ML modelinizin geçmiş performansını analiz eder ve hangi hisse türlerinde daha başarılı olduğunu gösterir.
    """)
    
    # Mevcut veri durumunu göster
    try:
        import sqlite3
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM ml_predictions')
        total_predictions = cursor.fetchone()[0]
        
        cursor.execute('SELECT DISTINCT model_type FROM ml_predictions')
        available_models = [row[0] for row in cursor.fetchall()]
        
        cursor.execute('SELECT MIN(prediction_date), MAX(prediction_date) FROM ml_predictions')
        date_range = cursor.fetchone()
        
        conn.close()
        
        # Bilgi kartı
        st.info(f"""
        📊 **Mevcut Veri Durumu:**
        - Toplam ML tahmin: **{total_predictions:,}** adet
        - Mevcut modeller: **{', '.join(available_models)}**
        - Tarih aralığı: **{date_range[0]}** - **{date_range[1]}**
        """)
        
    except Exception as e:
        st.warning("Veritabanı bilgileri alınamadı. ML tarama yapıldığından emin olun.")
    
    # Backtest parametreleri
    col1, col2, col3 = st.columns(3)
    
    with col1:
        backtest_period = st.selectbox(
            "Backtest Periyodu",
            ["Son 30 Gün", "Son 60 Gün", "Son 90 Gün", "Son 6 Ay", "Son 1 Yıl"],
            index=2
        )
    
    with col2:
        model_filter = st.multiselect(
            "Model Türleri",
            ["RandomForest", "XGBoost", "LightGBM", "Ensemble", "Hibrit Model"],
            default=["RandomForest", "Ensemble", "Hibrit Model"],  # Mevcut modellere göre güncellendi
            help="Veritabanında bulunan modeller seçildi"
        )
    
    with col3:
        min_confidence = st.slider(
            "Minimum Güven Oranı (%)",
            min_value=0,
            max_value=100,
            value=20,  # Daha düşük varsayılan değer
            step=5,
            help="Düşük değer daha fazla tahmin dahil eder"
        )
    
    if st.button("🚀 Backtest Çalıştır", type="primary"):
        run_backtest(backtest_period, model_filter, min_confidence)
    
    # Ana içerik
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Genel Performans", "📈 Hisse Bazlı Analiz", "🔍 Detaylı Sonuçlar", "💡 Öneriler"])
    
    with tab1:
        render_general_performance()
    
    with tab2:
        render_stock_analysis()
    
    with tab3:
        render_detailed_results()
    
    with tab4:
        render_recommendations()

def run_backtest(period: str, models: List[str], min_confidence: int):
    """Backtest çalıştır"""
    
    # Periyodu gün sayısına çevir
    period_days = {
        "Son 30 Gün": 30,
        "Son 60 Gün": 60,
        "Son 90 Gün": 90,
        "Son 6 Ay": 180,
        "Son 1 Yıl": 365
    }
    
    days = period_days[period]
    start_date = datetime.now() - timedelta(days=days)
    
    with st.spinner("Backtest çalıştırılıyor..."):
        # Debug bilgileri
        st.info(f"🔍 Arama kriterleri: {start_date.strftime('%Y-%m-%d')} tarihinden sonra, Modeller: {models}, Min. güven: %{min_confidence}")
        
        # Veritabanından tahminleri al
        predictions = get_ml_predictions_for_backtest(start_date, models, min_confidence)
        
        if not predictions:
            st.warning(f"❌ Seçilen kriterlere uygun tahmin bulunamadı.")
            st.info("💡 Çözüm önerileri:")
            st.markdown("- Minimum güven oranını düşürün (%20 deneyin)")
            st.markdown("- Daha geniş bir zaman aralığı seçin (Son 6 Ay)")
            st.markdown("- Farklı modeller seçin (Ensemble, Hibrit Model)")
            return
        else:
            st.success(f"✅ {len(predictions)} adet tahmin bulundu, analiz ediliyor...")
        
        # Her tahmin için gerçek sonuçları kontrol et
        backtest_results = []
        
        for pred in predictions:
            symbol = pred['symbol']
            prediction_date = pd.to_datetime(pred['prediction_date'])
            target_date = pd.to_datetime(pred['target_date']) if pred['target_date'] else prediction_date + timedelta(days=1)
            predicted_change = pred['prediction_percentage']
            confidence = pred['confidence_score']
            
            # Gerçek fiyat verilerini al
            try:
                stock_data = get_stock_data(symbol, period="1y")
                if stock_data is not None and len(stock_data) > 0:
                    
                    # Tahmin tarihindeki fiyatı bul
                    pred_price = get_price_on_date(stock_data, prediction_date)
                    
                    # Hedef tarihteki fiyatı bul  
                    target_price = get_price_on_date(stock_data, target_date)
                    
                    if pred_price and target_price:
                        actual_change = (target_price - pred_price) / pred_price
                        
                        # Sonuçları kaydet
                        backtest_results.append({
                            'symbol': symbol,
                            'prediction_date': prediction_date.strftime('%Y-%m-%d'),
                            'target_date': target_date.strftime('%Y-%m-%d'),
                            'predicted_change': predicted_change,
                            'actual_change': actual_change,
                            'confidence': confidence,
                            'model_type': pred['model_type'],
                            'success': (predicted_change > 0 and actual_change > 0) or (predicted_change < 0 and actual_change < 0),
                            'abs_error': abs(predicted_change - actual_change),
                            'prediction_id': pred['id']
                        })
            except Exception as e:
                continue
        
        # Sonuçları session state'e kaydet
        st.session_state.backtest_results = backtest_results
        
        if backtest_results:
            st.success(f"Backtest tamamlandı! {len(backtest_results)} tahmin analiz edildi.")
        else:
            st.warning("Backtest sonucu bulunamadı.")

def get_ml_predictions_for_backtest(start_date: datetime, models: List[str], min_confidence: int) -> List[Dict]:
    """Backtest için ML tahminlerini al"""
    
    try:
        if not os.path.exists(DB_FILE):
            st.warning("ML tahminleri veritabanı bulunamadı. Önce ML tarama yapın.")
            return []
            
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Tablo var mı kontrol et
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ml_predictions'")
        if cursor.fetchone() is None:
            st.warning("ML tahminleri tablosu bulunamadı. Önce ML tarama yapın.")
            conn.close()
            return []
        
        # Model filtresi için SQL IN clause oluştur
        if not models:
            models = ["RandomForest"]  # Varsayılan
            
        model_placeholders = ','.join(['?' for _ in models])
        
        query = f"""
        SELECT id, symbol, current_price, prediction_percentage, confidence_score, 
               prediction_result, model_type, features_used, target_date, prediction_date
        FROM ml_predictions
        WHERE prediction_date >= ? 
        AND model_type IN ({model_placeholders})
        AND confidence_score >= ?
        ORDER BY prediction_date DESC
        """
        
        params = [start_date.strftime('%Y-%m-%d %H:%M:%S')] + models + [min_confidence / 100]
        
        cursor.execute(query, params)
        
        columns = [description[0] for description in cursor.description]
        results = cursor.fetchall()
        
        predictions = []
        for row in results:
            prediction = dict(zip(columns, row))
            predictions.append(prediction)
        
        conn.close()
        return predictions
        
    except Exception as e:
        st.error(f"Veritabanı hatası: {str(e)}")
        return []

def get_price_on_date(stock_data: pd.DataFrame, target_date: pd.Timestamp) -> float:
    """Belirli bir tarihteki fiyatı bul"""
    
    try:
        # Tarihi normalize et
        target_date = target_date.normalize()
        
        # Exact match dene
        exact_match = stock_data[stock_data.index.normalize() == target_date]
        if len(exact_match) > 0:
            return float(exact_match['Close'].iloc[0])
        
        # En yakın tarihi bul
        stock_data_normalized = stock_data.copy()
        stock_data_normalized.index = stock_data_normalized.index.normalize()
        
        # Hedef tarihten sonraki ilk tarihi bul
        future_dates = stock_data_normalized[stock_data_normalized.index >= target_date]
        if len(future_dates) > 0:
            return float(future_dates['Close'].iloc[0])
        
        # Hedef tarihten önceki son tarihi bul
        past_dates = stock_data_normalized[stock_data_normalized.index <= target_date]
        if len(past_dates) > 0:
            return float(past_dates['Close'].iloc[-1])
        
        return None
        
    except Exception as e:
        return None

def render_general_performance():
    """Genel performans sekmesi"""
    
    if 'backtest_results' not in st.session_state or not st.session_state.backtest_results:
        st.info("Backtest çalıştırmak için yukarıdaki parametreleri ayarlayın ve 'Backtest Çalıştır' butonuna tıklayın.")
        return
    
    results = st.session_state.backtest_results
    df = pd.DataFrame(results)
    
    # Genel istatistikler
    st.subheader("📊 Genel Performans Metrikleri")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_predictions = len(df)
        st.metric("Toplam Tahmin", total_predictions)
    
    with col2:
        successful_predictions = df['success'].sum()
        success_rate = (successful_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        st.metric("Başarı Oranı", f"{success_rate:.1f}%")
    
    with col3:
        avg_confidence = df['confidence'].mean() * 100
        st.metric("Ortalama Güven", f"{avg_confidence:.1f}%")
    
    with col4:
        avg_error = df['abs_error'].mean() * 100
        st.metric("Ortalama Hata", f"{avg_error:.1f}%")
    
    # Model bazlı performans
    st.subheader("🔧 Model Bazlı Performans")
    
    model_performance = df.groupby('model_type').agg({
        'success': ['count', 'sum', 'mean'],
        'abs_error': 'mean',
        'confidence': 'mean'
    }).round(3)
    
    model_performance.columns = ['Toplam Tahmin', 'Başarılı Tahmin', 'Başarı Oranı', 'Ortalama Hata', 'Ortalama Güven']
    model_performance['Başarı Oranı'] = model_performance['Başarı Oranı'] * 100
    model_performance['Ortalama Hata'] = model_performance['Ortalama Hata'] * 100
    model_performance['Ortalama Güven'] = model_performance['Ortalama Güven'] * 100
    
    st.dataframe(model_performance, use_container_width=True)
    
    # Performans grafiği
    fig = px.bar(
        model_performance.reset_index(),
        x='model_type',
        y='Başarı Oranı',
        title="Model Başarı Oranları",
        color='Başarı Oranı',
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig, use_container_width=True)

def render_stock_analysis():
    """Hisse bazlı analiz sekmesi"""
    
    if 'backtest_results' not in st.session_state or not st.session_state.backtest_results:
        st.info("Önce backtest çalıştırın.")
        return
    
    results = st.session_state.backtest_results
    df = pd.DataFrame(results)
    
    st.subheader("📈 Hisse Bazlı Performans Analizi")
    
    # Hisse bazlı performans
    stock_performance = df.groupby('symbol').agg({
        'success': ['count', 'sum', 'mean'],
        'abs_error': 'mean',
        'confidence': 'mean',
        'predicted_change': 'mean',
        'actual_change': 'mean'
    }).round(3)
    
    stock_performance.columns = ['Toplam Tahmin', 'Başarılı Tahmin', 'Başarı Oranı', 'Ortalama Hata', 'Ortalama Güven', 'Ortalama Tahmin', 'Ortalama Gerçek']
    stock_performance['Başarı Oranı'] = stock_performance['Başarı Oranı'] * 100
    stock_performance['Ortalama Hata'] = stock_performance['Ortalama Hata'] * 100
    stock_performance['Ortalama Güven'] = stock_performance['Ortalama Güven'] * 100
    stock_performance['Ortalama Tahmin'] = stock_performance['Ortalama Tahmin'] * 100
    stock_performance['Ortalama Gerçek'] = stock_performance['Ortalama Gerçek'] * 100
    
    # Sadece 2+ tahmin yapılan hisseleri göster
    stock_performance_filtered = stock_performance[stock_performance['Toplam Tahmin'] >= 2]
    
    # Başarı oranına göre sırala
    stock_performance_filtered = stock_performance_filtered.sort_values('Başarı Oranı', ascending=False)
    
    st.dataframe(stock_performance_filtered, use_container_width=True)
    
    # Scatter plot: Tahmin vs Gerçek
    st.subheader("🎯 Tahmin vs Gerçek Değişim")
    
    fig = px.scatter(
        df,
        x='predicted_change',
        y='actual_change',
        color='success',
        size='confidence',
        hover_data=['symbol', 'model_type'],
        title="Tahmin Edilen vs Gerçekleşen Değişim",
        labels={
            'predicted_change': 'Tahmin Edilen Değişim',
            'actual_change': 'Gerçekleşen Değişim'
        }
    )
    
    # Ideal çizgi ekle
    fig.add_shape(
        type="line",
        x0=df['predicted_change'].min(),
        y0=df['predicted_change'].min(),
        x1=df['predicted_change'].max(),
        y1=df['predicted_change'].max(),
        line=dict(color="red", width=2, dash="dash"),
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_detailed_results():
    """Detaylı sonuçlar sekmesi"""
    
    if 'backtest_results' not in st.session_state or not st.session_state.backtest_results:
        st.info("Önce backtest çalıştırın.")
        return
    
    results = st.session_state.backtest_results
    df = pd.DataFrame(results)
    
    st.subheader("🔍 Detaylı Backtest Sonuçları")
    
    # Filtreleme seçenekleri
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbol_filter = st.multiselect(
            "Hisse Filtresi",
            options=df['symbol'].unique(),
            default=[]
        )
    
    with col2:
        success_filter = st.selectbox(
            "Başarı Durumu",
            options=["Tümü", "Başarılı", "Başarısız"],
            index=0
        )
    
    with col3:
        model_filter = st.multiselect(
            "Model Filtresi", 
            options=df['model_type'].unique(),
            default=df['model_type'].unique().tolist()
        )
    
    # Filtreleri uygula
    filtered_df = df.copy()
    
    if symbol_filter:
        filtered_df = filtered_df[filtered_df['symbol'].isin(symbol_filter)]
    
    if success_filter == "Başarılı":
        filtered_df = filtered_df[filtered_df['success'] == True]
    elif success_filter == "Başarısız":
        filtered_df = filtered_df[filtered_df['success'] == False]
    
    if model_filter:
        filtered_df = filtered_df[filtered_df['model_type'].isin(model_filter)]
    
    # Sonuçları tablo olarak göster
    display_df = filtered_df.copy()
    display_df['predicted_change'] = (display_df['predicted_change'] * 100).round(2)
    display_df['actual_change'] = (display_df['actual_change'] * 100).round(2)
    display_df['confidence'] = (display_df['confidence'] * 100).round(1)
    display_df['abs_error'] = (display_df['abs_error'] * 100).round(2)
    
    # Sütun adlarını Türkçe yap
    display_df = display_df.rename(columns={
        'symbol': 'Hisse',
        'prediction_date': 'Tahmin Tarihi',
        'target_date': 'Hedef Tarih',
        'predicted_change': 'Tahmin (%)',
        'actual_change': 'Gerçek (%)',
        'confidence': 'Güven (%)',
        'model_type': 'Model',
        'success': 'Başarılı',
        'abs_error': 'Hata (%)'
    })
    
    # Başarı durumunu emoji ile göster
    display_df['Başarılı'] = display_df['Başarılı'].map({True: '✅', False: '❌'})
    
    st.dataframe(
        display_df[['Hisse', 'Tahmin Tarihi', 'Hedef Tarih', 'Tahmin (%)', 'Gerçek (%)', 'Güven (%)', 'Model', 'Başarılı', 'Hata (%)']],
        use_container_width=True
    )

def render_recommendations():
    """Öneriler sekmesi"""
    
    if 'backtest_results' not in st.session_state or not st.session_state.backtest_results:
        st.info("Önce backtest çalıştırın.")
        return
    
    results = st.session_state.backtest_results
    df = pd.DataFrame(results)
    
    st.subheader("💡 Model İyileştirme Önerileri")
    
    # Genel başarı oranı
    overall_success = (df['success'].sum() / len(df)) * 100
    
    # Model performansı analizi
    model_performance = df.groupby('model_type')['success'].mean() * 100
    
    if len(model_performance) > 0:
        best_model = model_performance.idxmax()
        worst_model = model_performance.idxmin()
    
    # Öneriler
    recommendations = []
    
    if overall_success < 60:
        recommendations.append("⚠️ **Genel Başarı Oranı Düşük**: Model parametrelerini yeniden gözden geçirin. Feature engineering ve model tuning gerekebilir.")
    elif overall_success >= 60 and overall_success < 70:
        recommendations.append("🔄 **Orta Seviye Performans**: Modeli iyileştirmek için ensemble yöntemler deneyin.")
    else:
        recommendations.append("✅ **İyi Performans**: Mevcut model iyi çalışıyor, ancak sürekli izleme önemli.")
    
    if len(model_performance) > 1 and model_performance.max() - model_performance.min() > 15:
        recommendations.append(f"🎯 **Model Seçimi**: {best_model} modeli en iyi performansı gösteriyor (%{model_performance.max():.1f}). {worst_model} modelini gözden geçirin.")
    
    # Hata analizi  
    avg_error = df['abs_error'].mean() * 100
    if avg_error > 10:
        recommendations.append(f"📊 **Yüksek Hata Oranı**: Ortalama hata %{avg_error:.1f}. Model kalibrasyonu gerekebilir.")
    
    # Önerileri göster
    for i, recommendation in enumerate(recommendations, 1):
        st.markdown(f"{i}. {recommendation}")
    
    # Önerilen aksiyonlar
    st.subheader("🎯 Önerilen Aksiyonlar")
    
    action_items = [
        "1. **Veri Kalitesi**: Giriş verilerinin kalitesini kontrol edin",
        "2. **Feature Engineering**: Yeni özellikler eklemeyi deneyin", 
        "3. **Hyperparameter Tuning**: Model parametrelerini optimize edin",
        "4. **Cross-Validation**: Modeli farklı zaman dilimlerinde test edin",
        "5. **Ensemble Methods**: Birden fazla modeli birleştirin",
        "6. **Düzenli İzleme**: Modeli sürekli olarak izleyin ve güncelleyin"
    ]
    
    for action in action_items:
        st.markdown(action)

if __name__ == "__main__":
    render_ml_backtest_tab() 