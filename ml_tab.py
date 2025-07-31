import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yfinance as yf

from data.stock_data import get_stock_data, get_company_info, get_stock_news
from analysis.indicators import calculate_indicators
from ai.predictions import ml_price_prediction, backtest_models
from data.db_utils import save_analysis_result, get_model_versions, rollback_model_version
from data.utils import load_analysis_results
from config import ML_MODEL_PARAMS, PREDICTION_PERIODS, DEFAULT_PREDICTION_PERIOD

def render_ml_prediction_tab():
    """
    ML Tahmini sekmesini oluşturur
    """
    st.header("Makine Öğrenmesi ile Hisse Tahminleri")
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    # Session state'den selected_stock_for_analysis'i kontrol et - sadece bu sekme için
    initial_stock = ""
    if 'selected_stock_for_analysis' in st.session_state and st.session_state.selected_stock_for_analysis:
        initial_stock = st.session_state.selected_stock_for_analysis
        # Değişkeni kullandıktan sonra temizle
        st.session_state.selected_stock_for_analysis = ""
    
    with col1:
        stock_symbol = st.text_input("Hisse Senedi Kodu", value=initial_stock, key="ml_stock_input")
    
    with col2:
        prediction_type = st.selectbox(
            "Tahmin Türü",
            ["Fiyat Tahmini", "Yön Tahmini", "Volatilite Tahmini"],
            help="Ne tür tahmin yapmak istediğinizi seçin"
        )
    
    with col3:
        prediction_days = st.selectbox(
            "Tahmin Süresi (Gün)",
            [1, 3, 5, 7, 14, 30],
            index=2,
            help="Kaç gün sonrası için tahmin yapılacak"
        )
    
    with col4:
        # Boş satır ekleyerek hizalamayı düzeltiyoruz
        st.write("")
        predict_button = st.button("Tahmin Et", use_container_width=True, key="ml_predict_button")
    
    # Sadece buton tıklandığında veya initial stock varsa tahmin yap
    make_prediction = predict_button or (initial_stock != "" and stock_symbol != "")
    
    # Bilgi kutusu - iyileştirmeler hakkında
    with st.expander("🆕 Yeni İyileştirmeler", expanded=False):
        st.info("""
        **Bu versiyonda eklenen iyileştirmeler:**
        
        ✅ **Gelişmiş Feature Engineering:**
        - Gecikmeli fiyatlar (lag features)
        - Volatilite göstergeleri
        - Momentum göstergeleri
        - Volume-Price Trend (VPT)
        - True Range ve ATR
        
        ✅ **Hiperparametre Optimizasyonu:**
        - RandomizedSearchCV ile otomatik optimizasyon
        - Model performansına göre parametre seçimi
        
        ✅ **Walk-Forward Validation:**
        - Zaman serisi için uygun validasyon
        - Gerçekçi performans ölçümü
        
        ✅ **Gelişmiş Güven Skoru:**
        - Çok faktörlü güven hesaplama
        - R², yön doğruluğu, tutarlılık faktörleri
        
        ✅ **Daha İyi Veri İşleme:**
        - Yumuşak outlier temizleme
        - Akıllı feature seçimi
        
        ✅ **Ensemble Modeli:**
        - 4 farklı algoritmanın kombinasyonu
        - RandomForest + GradientBoosting + XGBoost + LightGBM
        - Voting Regressor ile güçlü tahminler
        - En yüksek güvenilirlik skoru (%92)
        """)
    
    # Gelişmiş parametreler
    with st.expander("⚙️ Gelişmiş Parametreler", expanded=False):
        col_p1, col_p2, col_p3 = st.columns(3)
        
        with col_p1:
            use_walk_forward = st.checkbox("Walk-Forward Validation Kullan", value=True, 
                                         help="Zaman serisi için daha gerçekçi validasyon")
            add_volatility = st.checkbox("Volatilite Düzeltmesi Ekle", value=False,
                                       help="Tahminlere rastgele volatilite ekler")
            
        with col_p2:
            enable_optimization = st.checkbox("Hiperparametre Optimizasyonu", value=True,
                                            help="Model parametrelerini otomatik optimize eder")
            cv_folds = st.selectbox("Cross-Validation Katları", [3, 5, 7], index=0,
                                  help="Daha fazla katman = daha güvenilir ama yavaş")
        
        with col_p3:
            confidence_threshold = st.slider("Minimum Güven Eşiği (%)", 30, 80, 50,
                                           help="Bu eşiğin altındaki modeller gösterilmez")
            max_features_ratio = st.slider("Max Feature Oranı", 0.5, 1.0, 0.8, 0.1,
                                         help="Kullanılacak maksimum feature oranı")
    
    # Geçmiş tahminleri yükle
    previous_results = load_analysis_results(analysis_type="ml")
    
    # Analiz yap veya otomatik analiz
    if make_prediction:
        if not stock_symbol:
            st.error("⚠️ Lütfen bir hisse senedi kodu girin!")
            st.stop()
        
        with st.spinner(f"📊 {stock_symbol} için gelişmiş ML tahminleri hazırlanıyor..."):
            
            try:
                # Veri çekme
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("📥 Hisse senedi verileri çekiliyor...")
                    progress_bar.progress(10)
                    
                # Veri çekme
                df = get_stock_data(stock_symbol, period="2y")  # 2 yıllık veri
                
                if df is None or df.empty:
                    st.error(f"❌ {stock_symbol} için veri bulunamadı!")
                    st.stop()
                
                progress_bar.progress(25)
                status_text.text("🔧 Teknik göstergeler hesaplanıyor...")
                
                # Teknik göstergeleri hesapla
                df_with_indicators = calculate_indicators(df)
                
                progress_bar.progress(40)
                status_text.text("⚙️ ML modelleri hazırlanıyor...")
                
                # Şirket bilgilerini al
                company_info = get_company_info(stock_symbol)
                current_price = df_with_indicators['Close'].iloc[-1]
                
                # Model türleri - gelişmiş sıralama
                model_types = ["RandomForest", "XGBoost", "LightGBM", "Ensemble"]
                
                # Prediction parametreleri
                prediction_params = {
                    'use_walk_forward_validation': use_walk_forward,
                    'add_volatility': add_volatility,
                    'enable_optimization': enable_optimization,
                    'cv_folds': cv_folds,
                    'max_features_ratio': max_features_ratio
                }
                
                # Model tahminlerini yap
                all_predictions = []
                successful_predictions = 0
                failed_models = []
                model_quality_scores = {}
                
                # Progress tracking
                total_models = len(model_types)
                
                for i, model_type in enumerate(model_types):
                    progress = 40 + (i * 40 // total_models)
                    progress_bar.progress(progress)
                    status_text.text(f"🤖 {model_type} modeli eğitiliyor ve test ediliyor... ({i+1}/{total_models})")
                    
                    try:
                        # Gelişmiş ML tahmin
                        prediction = ml_price_prediction(
                            stock_symbol, 
                            df_with_indicators, 
                            days_to_predict=prediction_days, 
                            threshold=0.03,
                            model_type=model_type,
                            model_params=None,
                            prediction_params=prediction_params
                        )
                        
                        if prediction and prediction.get('confidence', 0) >= confidence_threshold:
                            # Kalite skorunu hesapla
                            r2_score = prediction.get('r2_score', 0)
                            confidence = prediction.get('confidence', 0)
                            features_count = prediction.get('features_count', 0)
                            
                            # Walk-forward skorları varsa dahil et
                            wf_scores = prediction.get('walk_forward_scores', {})
                            wf_r2 = wf_scores.get('r2', 0) if wf_scores else 0
                            
                            # Kalite skoru hesaplama
                            quality_score = (
                                max(0, r2_score) * 0.4 +  # R² skoru
                                (confidence / 100) * 0.3 +  # Güven skoru  
                                (features_count / 50) * 0.1 +  # Feature zenginliği
                                max(0, wf_r2) * 0.2  # Walk-forward performansı
                            )
                            
                            model_quality_scores[model_type] = quality_score
                            all_predictions.append((model_type, prediction))
                            successful_predictions += 1
                            
                        else:
                            failed_models.append({
                                'model': model_type,
                                'reason': f"Düşük güven skoru: %{prediction.get('confidence', 0):.1f}" if prediction else "Tahmin başarısız"
                            })
                            
                    except Exception as e:
                        failed_models.append({
                            'model': model_type,
                            'reason': f"Hata: {str(e)}"
                        })
                
                progress_bar.progress(100)
                status_text.text("✅ Analiz tamamlandı!")
                
                # Progress container'ı temizle
                progress_container.empty()
                
                # Sonuçları göster
                if successful_predictions == 0:
                    st.error("❌ Hiçbir model başarılı tahmin yapamadı!")
                    
                    if failed_models:
                        st.write("**Başarısız Modeller:**")
                        for failure in failed_models:
                            st.write(f"- **{failure['model']}**: {failure['reason']}")
                    
                    st.stop()
                
                # Başarılı sonuçları göster
                st.success(f"✅ {successful_predictions}/{len(model_types)} model başarılı!")
                
                # Model kalitesine göre sırala
                sorted_predictions = sorted(all_predictions, 
                                          key=lambda x: model_quality_scores.get(x[0], 0), 
                                          reverse=True)
                
                # Şirket bilgileri
                col_info1, col_info2 = st.columns(2)
                
                with col_info1:
                    if company_info:
                        st.info(f"""
                        **📊 {company_info.get('shortName', stock_symbol)}**
                        - **Sektör:** {company_info.get('sector', 'Bilinmiyor')}
                        - **Güncel Fiyat:** {current_price:.2f} TL
                        - **Piyasa Değeri:** {company_info.get('marketCap', 'N/A')}
                        """)
                    else:
                        st.info(f"""
                        **📊 {stock_symbol}**
                        - **Güncel Fiyat:** {current_price:.2f} TL
                        """)
                
                with col_info2:
                    st.metric(
                        label="📈 Veri Kalitesi",
                        value=f"{sorted_predictions[0][1].get('data_quality_score', 0.5):.3f}",
                        help="Veri kalitesi skoru (0-1 arası, yüksek daha iyi)"
                    )
                    
                    st.metric(
                        label="🎯 En İyi Model",
                        value=f"{sorted_predictions[0][0]}",
                        delta=f"Kalite: {model_quality_scores.get(sorted_predictions[0][0], 0):.3f}",
                        help="En yüksek kalite skoruna sahip model"
                    )
                
                # Model karşılaştırma tablosu
                st.subheader("📋 Model Karşılaştırması")
                
                comparison_data = []
                for model_type, prediction in sorted_predictions:
                    
                    # Güncel fiyatı prediction içine ekle
                    prediction['current_price'] = current_price
                    
                    # Tahmin verilerini oluştur
                    if prediction_days >= 30:
                        predicted_price = prediction.get('prediction_30d')
                    elif prediction_days >= 7:
                        predicted_price = prediction.get('prediction_7d')
                    else:
                        predicted_price = prediction.get('prediction_7d', prediction.get('prediction_30d'))
                    
                    if predicted_price is not None:
                        change_pct = ((predicted_price - current_price) / current_price) * 100
                        
                        # Model kalitesi göstergesi
                        quality_score = model_quality_scores.get(model_type, 0)
                        if quality_score > 0.7:
                            quality_badge = "🟢 Yüksek"
                        elif quality_score > 0.5:
                            quality_badge = "🟡 Orta"
                        else:
                            quality_badge = "🔴 Düşük"
                        
                        # Walk-forward bilgisi
                        wf_info = ""
                        wf_scores = prediction.get('walk_forward_scores')
                        if wf_scores:
                            wf_r2 = wf_scores.get('r2', 0)
                            wf_direction = wf_scores.get('direction_accuracy', 0)
                            wf_info = f"WF-R²: {wf_r2:.3f}, WF-Yön: {wf_direction:.3f}"
                        
                        comparison_data.append({
                            'Model': model_type,
                            'Kalite': quality_badge,
                            'Mevcut Fiyat': f"{current_price:.2f} TL",
                            f'{prediction_days} Gün Tahmini': f"{predicted_price:.2f} TL",
                            'Değişim (%)': f"{change_pct:+.2f}%",
                            'R² Skoru': f"{prediction.get('r2_score', 0):.4f}",
                            'Güven': f"%{prediction.get('confidence', 0):.1f}",
                            'Özellik Sayısı': f"{prediction.get('features_count', 0)}",
                            'Walk-Forward': wf_info,
                            'Trend': prediction.get('trend', 'Belirsiz'),
                            'change_value': change_pct,
                            'r2_value': prediction.get('r2_score', 0),
                            'confidence_value': prediction.get('confidence', 0),
                            'quality_value': quality_score
                        })
                
                # Karşılaştırma tablosunu göster
                comparison_df = pd.DataFrame(comparison_data)
                
                # Gösterim için gereksiz kolonları kaldır
                display_df = comparison_df.drop(['change_value', 'r2_value', 'confidence_value', 'quality_value'], axis=1)
                
                # Tabloyu formatla ve göster
                st.dataframe(
                    display_df,
                    column_config={
                        "Model": st.column_config.TextColumn("Model Türü", width="small"),
                        "Kalite": st.column_config.TextColumn("Model Kalitesi", width="small"),
                        f'{prediction_days} Gün Tahmini': st.column_config.NumberColumn("Tahmin", format="%.2f TL"),
                        'Değişim (%)': st.column_config.TextColumn("Değişim", width="small"),
                        'Walk-Forward': st.column_config.TextColumn("Walk-Forward Skorları", width="medium"),
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # Detaylı model sonuçları
                st.subheader("📊 Detaylı Model Sonuçları")
                
                # En iyi modeli vurgula
                best_model_type = sorted_predictions[0][0]
                best_prediction = sorted_predictions[0][1]
                
                # En iyi model detayları
                col_best1, col_best2, col_best3, col_best4 = st.columns(4)
                
                with col_best1:
                    st.metric(
                        label="🏆 En İyi Model",
                        value=best_model_type,
                        delta=f"R²: {best_prediction.get('r2_score', 0):.4f}"
                    )
                
                with col_best2:
                    if prediction_days >= 30:
                        pred_price = best_prediction.get('prediction_30d', current_price)
                    elif prediction_days >= 7:
                        pred_price = best_prediction.get('prediction_7d', current_price)
                    else:
                        pred_price = best_prediction.get('prediction_7d', current_price)
                    
                    change_pct = ((pred_price - current_price) / current_price) * 100
                    st.metric(
                        label=f"📈 {prediction_days} Gün Tahmini",
                        value=f"{pred_price:.2f} TL",
                        delta=f"{change_pct:+.2f}%"
                    )
                
                with col_best3:
                    st.metric(
                        label="🎯 Güven Skoru",
                        value=f"%{best_prediction.get('confidence', 0):.1f}",
                        delta=f"Kalite: {model_quality_scores.get(best_model_type, 0):.3f}"
                    )
                
                with col_best4:
                    st.metric(
                        label="🔧 Özellik Sayısı", 
                        value=f"{best_prediction.get('features_count', 0)}",
                        delta=f"RMSE: {best_prediction.get('rmse', 0):.2f}"
                    )
                
                # Model performans grafikleri
                st.subheader("📈 Tahmin Grafikleri")
                
                # Tarihsel veriler için grafik
                fig = go.Figure()
                
                # Tarihsel fiyat
                last_60_days = df_with_indicators.tail(60)
                fig.add_trace(go.Scatter(
                    x=last_60_days.index,
                    y=last_60_days['Close'],
                    mode='lines',
                    name='Tarihsel Fiyat',
                    line=dict(color='blue', width=2)
                ))
                
                # Model tahminleri
                colors = ['red', 'green', 'orange', 'purple', 'brown']
                line_styles = ['solid', 'dash', 'dot', 'dashdot', 'longdash']
                
                for i, (model_type, prediction) in enumerate(sorted_predictions):
                    try:
                        predictions_df = prediction.get('predictions_df')
                        if predictions_df is not None and not predictions_df.empty:
                            # Kalite skoruna göre çizgi kalınlığı
                            quality = model_quality_scores.get(model_type, 0)
                            line_width = max(2, int(quality * 6))  # 2-6 arası kalınlık
                            
                            # Kalite skoruna göre opaklık - 0-1 aralığında sınırla
                            opacity = max(0.6, min(1.0, quality))  # Quality değerini 1.0'a sınırla
                            
                            line_dash = line_styles[i % len(line_styles)]
                            color = colors[i % len(colors)]
                            
                            # Model adına kalite badge'i ekle
                            quality_badge = "🟢" if quality > 0.7 else "🟡" if quality > 0.5 else "🔴"
                            model_name = f'{model_type} {quality_badge} (Kalite: {quality:.2f})'
                        
                        # Tahmin çizgisi
                        fig.add_trace(
                            go.Scatter(
                                x=predictions_df.index,
                                y=predictions_df['Predicted Price'],
                                mode='lines',
                                line=dict(
                                    color=color, 
                                    width=line_width, 
                                    dash=line_dash
                                ),
                                name=model_name,
                                opacity=opacity
                            )
                        )
                    except Exception as e:
                        st.warning(f"{model_type} modeli grafiği çizilemedi: {str(e)}")
                        continue
                
                # Bugünkü fiyat işaretleme
                try:
                    # Pandas Timestamp'i Plotly uyumlu formata çevir
                    current_date = pd.to_datetime(df_with_indicators.index[-1])
                    
                    fig.add_vline(
                        x=current_date, 
                        line_dash="dash", 
                        line_color="gray",
                        annotation_text="Bugün"
                    )
                except Exception as vline_error:
                    # add_vline başarısız olursa add_shape kullan
                    try:
                        current_date = df_with_indicators.index[-1]
                        fig.add_shape(
                            type="line",
                            x0=current_date, x1=current_date,
                            y0=0, y1=1,
                            yref="paper",
                            line=dict(color="gray", dash="dash"),
                        )
                        fig.add_annotation(
                            x=current_date,
                            y=1.02,
                            yref="paper",
                            text="Bugün",
                            showarrow=False,
                            font=dict(size=12, color="gray")
                        )
                    except Exception as shape_error:
                        # Her ikisi de başarısız olursa sadece uyarı ver
                        st.warning("Bugünkü tarih çizgisi eklenemedi.")
                
                fig.update_layout(
                    title=f"{stock_symbol} - Gelişmiş ML Fiyat Tahminleri",
                    xaxis_title="Tarih",
                    yaxis_title="Fiyat (TL)",
                    hovermode='x unified',
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Walk-forward validation sonuçları (varsa)
                wf_results = [p[1].get('walk_forward_scores') for p in sorted_predictions if p[1].get('walk_forward_scores')]
                
                if wf_results:
                    st.subheader("🔄 Walk-Forward Validation Sonuçları")
                    
                    wf_df = pd.DataFrame([
                        {
                            'Model': model_type,
                            'WF R² Skoru': wf_scores.get('r2', 0),
                            'WF MAE': wf_scores.get('mae', 0),
                            'WF Yön Doğruluğu': wf_scores.get('direction_accuracy', 0),
                            'WF MSE': wf_scores.get('mse', 0)
                        }
                        for model_type, prediction in sorted_predictions
                        for wf_scores in [prediction.get('walk_forward_scores')] if wf_scores
                    ])
                    
                    if not wf_df.empty:
                        st.dataframe(
                            wf_df,
                            column_config={
                                "WF R² Skoru": st.column_config.NumberColumn("R² Skoru", format="%.4f"),
                                "WF MAE": st.column_config.NumberColumn("MAE", format="%.2f"),
                                "WF Yön Doğruluğu": st.column_config.NumberColumn("Yön Doğruluğu", format="%.3f"),
                                "WF MSE": st.column_config.NumberColumn("MSE", format="%.2f"),
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        st.info("💡 Walk-Forward Validation, modelin gerçek zamanlı performansını daha iyi ölçer.")
                
                # Başarısız modeller (varsa)
                if failed_models:
                    with st.expander("⚠️ Başarısız Modeller", expanded=False):
                        for failure in failed_models:
                            st.warning(f"**{failure['model']}**: {failure['reason']}")
                
                # Analizi kaydet
                try:
                    analysis_result = {
                        'symbol': stock_symbol,
                        'analysis_type': 'ml_enhanced',
                        'predictions': comparison_data,
                        'best_model': best_model_type,
                        'successful_count': successful_predictions,
                        'total_count': len(model_types),
                        'parameters': prediction_params
                    }
                    
                    save_analysis_result(analysis_result)
                    st.success("📝 Analiz sonuçları kaydedildi!")
                    
                except Exception as e:
                    st.warning(f"Analiz kaydedilemedi: {str(e)}")
                
            except Exception as e:
                st.error(f"❌ Analiz sırasında hata oluştu: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Geçmiş sonuçlar
    if previous_results:
        st.subheader("📚 Geçmiş ML Analizleri")
        
        ml_results = [r for r in previous_results if r.get('analysis_type') in ['ml', 'ml_enhanced']]
        
        if ml_results:
            # Son 5 analizi göster
            recent_results = sorted(ml_results, key=lambda x: x.get('timestamp', ''), reverse=True)[:5]
            
            for result in recent_results:
                with st.expander(f"📊 {result.get('symbol', 'N/A')} - {result.get('timestamp', 'N/A')[:16]}", expanded=False):
                    col_h1, col_h2 = st.columns(2)
                    
                    with col_h1:
                        st.write(f"**En İyi Model:** {result.get('best_model', 'N/A')}")
                        st.write(f"**Başarı Oranı:** {result.get('successful_count', 0)}/{result.get('total_count', 0)}")
                    
                    with col_h2:
                        if result.get('predictions'):
                            best_prediction = result['predictions'][0] if result['predictions'] else {}
                            st.write(f"**R² Skoru:** {best_prediction.get('r2_value', 'N/A')}")
                            st.write(f"**Güven:** {best_prediction.get('confidence_value', 'N/A')}")
        else:
            st.info("Henüz ML analizi yapılmamış.")

# ... existing code ... 