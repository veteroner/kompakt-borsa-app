import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time
import webbrowser
import random
import traceback
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import requests
from bs4 import BeautifulSoup
import re
import importlib.util

# Config sistemi
from config import ML_MODEL_PARAMS, API_KEYS

# Transformers kütüphanesini try-except içinde import ediyoruz
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("Transformers kütüphanesi yüklenemedi. Basit duyarlılık analizi kullanılacak.")

from data.news_data import get_stock_news, NewsSource, analyze_news_with_gemini

# Log mesajlarını görüntülemek için yardımcı fonksiyon
def display_log_message(message, log_container=None, type="info"):
    """İşlem günlüğüne mesaj ekler"""
    # Günlük yoksa konsola yaz
    import logging
    logger = logging.getLogger(__name__)
    if type == "error":
        logger.error(message)
        # UI'da sadece log_container varsa ve expanded=True ise göster
        if log_container:
            try:
                log_container.error(message)
            except:
                pass  # Expander kapalı olabilir, hata vermemesi için
    elif type == "warning":
        logger.warning(message)
        # UI'da sadece log_container varsa ve expanded=True ise göster
        if log_container:
            try:
                log_container.warning(message)
            except:
                pass  # Expander kapalı olabilir, hata vermemesi için
    else:
        logger.info(message)
        # UI'da sadece log_container varsa ve expanded=True ise göster
        if log_container:
            try:
                log_container.info(message)
            except:
                pass  # Expander kapalı olabilir, hata vermemesi için

# Sentiment analiz modeli - global tanımlama ve lazy loading
@st.cache_resource
def load_sentiment_model():
    """Duyarlılık analizi modelini yükler"""
    if not TRANSFORMERS_AVAILABLE:
        return simple_sentiment_analysis
    
    try:
        # Daha stabil bir model kullan - Türkçe dil modeli
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        
        # Türkçe BERT-base model - daha eski ve stabil
        model_name = "dbmdz/bert-base-turkish-cased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Pipeline kullanarak modeli oluştur
        nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        return nlp
    except Exception as e:
        st.warning(f"Transformers modeli yüklenemedi: {str(e)}. Basit analiz kullanılacak.")
        return simple_sentiment_analysis

# Basit duyarlılık analizi - transformers olmadan çalışır
def simple_sentiment_analysis(text):
    """Basit kelime tabanlı duyarlılık analizi"""
    if not text:
        return {"label": "POSITIVE", "score": 0.5}
    
    try:
        # NLTK ve TextBlob kurulumunu kontrol et
        has_textblob = importlib.util.find_spec("textblob") is not None
        has_nltk = importlib.util.find_spec("nltk") is not None
        
        # Eğer NLTK kurulu ise
        if has_nltk:
            try:
                import nltk
                from nltk.tokenize import word_tokenize
                
                # Gerekli NLTK verilerini kontrol et ve indir
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    try:
                        nltk.download('punkt', quiet=True)
                    except:
                        pass
            except:
                has_nltk = False
                
        # TextBlob ile duyarlılık analizi 
        if has_textblob:
            try:
                from textblob import TextBlob
                
                # Türkçe örneği tanımla (TextBlob eğitimi için)
                tr_training_data = [
                    ('Bu ürün harika!', 'pos'),
                    ('Çok kötü bir deneyim', 'neg'),
                    ('Çok memnun kaldım', 'pos'),
                    ('Hayal kırıklığına uğradım', 'neg'),
                    ('Kesinlikle öneririm', 'pos'),
                    ('Fiyatı uygun değil', 'neg'),
                    ('Çok kaliteli', 'pos'),
                    ('Kullanımı zor', 'neg'),
                    ('Hızlı teslimat', 'pos'),
                    ('Kargo hasarlı geldi', 'neg')
                ]
                
                # Metni analiz et
                analysis = TextBlob(text)
                
                # TextBlob polarity -1 ile 1 arasında değer döndürür
                # -1: çok olumsuz, 0: nötr, 1: çok olumlu
                polarity = analysis.sentiment.polarity
                
                # 0-1 aralığına dönüştür
                normalized_score = (polarity + 1) / 2
                
                label = "POSITIVE" if normalized_score >= 0.5 else "NEGATIVE"
                
                return {"label": label, "score": normalized_score if label == "POSITIVE" else 1 - normalized_score}
                
            except Exception as e:
                print(f"TextBlob hatası: {str(e)}")
                # Eğer TextBlob hatası verirse kelime listesi yöntemine dön
        
        # Geliştirilmiş kelime listesi metodu - diğer yöntemler başarısız olursa
        # Türkçe olumlu ve olumsuz kelimelerin genişletilmiş listesi
        positive_words = {
            'artış', 'yükseliş', 'kazanç', 'kâr', 'rekor', 'başarı', 'pozitif', 'olumlu', 'güçlü', 'büyüme', 
            'iyileşme', 'yükseldi', 'arttı', 'çıktı', 'güven', 'istikrar', 'avantaj', 'fırsat', 'yatırım',
            'imzalandı', 'anlaşma', 'destek', 'teşvik', 'ivme', 'fayda', 'artırdı', 'kazandı', 'genişleme',
            'ihracat', 'ciro', 'teşvik', 'ödül', 'toparlanma', 'umut', 'iyi', 'memnuniyet', 'ralli',
            'yüksek', 'çözüm', 'artacak', 'başarılı', 'kazanım', 'gelişme', 'ilerleme', 'potansiyel',
            'güçlendi', 'atılım', 'değerlendi', 'hedef', 'inovasyon', 'öncü', 'lider', 'performans',
            'verimli', 'karlı', 'stratejik', 'sürdürülebilir', 'yenilikçi', 'büyük',
            # Finansal özel terimler
            'temettü', 'bedelsiz', 'pay', 'program', 'geri alım', 'hisse geri alım', 'pay geri alım',
            'bedelli', 'sermaye artırım', 'prim', 'bono', 'ihraç', 'ayrılacak', 'alacak', 'dağıtacak',
            'anlaşma', 'sözleşme', 'patent', 'lisans', 'teknoloji', 'ortaklık',
            # Ek olumlu terimler
            'güçlenerek', 'kazançlı', 'başarıyla', 'cazip', 'avantajlı', 'ideal', 'popüler',
            'geliştirdi', 'ilgi', 'talebi arttı', 'önemli', 'stratejik', 'prestijli', 'önde gelen',
            'yükselen', 'daha iyi', 'etkili', 'prim yaptı', 'değer kazandı', 'artış gösterdi',
            'kazandırdı', 'yükselişte', 'gelir', 'büyüyor', 'gelişti', 'işbirliği', 'destekledi',
            'onaylandı', 'sağlam', 'güven veriyor', 'istikrarlı', 'avantaj sağlıyor',
            'öneriliyor', 'tavsiye', 'gelişme gösterdi', 'güven artışı', 'reform', 'iyileştirme',
            'çözüm sağladı', 'potansiyel', 'fayda', 'dengeli', 'olumlu etki', 'rakamlar yukarı',
            'sevindirici', 'hızlı', 'başarılı sonuç'
        }
        
        negative_words = {
            'düşüş', 'kayıp', 'zarar', 'risk', 'gerileme', 'olumsuz', 'negatif', 'zayıf', 'belirsizlik', 
            'endişe', 'azaldı', 'düştü', 'kaybetti', 'gecikme', 'borç', 'iflas', 'kriz', 'tehdit', 'sorun',
            'başarısız', 'yaptırım', 'ceza', 'iptal', 'durgunluk', 'darbe', 'kötü', 'daralma', 'kesinti',
            'baskı', 'paniği', 'çöküş', 'alarm', 'tedirgin', 'zor', 'şok', 'dava', 'soruşturma', 'satış',
            'düşük', 'ağır', 'kötüleşme', 'panik', 'küçülme', 'yavaşlama', 'kapatma', 'haciz', 'çöktü',
            'bozulma', 'çıkmaz', 'açık', 'açıklar', 'gerileyecek', 'olumsuzluk', 'ertelendi', 'reddedildi',
            'azalacak', 'kaygı', 'uyarı', 'sıkıntı', 'pahalı', 'vergi', 'engel', 'hayal kırıklığı',
            # Ek olumsuz terimler
            'zor durum', 'kötüleşti', 'yetersiz', 'daraldı', 'durgunluk', 'sıkıntıda', 'zayıflama',
            'kötü performans', 'kredi notu düştü', 'güvensizlik', 'ciddi sorun', 'resesyon',
            'enflasyon baskısı', 'yasaklandı', 'manipülasyon', 'ağır koşullar', 'eleştiri',
            'düşüşte', 'kaybetti', 'zararda', 'olumsuz etkilendi', 'azalıyor', 'geriledi',
            'tahribat', 'şikayet', 'kriz derinleşiyor', 'tükenme', 'darbe aldı', 'çökme', 'piyasa şoku',
            'ihtiyatlı olmak', 'risk artışı', 'karışıklık', 'belirsizlik artıyor', 'endişe verici',
            'başarısız oldu', 'hata', 'kayıp yaşanıyor', 'altında kaldı', 'dibe vurdu', 'düşüş eğilimi',
            'batık', 'değer kaybetti', 'talep azaldı', 'zayıf tahmin', 'darbe vurdu', 'kırılgan',
            'yaptırım geldi', 'ağır çekim', 'fiyat düşüşü', 'düşüş hızlandı', 'olumsuz sinyal'
        }
        
        # NLTK kullanarak gelişmiş tokenizasyon (eğer varsa)
        if has_nltk:
            try:
                # Daha iyi tokenizasyon için NLTK kullan
                tokens = word_tokenize(text.lower(), language='turkish')
                
                # Metindeki kelimeleri kontrol et (NLTK ile)
                positive_count = sum(1 for token in tokens if token in positive_words)
                negative_count = sum(1 for token in tokens if token in negative_words)
                
                # Cümle bazlı analiz
                sentences = nltk.sent_tokenize(text.lower(), language='turkish')
                for sentence in sentences:
                    sentence_tokens = word_tokenize(sentence, language='turkish')
                    # Cümlede olumlu/olumsuz kelime var mı?
                    has_positive = any(token in positive_words for token in sentence_tokens)
                    has_negative = any(token in negative_words for token in sentence_tokens)
                    
                    # Cümlelerin ağırlığını ayarla
                    if has_positive and not has_negative:
                        positive_count += 0.5  # Olumlu cümleye bonus
                    if has_negative and not has_positive:
                        negative_count += 0.5  # Olumsuz cümleye bonus
                
                total = positive_count + negative_count
                if total == 0:
                    return {"label": "POSITIVE", "score": 0.5}
                
                score = positive_count / total if total > 0 else 0.5
                label = "POSITIVE" if score >= 0.5 else "NEGATIVE"
                
                return {"label": label, "score": score if label == "POSITIVE" else 1 - score}
                
            except Exception as e:
                print(f"NLTK hatası: {str(e)}")
                # NLTK hatası verirse basit kelime eşleştirme yöntemine dön
        
        # Basit kelime eşleştirme - diğer hepsi başarısız olursa
        # Metin içindeki kelimeleri kontrol et
        text = text.lower()
        words = text.split()
        
        positive_count = sum(1 for word in words if any(pos_word in word for pos_word in positive_words))
        negative_count = sum(1 for word in words if any(neg_word in word for neg_word in negative_words))
        
        total = positive_count + negative_count
        if total == 0:
            return {"label": "POSITIVE", "score": 0.5}
        
        score = positive_count / total if total > 0 else 0.5
        label = "POSITIVE" if score >= 0.5 else "NEGATIVE"
        
        return {"label": label, "score": score if label == "POSITIVE" else 1 - score}
    
    except Exception as e:
        # Herhangi bir hata durumunda varsayılan değer döndür
        print(f"Duyarlılık analizi hatası: {str(e)}")
        return {"label": "POSITIVE", "score": 0.5}

# Web sayfası içeriğini çekme fonksiyonu
def fetch_news_content(url, log_container=None):
    """Haber içeriğini çeker"""
    if not url or url == "#":
        display_log_message("Geçersiz URL", log_container, "warning")
        return None
        
    try:
        # Farklı User-Agent'lar
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1"
        ]
        
        # Başarı olana kadar farklı user-agent'larla deneme yap
        content = None
        last_error = None
        
        for user_agent in user_agents:
            try:
                headers = {
                    "User-Agent": user_agent,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "tr,en-US;q=0.8,en;q=0.5",
                    "Referer": "https://www.google.com/",
                    "DNT": "1",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                    "Cache-Control": "max-age=0",
                }
                
                display_log_message(f"İçerik çekiliyor: {url} (User-Agent: {user_agent[:20]}...)", log_container)
                response = requests.get(url, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    display_log_message(f"Başarılı yanıt alındı: {url}", log_container)
                    soup = BeautifulSoup(response.content, "html.parser")
                    
                    # Makale içeriğini bulmaya çalış - farklı yöntemler dene
                    content = ""
                    
                    # Yaygın makale içerik alanlarını kontrol et - Türkçe siteler için özel selektörler eklendi
                    article_selectors = [
                        "article", 
                        "div.content", 
                        "div.article-body", 
                        "div.post-content", 
                        "div.entry-content", 
                        "div.story-body",
                        # Türk haber siteleri için özel
                        "div.news-detail", 
                        "div.haberDetay", 
                        "div.haber_metni",
                        "div.DetailedNews", 
                        "div.news-body",
                        "div.text_content",
                        # Finansal siteler için
                        "div.newsContent",
                        "div.article__content",
                        "div.article-container"
                    ]
                    
                    for selector in article_selectors:
                        article = soup.select_one(selector)
                        if article:
                            # İçerik paragraflarını bul
                            paragraphs = article.find_all("p")
                            if paragraphs:
                                content = " ".join(p.get_text().strip() for p in paragraphs)
                                break
                    
                    # Hala içerik bulunamadıysa, tüm paragrafları dene
                    if not content:
                        paragraphs = soup.find_all("p")
                        content = " ".join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50)
                    
                    # İçerik yoksa
                    if not content:
                        display_log_message("İçerik bulunamadı, meta açıklaması deneniyor", log_container, "warning")
                        # Meta açıklamasını al
                        meta_desc = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
                        if meta_desc:
                            content = meta_desc.get("content", "")
                    
                    # Başlığı da almaya çalış
                    title = ""
                    title_tag = soup.find("title")
                    if title_tag:
                        title = title_tag.get_text().strip()
                    
                    # Başlık yoksa OG başlığını kontrol et
                    if not title:
                        og_title = soup.find("meta", attrs={"property": "og:title"})
                        if og_title:
                            title = og_title.get("content", "")
                    
                    # Tarih elementi
                    publish_date = ""
                    date_selectors = [
                        'meta[property="article:published_time"]',
                        'time', 
                        'span.date', 
                        'div.date',
                        'span.tarih',
                        'div.publish-date',
                        'div.news-date',
                        'span.news-time'
                    ]
                    
                    for selector in date_selectors:
                        date_element = soup.select_one(selector)
                        if date_element:
                            if selector.startswith('meta'):
                                publish_date = date_element.get('content', '')
                            else:
                                publish_date = date_element.get_text().strip()
                            if publish_date:
                                break
                    
                    # İçerik bulunduysa ve yeterince uzunsa sonucu döndür
                    if content and len(content) > 50:
                        return {
                            "title": title,
                            "content": content,
                            "publish_date": publish_date
                        }
                else:
                    last_error = f"HTTP hatası: {response.status_code}"
                    display_log_message(last_error, log_container, "warning")
                
            except Exception as req_error:
                last_error = str(req_error)
                display_log_message(f"İstek hatası: {last_error}", log_container, "warning")
                continue
        
        # Tüm user-agent'lar denendi, içerik alınamadı
        if not content:
            display_log_message(f"İçerik çekilemedi: {last_error}", log_container, "error")
            return None
                
    except Exception as e:
        display_log_message(f"İçerik çekerken hata: {str(e)}", log_container, "error")
        return None

# Geçici bir haber analiz fonksiyonu ekleyelim
def analyze_news(url, log_container=None):
    """Haberleri analiz eden fonksiyon"""
    try:
        if log_container:
            display_log_message(f"Haber analiz ediliyor: {url}", log_container)
        
        # URL kontrolü
        if not url or url == "#":
            if log_container:
                display_log_message("Geçersiz URL", log_container, "error")
            return {
                "success": False,
                "error": "Geçersiz URL"
            }
            
        # Web sayfasını Requests ile çekmeyi dene
        try:
            # Haber içeriğini çek
            news_data = fetch_news_content(url, log_container)
            if not news_data or not news_data.get("content"):
                if log_container:
                    display_log_message("İçerik çekilemedi", log_container, "warning")
                return analyze_news_with_gemini(url, log_container)
                
            content = news_data.get("content")
            
            # İçerik çok uzunsa kısaltma yap 
            if len(content) > 500:
                # Analiz için ilk 500 karakter
                analysis_content = content[:500]
            else:
                analysis_content = content
                
            # Duyarlılık analizi yap
            if log_container:
                if TRANSFORMERS_AVAILABLE:
                    display_log_message("Transformers ile duyarlılık analizi yapılıyor...", log_container)
                else:
                    display_log_message("Basit duyarlılık analizi kullanılıyor...", log_container)
            
            sentiment_model = load_sentiment_model()
            
            # Model fonksiyon veya pipeline olabilir
            if TRANSFORMERS_AVAILABLE:
                result = sentiment_model(analysis_content)[0]
                
                # Transformers sonucunu işle
                if result["label"] == "POSITIVE":
                    sentiment_score = result["score"]
                    sentiment_label = "Olumlu" if sentiment_score > 0.65 else ("Olumsuz" if sentiment_score < 0.35 else "Nötr")
                else:
                    sentiment_score = 1 - result["score"]  # NEGATIVE ise skoru tersine çevir
                    sentiment_label = "Olumlu" if sentiment_score > 0.65 else ("Olumsuz" if sentiment_score < 0.35 else "Nötr")
            else:
                # Basit analiz sonucunu kullan
                result = sentiment_model(analysis_content)
                sentiment_label = result.get("sentiment", "Nötr")
                sentiment_score = result.get("score", 0.5)
            
            # Özet oluştur (basit method)
            summary = content[:200] + "..." if len(content) > 200 else content
            
            # Duyarlılık açıklaması
            sentiment_explanation = get_sentiment_explanation(sentiment_score)
            
            # Sonuçları hazırla
            return {
                "success": True,
                "title": news_data.get("title", "Başlık Bulunamadı"),
                "authors": "Belirtilmemiş",
                "publish_date": news_data.get("publish_date", "Belirtilmemiş"),
                "content": content,
                "sentiment": sentiment_label,
                "sentiment_score": sentiment_score,
                "ai_summary": summary,
                "ai_analysis": {
                    "etki": sentiment_label.lower(),
                    "etki_sebebi": sentiment_explanation,
                    "önemli_noktalar": []
                }
            }
        except Exception as req_error:
            # İçerik çekme hatası
            if log_container:
                display_log_message(f"İçerik çekme hatası: {str(req_error)}", log_container, "warning")
        
        # İçerik çekilemedi veya hata oluştu, standart Gemini analizi kullan
        return analyze_news_with_gemini(url, log_container)
        
    except Exception as e:
        if log_container:
            display_log_message(f"Haber analizi sırasında hata: {str(e)}", log_container, "error")
        return {
            "success": False,
            "error": str(e)
        }

# Duyarlılık değerini yorumlama fonksiyonu
def get_sentiment_explanation(score):
    """Duyarlılık puanına göre açıklama döndürür"""
    if score >= 0.7:
        return "Bu haber piyasa/şirket için oldukça olumlu içerik barındırıyor. İlgili hisse için potansiyel yükseliş işareti olabilir."
    elif score >= 0.55:
        return "Bu haber genel olarak olumlu bir ton taşıyor, ancak kesin bir yatırım sinyali olarak yorumlanmamalı."
    elif score <= 0.3:
        return "Bu haber piyasa/şirket için olumsuz içerik barındırıyor. Dikkatli olmakta fayda var."
    elif score <= 0.45:
        return "Bu haber hafif olumsuz bir tona sahip, ancak yatırım kararınızı tek başına etkileyecek düzeyde değil."
    else:
        return "Bu haber nötr bir tona sahip, yatırım açısından belirgin bir sinyal içermiyor."

def render_stock_news_tab():
    """
    Hisse senedi haberleri sekmesini oluşturur
    """
    st.title("Hisse Senedi Haberleri 📰")
    
    # Gemini analizi için session state kontrol et
    if 'show_news_analysis' not in st.session_state:
        st.session_state.show_news_analysis = False
        st.session_state.news_url = ""
        st.session_state.news_analysis_results = None
    
    # Analiz edilmiş haberleri takip etmek için session state kontrol et
    if 'analyzed_news_ids' not in st.session_state:
        st.session_state.analyzed_news_ids = []
    
    # İşlem günlüğü expander'ı - varsayılan olarak kapalı
    log_expander = st.expander("İşlem Günlüğü (Detaylar için tıklayın)", expanded=False)
    
    # Eğer analiz gösterilmesi gerekiyorsa
    if st.session_state.show_news_analysis and st.session_state.news_url:
        with st.spinner("Haber analiz ediliyor..."):
            if st.session_state.news_analysis_results is None:
                # Analiz daha önce yapılmamışsa, analizi gerçekleştir
                display_log_message("Haber analizi başlatılıyor...", log_expander)
                analysis_results = analyze_news(st.session_state.news_url, log_expander)
                st.session_state.news_analysis_results = analysis_results
            else:
                # Daha önce yapılan analizi kullan
                analysis_results = st.session_state.news_analysis_results
            
            # Analiz sonuçlarını göster
            if analysis_results.get("success", False):
                # Haber içeriği için container
                with st.expander("Haber İçeriği", expanded=True):
                    st.markdown(f"## {analysis_results['title']}")
                    st.markdown(f"**Yazar:** {analysis_results['authors']} | **Tarih:** {analysis_results['publish_date']}")
                    st.markdown("---")
                    st.markdown(analysis_results['content'])
                
                # Analiz sonuçları için container
                st.subheader("Yapay Zeka Analizi")
                
                # Duyarlılık analizi
                sentiment = analysis_results['sentiment']
                sentiment_score = analysis_results['sentiment_score']
                sentiment_color = "green" if sentiment == "Olumlu" else ("red" if sentiment == "Olumsuz" else "gray")
                sentiment_explanation = get_sentiment_explanation(sentiment_score)
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("Duyarlılık", sentiment, f"{sentiment_score:.2f}")
                
                with col2:
                    st.markdown(f"""
                    <div style="border-left:5px solid {sentiment_color}; padding-left:15px; margin-top:10px;">
                    <h4>Haber Özeti</h4>
                    <p>{analysis_results.get('ai_summary', 'Özet yapılamadı.')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Duyarlılık açıklaması ekle
                    st.markdown(f"""
                    <div style="margin-top:10px; padding:10px; background-color:#f8f9fa; border-radius:5px;">
                    <p><strong>Yorum:</strong> {sentiment_explanation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Piyasa etkisi
                impact_analysis = analysis_results.get('ai_analysis', {})
                impact = impact_analysis.get('etki', 'nötr')
                impact_reason = impact_analysis.get('etki_sebebi', 'Belirtilmemiş')
                impact_color = "green" if impact == "olumlu" else ("red" if impact == "olumsuz" else "gray")
                
                st.markdown(f"""
                <div style="border-left:5px solid {impact_color}; padding-left:15px; margin-top:20px;">
                <h4>Piyasa Etkisi: {impact.title()}</h4>
                <p>{impact_reason}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Önemli noktalar
                key_points = impact_analysis.get('önemli_noktalar', [])
                if key_points:
                    st.markdown("### Önemli Noktalar")
                    for point in key_points:
                        st.markdown(f"- {point}")
                
                # Özet ve aksiyon önerileri burada...
                
                # Kapat butonu
                cols = st.columns([3, 1])
                
                # Kapat butonu
                with cols[1]:
                    if st.button("Analizi Kapat", key="close_analysis"):
                        st.session_state.show_news_analysis = False
                        st.session_state.news_url = ""
                        st.session_state.news_analysis_results = None
                        st.rerun()
            else:
                st.error(f"Haber analizi yapılamadı: {analysis_results.get('error', 'Bilinmeyen hata')}")
                if st.button("Geri Dön"):
                    st.session_state.show_news_analysis = False
                    st.session_state.news_url = ""
                    st.session_state.news_analysis_results = None
                    st.rerun()
        
        # Analiz modu aktifse, diğer içerikleri gösterme
        return
    
    # Normal haber arama arayüzü - analiz modu aktif değilse
    # Daha kompakt bir arayüz için sütunlar
    col1, col2, col3 = st.columns([3, 1.5, 1])
    
    with col1:
        # Config'den default stock al
        default_stock = ML_MODEL_PARAMS.get("default_stock", "THYAO")
        stock_symbol = st.text_input("Hisse Senedi Kodu (örn: THYAO)", default_stock, key="news_stock_symbol")
    
    with col2:
        # Haber kaynakları seçimi
        available_providers = [
            "Google News", 
            "Yahoo Finance"
        ]
        # Diğer tüm kaynakları kaldırdık (KAP, Direct BIST, Bing News, TradingView)
        selected_providers = st.multiselect(
            "Haber Kaynakları", 
            available_providers,
            default=available_providers,  # Tüm kaynakları otomatik seçili yap
            key="news_providers"
        )
    
    with col3:
        # Config'den max results seçeneklerini al
        max_results_options = ML_MODEL_PARAMS.get("max_news_results", [5, 10, 15, 20])
        max_results = st.selectbox("Maksimum Haber", max_results_options, index=1)
        search_btn = st.button("Haberleri Getir")
    
    # Sonuçlar için container - tüm sonuçları burada göstereceğiz
    results_container = st.container()
    
    # Eğer analiz gösterilmiyorsa normal haber listesini göster
    if search_btn or ('news_last_symbol' in st.session_state and st.session_state.news_last_symbol == stock_symbol):
        stock_symbol = stock_symbol.upper().strip()
        st.session_state.news_last_symbol = stock_symbol
        
        display_log_message(f"{stock_symbol} ile ilgili haberler aranıyor...", log_expander)
        
        # Spinner'ı kaldırıyoruz, doğrudan results_container içinde çalışacağız
        with results_container:
            try:
                # En az bir kaynak seçilmemişse uyarı ver
                if not selected_providers:
                    st.warning("Lütfen en az bir haber kaynağı seçin.")
                    return
                
                # Get news data, progress göstergesini log expander'ına yönlendir
                display_log_message(f"Haberler getiriliyor... (Bu işlem biraz zaman alabilir)", log_expander)
                news_df = get_stock_news(stock_symbol, max_results=max_results, 
                                        progress_container=log_expander,
                                        providers=selected_providers)
                
                if news_df is not None and len(news_df) > 0:
                    display_log_message(f"{len(news_df)} haber bulundu", log_expander, "success")
                    
                    # Liste olarak dönen haberleri DataFrame'e dönüştür
                    news_df = pd.DataFrame(news_df)
                    
                    # Alan isimlerini kontrol et ve gerekirse düzelt
                    # Eğer 'published_datetime' varsa ama 'published_date' yoksa, ismini değiştir
                    if 'published_datetime' in news_df.columns and 'published_date' not in news_df.columns:
                        news_df = news_df.rename(columns={'published_datetime': 'published_date'})
                    
                    # 'pub_date' alanını ekle
                    if 'pub_date' not in news_df.columns and 'published_datetime' in news_df.columns:
                        news_df['pub_date'] = news_df['published_datetime']
                    elif 'pub_date' not in news_df.columns and 'published_date' in news_df.columns:
                        news_df['pub_date'] = news_df['published_date']
                    
                    # 'sentiment' alanı yoksa, varsayılan olarak nötr değerler ekle
                    if 'sentiment' not in news_df.columns:
                        news_df['sentiment'] = 0.5  # Nötr değer
                    
                    # 'sentiment' türünü kontrol et ve sayısal değere çevir
                    news_df['sentiment'] = news_df['sentiment'].apply(
                        lambda x: 0.5 if isinstance(x, str) and x == "Nötr" else 
                                 (0.8 if isinstance(x, str) and x == "Olumlu" else 
                                 (0.2 if isinstance(x, str) and x == "Olumsuz" else 
                                 (float(x) if isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '', 1).isdigit()) else 0.5)))
                    )
                    
                    # 'url' alanı yoksa, 'link' alanını kullan
                    if 'url' not in news_df.columns and 'link' in news_df.columns:
                        news_df['url'] = news_df['link']
                    
                    # image_url alanını kontrol et, yoksa boş string ekle
                    if 'image_url' not in news_df.columns:
                        news_df['image_url'] = ""
                    
                    # Her bir haberde eksik alanları kontrol et ve varsayılan değerlerle doldur
                    for index, row in news_df.iterrows():
                        for field in ['title', 'source', 'summary', 'url', 'link']:
                            if field not in row or pd.isna(row[field]) or row[field] == '':
                                if field == 'title':
                                    news_df.at[index, field] = 'Başlık Yok'
                                elif field == 'source':
                                    news_df.at[index, field] = 'Kaynak Belirsiz'
                                elif field == 'summary':
                                    news_df.at[index, field] = 'Özet alınamadı.'
                                elif field in ['url', 'link']:
                                    news_df.at[index, field] = '#'
                                    
                    # 'provider' alanını ekleyelim, yoksa
                    if 'provider' not in news_df.columns:
                        news_df['provider'] = 'Google News'
                    
                    # Display news
                    st.subheader(f"{stock_symbol} ile İlgili Haberler")
                    
                    # Displaying news articles
                    with st.container():
                        # Results header
                        st.markdown("""
                        <div style="margin-top:20px; margin-bottom:10px;">
                            <h2 style="color:#333; border-bottom:2px solid #2196F3; padding-bottom:8px;">📰 Haber Sonuçları</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        display_log_message(f"{len(news_df)} adet {stock_symbol} ile ilgili haber bulundu", log_expander, "success")
                        
                        # Add filter options
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            sort_by = st.selectbox("Sıralama:", ["Tarih (Yeni->Eski)", "Tarih (Eski->Yeni)", "Duyarlılık (Olumlu)", "Duyarlılık (Olumsuz)"])
                        with col2:
                            sentiment_filter = st.selectbox("Duyarlılık Filtresi:", ["Tümü", "Olumlu", "Nötr", "Olumsuz"])
                        with col3:
                            source_filter = st.selectbox("Kaynak Filtresi:", ["Tümü"] + news_df["source"].unique().tolist())
                        
                        # Daha modern bir filtre görünümü oluştur
                        st.markdown("""
                        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 15px 0; border: 1px solid #e9ecef;">
                            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                <div style="color: #555; font-size: 0.9rem; font-weight: bold;">
                                    <i class="material-icons" style="font-size: 0.9rem; vertical-align: middle;">filter_list</i> 
                                    Aktif Filtreler
                                </div>
                            </div>
                            <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                        """, unsafe_allow_html=True)
                        
                        # Sıralama filtresi
                        sort_icon = "arrow_downward" if "Eski->Yeni" in sort_by or "Olumsuz" in sort_by else "arrow_upward"
                        sort_color = "#5C6BC0" # Indigo rengi
                        st.markdown(f"""
                        <div style="background-color: white; padding: 8px 15px; border-radius: 20px; display: inline-block; 
                                    border: 1px solid {sort_color}; color: {sort_color};">
                            <i class="material-icons" style="font-size: 0.8rem; vertical-align: middle;">{sort_icon}</i>
                            <span style="font-size: 0.85rem; vertical-align: middle;"> {sort_by}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Duyarlılık filtresi
                        if sentiment_filter != "Tümü":
                            sentiment_color = "#4CAF50" if sentiment_filter == "Olumlu" else ("#F44336" if sentiment_filter == "Olumsuz" else "#FF9800")
                            st.markdown(f"""
                            <div style="background-color: white; padding: 8px 15px; border-radius: 20px; display: inline-block; 
                                        border: 1px solid {sentiment_color}; color: {sentiment_color};">
                                <span style="font-size: 0.85rem;"> {sentiment_filter} Haberler</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Kaynak filtresi
                        if source_filter != "Tümü":
                            st.markdown(f"""
                            <div style="background-color: white; padding: 8px 15px; border-radius: 20px; display: inline-block; 
                                        border: 1px solid #009688; color: #009688;">
                                <span style="font-size: 0.85rem;"> Kaynak: {source_filter}</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        st.markdown("""
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Apply filters
                        filtered_df = news_df.copy()
                        
                        # Apply sentiment filter
                        if sentiment_filter == "Olumlu":
                            filtered_df = filtered_df[filtered_df["sentiment"] > 0.65]
                        elif sentiment_filter == "Olumsuz":
                            filtered_df = filtered_df[filtered_df["sentiment"] < 0.35]
                        elif sentiment_filter == "Nötr":
                            filtered_df = filtered_df[(filtered_df["sentiment"] >= 0.35) & (filtered_df["sentiment"] <= 0.65)]
                        
                        # Apply source filter
                        if source_filter != "Tümü":
                            filtered_df = filtered_df[filtered_df["source"] == source_filter]
                        
                        # Apply sorting
                        if sort_by == "Tarih (Yeni->Eski)":
                            filtered_df = filtered_df.sort_values(by="pub_date", ascending=False)
                        elif sort_by == "Tarih (Eski->Yeni)":
                            filtered_df = filtered_df.sort_values(by="pub_date", ascending=True)
                        elif sort_by == "Duyarlılık (Olumlu)":
                            filtered_df = filtered_df.sort_values(by="sentiment", ascending=False)
                        elif sort_by == "Duyarlılık (Olumsuz)":
                            filtered_df = filtered_df.sort_values(by="sentiment", ascending=True)
                        
                        # Display filtered news count
                        if len(filtered_df) != len(news_df):
                            st.info(f"Filtreleme sonucu: {len(filtered_df)} haber gösteriliyor (toplam {len(news_df)} haberden)")
                        
                        # Display each news article as a card
                        for _, news in filtered_df.iterrows():
                            # Determine sentiment color and label
                            sentiment_value = news["sentiment"]
                            if sentiment_value > 0.65:
                                sentiment_color = "#4CAF50"  # green
                                sentiment_label = "Olumlu"
                            elif sentiment_value < 0.35:
                                sentiment_color = "#F44336"  # red
                                sentiment_label = "Olumsuz"
                            else:
                                sentiment_color = "#FF9800"  # amber
                                sentiment_label = "Nötr"
                            
                            # Format date
                            try:
                                pub_date = news["pub_date"].strftime("%d.%m.%Y %H:%M") if not isinstance(news["pub_date"], str) else news["pub_date"]
                            except:
                                pub_date = "Tarih bilinmiyor"
                            
                            # Prepare summary
                            summary = news["summary"] if news["summary"] and news["summary"] != "Özet alınamadı" else "Bu haber için özet bulunmuyor."
                            if len(summary) > 280:
                                summary = summary[:280] + "..."
                            
                            # Get source
                            source = news["source"] if "source" in news else "Bilinmeyen Kaynak"
                            
                            # Eğer başlık boşsa veya "Başlık Yok" ise özeti başlık olarak kullan
                            title = news['title']
                            if not title or title == 'Başlık Yok':
                                # Özeti başlık olarak kullan (kısalt)
                                title = summary[:80] + "..." if len(summary) > 80 else summary
                            
                            # Streamlit native componenti ile haber kartı oluştur
                            with st.container():
                                st.markdown(f"""<div style="margin-bottom:10px;"></div>""", unsafe_allow_html=True)
                                cols = st.columns([3, 1])
                                with cols[0]:
                                    st.markdown(f"### {title}")
                                    st.markdown(f"""
                                    <div style="font-size:0.85rem; color:#666; margin-bottom:8px;">
                                        <span style="background-color:#f0f0f0; padding:3px 6px; border-radius:3px; margin-right:8px;">
                                            <strong>{source}</strong>
                                        </span>
                                        <span style="color:#888;">
                                            {pub_date}
                                        </span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    st.markdown(f"""
                                    <div style="padding: 10px; border-left: 3px solid #2196F3; background-color: #f8f9fa; margin-bottom: 10px;">
                                        {summary}
                                    </div>
                                    """, unsafe_allow_html=True)
                                    st.markdown(f"""
                                    <a href="{news['link']}" target="_blank" style="text-decoration:none;">
                                        <div style="display:inline-block; padding:5px 15px; background-color:#2196F3; color:white; 
                                                    border-radius:4px; font-weight:bold; text-align:center; margin-top:5px;">
                                            Haberi Oku
                                        </div>
                                    </a>
                                    """, unsafe_allow_html=True)
                                with cols[1]:
                                    sentiment_box = f"""
                                    <div style="background-color:{sentiment_color}; color:white; padding:10px; border-radius:8px; 
                                              text-align:center; box-shadow:0 2px 5px rgba(0,0,0,0.1);">
                                        <h4 style="margin:0; font-size:1.2rem;">{sentiment_label}</h4>
                                        <p style="margin:0; font-size:1.5rem; font-weight:bold;">{sentiment_value:.2f}</p>
                                    </div>
                                    """
                                    st.markdown(sentiment_box, unsafe_allow_html=True)
                                    
                                    # Eğer resim varsa göster
                                    if "image_url" in news and news["image_url"]:
                                        try:
                                            st.image(news["image_url"], use_column_width=True)
                                        except Exception as img_error:
                                            st.warning(f"Görsel yüklenemedi: {str(img_error)}")
                                
                                st.markdown("""
                                <hr style="height:1px; border:none; background-color:#e0e0e0; margin:15px 0;">
                                """, unsafe_allow_html=True)
                        
                        if len(filtered_df) == 0:
                            st.warning("Filtrelenen sonuçlarda haber bulunamadı. Lütfen filtre seçeneklerini değiştirin.")
                    
                    # News sentiment analysis
                    st.markdown("""
                    <div style="margin-top:20px; margin-bottom:10px;">
                        <h2 style="color:#333; border-bottom:2px solid #2196F3; padding-bottom:8px;">📊 Haber Duyarlılık Analizi</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Calculate sentiment stats
                    positive_news = news_df[news_df['sentiment'] > 0.65]
                    negative_news = news_df[news_df['sentiment'] < 0.35]
                    neutral_news = news_df[(news_df['sentiment'] >= 0.35) & (news_df['sentiment'] <= 0.65)]
                    
                    # Haber sayıları
                    pos_count = len(positive_news)
                    neg_count = len(negative_news)
                    neu_count = len(neutral_news)
                    total_count = len(news_df)
                    
                    # Ortalama duyarlılık skoru hesaplanıyor
                    avg_sentiment = news_df['sentiment'].mean() if total_count > 0 else 0.5
                    
                    # Duyarlılık dağılımını göster - modern görünüm
                    col1, col2, col3 = st.columns(3)
                    
                    def create_metric_html(title, value, percentage, color):
                        return f"""
                        <div style="background-color: white; border-radius: 10px; padding: 15px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <h3 style="color: {color}; margin-bottom: 5px;">{value}</h3>
                            <div style="color: #666; font-size: 0.9rem; margin-bottom: 10px;">{title}</div>
                            <div style="background-color: #f0f0f0; height: 5px; border-radius: 5px; margin-top: 10px;">
                                <div style="width: {percentage}%; height: 100%; background-color: {color}; border-radius: 5px;"></div>
                            </div>
                            <div style="color: #888; font-size: 0.8rem; text-align: right; margin-top: 5px;">{percentage}%</div>
                        </div>
                        """
                    
                    with col1:
                        if total_count > 0:
                            positive_percentage = round((pos_count / total_count) * 100)
                            # st.metric("Olumlu Haberler", pos_count, f"{positive_percentage}%")
                            st.markdown(create_metric_html("Olumlu Haberler", pos_count, positive_percentage, "#4CAF50"), unsafe_allow_html=True)
                        else:
                            # st.metric("Olumlu Haberler", 0, "0%")
                            st.markdown(create_metric_html("Olumlu Haberler", 0, 0, "#4CAF50"), unsafe_allow_html=True)
                    
                    with col2:
                        if total_count > 0:
                            neutral_percentage = round((neu_count / total_count) * 100)
                            # st.metric("Nötr Haberler", neu_count, f"{neutral_percentage}%")
                            st.markdown(create_metric_html("Nötr Haberler", neu_count, neutral_percentage, "#FF9800"), unsafe_allow_html=True)
                        else:
                            # st.metric("Nötr Haberler", 0, "0%")
                            st.markdown(create_metric_html("Nötr Haberler", 0, 0, "#FF9800"), unsafe_allow_html=True)
                    
                    with col3:
                        if total_count > 0:
                            negative_percentage = round((neg_count / total_count) * 100)
                            # st.metric("Olumsuz Haberler", neg_count, f"{negative_percentage}%")
                            st.markdown(create_metric_html("Olumsuz Haberler", neg_count, negative_percentage, "#F44336"), unsafe_allow_html=True)
                        else:
                            # st.metric("Olumsuz Haberler", 0, "0%")
                            st.markdown(create_metric_html("Olumsuz Haberler", 0, 0, "#F44336"), unsafe_allow_html=True)
                    
                    # Genel duyarlılık skoru kartı
                    sentiment_label = "Olumlu" if avg_sentiment > 0.65 else ("Olumsuz" if avg_sentiment < 0.35 else "Nötr")
                    sentiment_color = "#4CAF50" if avg_sentiment > 0.65 else ("#F44336" if avg_sentiment < 0.35 else "#FF9800")
                    
                    # Streamlit native komponentlerini kullanarak gösterme
                    st.markdown("### Genel Duyarlılık Skoru")
                    cols = st.columns([1, 2])
                    with cols[0]:
                        st.markdown(f"""
                        <div style="background-color: {sentiment_color}; color: white; padding: 20px; border-radius: 10px; 
                                   text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                            <div style="font-size: 0.8rem; margin-bottom: 5px; opacity: 0.9;">Ortalama Skor</div>
                            <h2 style="margin:0; font-size: 2.5rem;">{avg_sentiment:.2f}</h2>
                            <p style="margin:5px 0 0 0; font-size: 1.2rem; font-weight: bold;">{sentiment_label}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[1]:
                        # Daha modern bir özet kutusu
                        st.markdown(f"""
                        <div style="background-color: white; padding: 20px; border-radius: 10px; height: 100%;
                                  box-shadow: 0 2px 5px rgba(0,0,0,0.05); border-left: 5px solid {sentiment_color};">
                            <h4 style="margin-top:0; color: #333;">Duyarlılık Özeti</h4>
                            <p style="margin-bottom:10px; font-size: 1rem;">
                                <strong>{stock_symbol}</strong> hissesinin haberlerine göre genel duyarlılık 
                                <span style="color:{sentiment_color}; font-weight:bold;">{sentiment_label.lower()}</span> düzeydedir.
                            </p>
                            <p style="color: #666; font-size: 0.9rem;">
                                Bu analiz, incelenen <strong>{total_count}</strong> haberin duyarlılık değerlendirmesine dayanmaktadır.
                                <br>
                                <span style="font-size: 0.8rem; color: #888; font-style: italic;">
                                    ({pos_count} olumlu, {neu_count} nötr, {neg_count} olumsuz haber)
                                </span>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Günlük duyarlılık özeti ekle
                    if not pd.api.types.is_datetime64_any_dtype(news_df['pub_date']):
                        news_df['pub_date'] = pd.to_datetime(news_df['pub_date'], errors='coerce')
                    
                    # Timezone bilgisini kontrol et ve gerekirse kaldır
                    if hasattr(news_df['pub_date'].dtype, 'tz') and news_df['pub_date'].dtype.tz is not None:
                        # Timezone bilgisi var, UTC'den dönüştür
                        news_df['pub_date'] = news_df['pub_date'].dt.tz_localize(None)
                    
                    # Son 24 saat, 3 gün ve 7 günlük haber analizini yap
                    now = pd.Timestamp.now()
                    last_24h = now - pd.Timedelta(days=1)
                    last_3d = now - pd.Timedelta(days=3)
                    last_7d = now - pd.Timedelta(days=7)
                    
                    # Zaman aralıklarındaki haberleri filtrele
                    news_24h = news_df[news_df['pub_date'] >= last_24h]
                    news_3d = news_df[news_df['pub_date'] >= last_3d]
                    news_7d = news_df[news_df['pub_date'] >= last_7d]
                    
                    # Her zaman aralığı için duyarlılık hesapla
                    sentiment_24h = news_24h['sentiment'].mean() if len(news_24h) > 0 else None
                    sentiment_3d = news_3d['sentiment'].mean() if len(news_3d) > 0 else None
                    sentiment_7d = news_7d['sentiment'].mean() if len(news_7d) > 0 else None
                    
                    # Günlük özet kartını oluştur
                    st.markdown("### Dönemsel Duyarlılık Özeti")
                    
                    # 3 Sütunlu bir düzen oluştur
                    col1, col2, col3 = st.columns(3)
                    
                    # Son 24 saat
                    with col1:
                        if sentiment_24h is not None:
                            label_24h = "Olumlu" if sentiment_24h > 0.6 else ("Olumsuz" if sentiment_24h < 0.4 else "Nötr")
                            color_24h = "#4CAF50" if sentiment_24h > 0.6 else ("#F44336" if sentiment_24h < 0.4 else "#FF9800")
                            st.markdown("##### Son 24 Saat")
                            # st.metric("Duyarlılık", f"{sentiment_24h:.2f}", f"{label_24h}")
                            # st.write(f"{len(news_24h)} haber")
                            st.markdown(f"""
                            <div style="background-color: white; padding: 15px; border-radius: 8px; border-left: 5px solid {color_24h}; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                                <div style="font-size: 1.2rem; font-weight: bold; color: {color_24h};">{sentiment_24h:.2f}</div>
                                <div style="font-size: 0.9rem; color: #666; margin-bottom: 5px;">{label_24h}</div>
                                <div style="font-size: 0.8rem; color: #888; text-align: right;">
                                    <span style="background-color: #f0f0f0; padding: 2px 8px; border-radius: 10px;">{len(news_24h)} haber</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("##### Son 24 Saat")
                            # st.write("Veri yok")
                            # st.write("0 haber")
                            st.markdown("""
                            <div style="background-color: white; padding: 15px; border-radius: 8px; border-left: 5px solid #ccc; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                                <div style="font-size: 1.2rem; font-weight: bold; color: #888;">-</div>
                                <div style="font-size: 0.9rem; color: #666; margin-bottom: 5px;">Veri yok</div>
                                <div style="font-size: 0.8rem; color: #888; text-align: right;">
                                    <span style="background-color: #f0f0f0; padding: 2px 8px; border-radius: 10px;">0 haber</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Son 3 gün
                    with col2:
                        if sentiment_3d is not None:
                            label_3d = "Olumlu" if sentiment_3d > 0.6 else ("Olumsuz" if sentiment_3d < 0.4 else "Nötr")
                            color_3d = "#4CAF50" if sentiment_3d > 0.6 else ("#F44336" if sentiment_3d < 0.4 else "#FF9800")
                            st.markdown("##### Son 3 Gün")
                            # st.metric("Duyarlılık", f"{sentiment_3d:.2f}", f"{label_3d}")
                            # st.write(f"{len(news_3d)} haber")
                            st.markdown(f"""
                            <div style="background-color: white; padding: 15px; border-radius: 8px; border-left: 5px solid {color_3d}; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                                <div style="font-size: 1.2rem; font-weight: bold; color: {color_3d};">{sentiment_3d:.2f}</div>
                                <div style="font-size: 0.9rem; color: #666; margin-bottom: 5px;">{label_3d}</div>
                                <div style="font-size: 0.8rem; color: #888; text-align: right;">
                                    <span style="background-color: #f0f0f0; padding: 2px 8px; border-radius: 10px;">{len(news_3d)} haber</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("##### Son 3 Gün")
                            # st.write("Veri yok")
                            # st.write("0 haber")
                            st.markdown("""
                            <div style="background-color: white; padding: 15px; border-radius: 8px; border-left: 5px solid #ccc; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                                <div style="font-size: 1.2rem; font-weight: bold; color: #888;">-</div>
                                <div style="font-size: 0.9rem; color: #666; margin-bottom: 5px;">Veri yok</div>
                                <div style="font-size: 0.8rem; color: #888; text-align: right;">
                                    <span style="background-color: #f0f0f0; padding: 2px 8px; border-radius: 10px;">0 haber</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Son 7 gün
                    with col3:
                        if sentiment_7d is not None:
                            label_7d = "Olumlu" if sentiment_7d > 0.6 else ("Olumsuz" if sentiment_7d < 0.4 else "Nötr")
                            color_7d = "#4CAF50" if sentiment_7d > 0.6 else ("#F44336" if sentiment_7d < 0.4 else "#FF9800")
                            st.markdown("##### Son 7 Gün")
                            # st.metric("Duyarlılık", f"{sentiment_7d:.2f}", f"{label_7d}")
                            # st.write(f"{len(news_7d)} haber")
                            st.markdown(f"""
                            <div style="background-color: white; padding: 15px; border-radius: 8px; border-left: 5px solid {color_7d}; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                                <div style="font-size: 1.2rem; font-weight: bold; color: {color_7d};">{sentiment_7d:.2f}</div>
                                <div style="font-size: 0.9rem; color: #666; margin-bottom: 5px;">{label_7d}</div>
                                <div style="font-size: 0.8rem; color: #888; text-align: right;">
                                    <span style="background-color: #f0f0f0; padding: 2px 8px; border-radius: 10px;">{len(news_7d)} haber</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("##### Son 7 Gün")
                            # st.write("Veri yok")
                            # st.write("0 haber")
                            st.markdown("""
                            <div style="background-color: white; padding: 15px; border-radius: 8px; border-left: 5px solid #ccc; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                                <div style="font-size: 1.2rem; font-weight: bold; color: #888;">-</div>
                                <div style="font-size: 0.9rem; color: #666; margin-bottom: 5px;">Veri yok</div>
                                <div style="font-size: 0.8rem; color: #888; text-align: right;">
                                    <span style="background-color: #f0f0f0; padding: 2px 8px; border-radius: 10px;">0 haber</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Karşılaştırma özeti
                    if sentiment_24h is not None and sentiment_7d is not None:
                        change = sentiment_24h - sentiment_7d
                        change_pct = (change / sentiment_7d) * 100 if sentiment_7d != 0 else 0
                        change_direction = "yükselmiş" if change > 0.05 else ("düşmüş" if change < -0.05 else "benzer kalmış")
                        
                        st.markdown("---")
                        st.markdown(f"**Karşılaştırma:** Son 24 saatteki duyarlılık, son 7 güne göre {abs(change_pct):.1f}% oranında {change_direction}.")
                        st.caption("Bu değişim, son haberlerdeki ton değişikliğini gösterir ve hisse senedi hareketlerini doğrudan yansıtmayabilir.")
                    
                    # Duyarlılık zaman serisi grafiğini oluştur
                    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
                    st.subheader("Duyarlılık Zaman Serisi")
                    
                    # Zaman serisi grafiği için verileri hazırla
                    if not news_df.empty and 'sentiment' in news_df.columns and 'pub_date' in news_df.columns:
                        try:
                            # Tarih sütununu datetime formatına dönüştür
                            news_df['pub_date'] = pd.to_datetime(news_df['pub_date'])
                            
                            # Timezone bilgisini kontrol et ve gerekirse kaldır
                            if hasattr(news_df['pub_date'].dtype, 'tz') and news_df['pub_date'].dtype.tz is not None:
                                # Timezone bilgisi var, UTC'den dönüştür
                                news_df['pub_date'] = news_df['pub_date'].dt.tz_localize(None)
                            
                            # Tarihe göre sırala
                            news_df_sorted = news_df.sort_values('date')
                            
                            # Tarih ve duyarlılık verilerini al
                            dates = news_df_sorted['date']
                            sentiments = news_df_sorted['sentiment']
                            
                            # Hareketli ortalama hesapla (eğer yeterli veri varsa)
                            window_size = min(5, len(news_df_sorted))
                            if window_size > 1:
                                news_df_sorted['rolling_sentiment'] = news_df_sorted['sentiment'].rolling(window=window_size).mean()
                            
                            # Grafiği oluştur
                            fig = plt.figure(figsize=(10, 5))
                            ax = fig.add_subplot(111)
                            
                            # Duyarlılık verilerini çiz
                            ax.scatter(dates, sentiments, alpha=0.6, color='blue', label='Haber Duyarlılık')
                            
                            # Hareketli ortalamayı çiz (eğer yeterli veri varsa)
                            if window_size > 1:
                                ax.plot(dates, news_df_sorted['rolling_sentiment'], color='red', linewidth=2, label=f'{window_size} Haberlik Hareketli Ortalama')
                            
                            # 0.5 çizgisini ekle (nötr duyarlılık)
                            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Nötr Duyarlılık')
                            
                            # Olumlu bölge
                            ax.axhspan(0.6, 1.0, alpha=0.2, color='green', label='Olumlu Bölge')
                            
                            # Olumsuz bölge
                            ax.axhspan(0.0, 0.4, alpha=0.2, color='red', label='Olumsuz Bölge')
                            
                            # Grafik özelliklerini ayarla
                            ax.set_ylabel('Duyarlılık Skoru')
                            ax.set_xlabel('Tarih')
                            ax.set_ylim(0, 1)
                            ax.legend(loc='best')
                            ax.grid(True, alpha=0.3)
                            
                            # Tarihleri formatlama
                            fig.autofmt_xdate()
                            
                            # Grafik başlığı
                            plt.title(f"{stock_symbol} Hisse Haberleri Duyarlılık Trendi", fontsize=12)
                            plt.tight_layout()
                            
                            # Streamlit'te göster
                            st.pyplot(fig)
                            
                            # Memory leak'i önlemek için figür'ü kapat
                            plt.close(fig)
                            
                            # Trend analizi
                            if len(news_df_sorted) >= 3:
                                # Eğim hesapla
                                x = np.arange(len(sentiments))
                                slope, _, _, _, _ = linregress(x, sentiments)
                                
                                # Trend mesajı
                                trend_message = ""
                                if slope > 0.05:
                                    trend_message = f"<p style='color:#4CAF50;'>📈 <strong>Yükselen Trend:</strong> {stock_symbol} için haber duyarlılığı olumlu yönde artıyor.</p>"
                                elif slope < -0.05:
                                    trend_message = f"<p style='color:#F44336;'>📉 <strong>Düşen Trend:</strong> {stock_symbol} için haber duyarlılığı olumsuz yönde ilerliyor.</p>"
                                else:
                                    trend_message = f"<p style='color:#FF9800;'>📊 <strong>Stabil Trend:</strong> {stock_symbol} için haber duyarlılığında önemli bir değişim görülmüyor.</p>"
                                
                                st.markdown(trend_message, unsafe_allow_html=True)
                        
                        except Exception as e:
                            st.warning(f"Zaman serisi grafiği oluşturulurken bir hata oluştu: {str(e)}")
                            display_log_message(f"Grafik hatası: {str(e)}", log_expander, "warning")
                    else:
                        st.info("Zaman serisi grafiği için yeterli veri bulunmuyor.")
            
                    # Duyarlılık zaman serisi grafiğini oluştur
                    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
                    st.subheader("Haber Kaynakları Analizi")
                    
                    # Haber kaynakları analizi için verileri hazırla
                    if not news_df.empty and 'source' in news_df.columns:
                        try:
                            # Kaynakların sayısını hesapla
                            source_counts = news_df['source'].value_counts()
                            
                            # Pasta grafik için hazırlık
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                # Pasta grafiği oluştur
                                fig, ax = plt.subplots(figsize=(8, 6))
                                wedges, texts, autotexts = ax.pie(
                                    source_counts.values, 
                                    labels=None,
                                    autopct='%1.1f%%',
                                    startangle=90,
                                    wedgeprops={'edgecolor': 'white', 'linewidth': 1},
                                    textprops={'fontsize': 12, 'color': 'white'},
                                    explode=[0.05] * len(source_counts)  # Tüm dilimleri biraz dışarı çıkar
                                )
                                
                                # Pasta grafiğin görünümünü özelleştir
                                plt.setp(autotexts, size=10, weight="bold")
                                ax.set_title(f"{stock_symbol} için Haber Kaynakları Dağılımı", fontsize=14)
                                plt.tight_layout()
                                
                                # Streamlit'te göster
                                st.pyplot(fig)
                                
                                # Memory leak'i önlemek için figür'ü kapat
                                plt.close(fig)
                            
                            with col2:
                                # Kaynak sayılarını tablo olarak göster
                                st.markdown("<h4 style='text-align: center;'>Kaynak Dağılımı</h4>", unsafe_allow_html=True)
                                
                                # Görsel tabloya çevir
                                for source, count in source_counts.items():
                                    percentage = (count / source_counts.sum()) * 100
                                    st.markdown(
                                        f"<div style='padding: 8px; margin-bottom: 8px; background-color: rgba(0,0,0,0.05); border-radius: 5px;'>"
                                        f"<span style='font-weight: bold;'>{source}</span><br/>"
                                        f"{count} haber ({percentage:.1f}%)"
                                        f"</div>",
                                        unsafe_allow_html=True
                                    )
                                
                                # Kaynak çeşitliliği analizi
                                source_diversity = len(source_counts)
                                total_sources = len(news_df)
                                
                                if source_diversity >= 4:
                                    diversity_message = f"<p style='color:#4CAF50;'>✅ <strong>Yüksek Çeşitlilik:</strong> {source_diversity} farklı kaynak. Çeşitli perspektifler sunar.</p>"
                                elif source_diversity >= 2:
                                    diversity_message = f"<p style='color:#FF9800;'>⚠️ <strong>Orta Çeşitlilik:</strong> {source_diversity} farklı kaynak. Daha fazla kaynak çeşitliliği faydalı olabilir.</p>"
                                else:
                                    diversity_message = f"<p style='color:#F44336;'>⚠️ <strong>Düşük Çeşitlilik:</strong> Sadece {source_diversity} kaynak. Tek kaynağa dayalı görüş yanıltıcı olabilir.</p>"
                                
                                st.markdown(diversity_message, unsafe_allow_html=True)
                        
                        except Exception as e:
                            st.warning(f"Haber kaynakları grafiği oluşturulurken bir hata oluştu: {str(e)}")
                            display_log_message(f"Kaynak grafiği hatası: {str(e)}", log_expander, "warning")
                    else:
                        st.info("Haber kaynakları analizi için yeterli veri bulunmuyor.")
            
            except Exception as e:
                st.error(f"Haber arama sırasında bir hata oluştu: {str(e)}")
                display_log_message(f"Hata: {str(e)}", log_expander, "error")
                display_log_message(f"Hata detayı: {traceback.format_exc()}", log_expander, "error")
                st.error(traceback.format_exc())
    
    else:
        with results_container:
            st.info("Hisse senedi kodunu girin ve 'Haberleri Getir' butonuna tıklayın.") 
