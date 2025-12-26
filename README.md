# Human vs AI Metin Tespit Projesi

Bu proje, verilen bir metnin **insan tarafından mı yoksa yapay zeka tarafından mı yazıldığını** tespit etmek amacıyla geliştirilmiştir.

## Projenin Amacı
- İnsan ve yapay zeka tarafından yazılmış metinleri ayırt etmek  
- Makine öğrenmesi kullanarak metin sınıflandırması yapmak  
- Metin verileri üzerinde analiz ve testler gerçekleştirmek  

## Kullanılan Teknolojiler
- Python  
- Scikit-learn  
- Pandas  
- NumPy  
- TF-IDF Vectorizer  
- Logistic Regression  
- Pytest  

## Proje Yapısı
- Veri seti (insan ve yapay zeka metinleri)  
- Metin ön işleme ve vektörleştirme  
- Makine öğrenmesi modeli  
- White-Box test senaryoları  
- Kaydedilmiş model ve vektörizer dosyaları (.pkl)

## Çalışma Mantığı
1. Metinler TF-IDF yöntemi ile sayısal vektörlere dönüştürülür  
2. Logistic Regression, Naive Bayes, Linear SVC, LightGBM, Random Forest modeli eğitilir  
3. Girilen metin model tarafından analiz edilir  
4. Çıktı **Human (İnsan)** veya **AI (Yapay Zeka)** olarak üretilir  

## Test Süreci
- Modelin tahmin üretip üretmediği kontrol edilir  
- Boş veya hatalı girdiler test edilir  
- White-Box test yaklaşımı kullanılmıştır  

## Projeyi Çalıştırma
```bash
pip install -r requirements.txt
python main.py
