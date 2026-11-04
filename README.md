# 🍾 Derin Öğrenme (CNN / YOLOv8) ile Otomatik Şişe Sayım Sistemi

Bu proje, **tek bir kaynak fotoğraftan** görüntü işleme ile otomatik etiketleme yapıp, görseli farklı açılarda döndürerek (**Data Augmentation**) veri setini genişletir ve hazır **YOLOv8n (CNN)** nesne tespit modelini yeniden eğiterek (**Fine-Tuning**) şişeleri sayar.

---

## 📁 Proje Dosya Yapısı

- `original_bottles.jpg` : Kaynak şişe görseli (83 adet şişe içerir).
- `prepare_dataset.py` : Kaynak görseldeki kırmızı şişe ağızlarını LAB renk uzayı ve morfolojik filtrelerle otomatik tespit eder, $0^\circ - 355^\circ$ döndürme, aynalama, kırpma ve parlaklık çeşitlemeleriyle **133 görsellik** YOLO veri seti (`dataset/`) üretir.
- `train.py` : Hazır `yolov8n.pt` CNN modelini üretilen veri seti ile yeniden eğitir ve en iyi ağırlıkları `models/bottle_yolov8_best.pt` konumuna kaydeder.
- `predict.py` : Eğitilmiş modeli kullanarak görsel üzerindeki şişeleri tespit eder, numaralandırır, sayar ve sonuç görsellerini `outputs/` klasörüne kaydeder.
- `app.py` : Fotoğraf yükleyip veya canlı döndürme açılarıyla test yapabileceğiniz **Streamlit** görsel arayüzü.
- `baslat_arayuz.bat` : Görsel arayüzü çift tıklamayla başlatan kısayol dosyası.

---

## 🚀 Kullanım Adımları

### 1. Veri Setini Oluşturma ve Döndürme ile Çoğaltma
```bash
python prepare_dataset.py
```

### 2. Hazır Modeli (YOLOv8n CNN) Yeniden Eğitme
```bash
python train.py
```

### 3. Komut Satırından Sayım ve Test
```bash
python predict.py
```
Farklı bir görsel üzerinde test etmek için:
```bash
python predict.py --image yeni_gorsel.jpg --conf 0.35
```

### 4. Görsel Arayüzü (Streamlit) Başlatma
`baslat_arayuz.bat` dosyasına çift tıklayabilir veya terminalden şu komutu çalıştırabilirsiniz:
```bash
streamlit run app.py
```

---

## 📊 Eğitim ve Test Sonuçları

- **Toplam Şişe Sayısı (Gerçek):** 83 Adet
- **Model Sayım Sonucu ($0^\circ, 90^\circ, 180^\circ, 270^\circ$):** **83 Adet (%100 Doğruluk)**
- **Doğrulama (Validation) Metrikleri:**
  - **Precision (Kesinlik):** `%99.6`
  - **Recall (Duyarlılık):** `%100.0`
  - **mAP@50:** `%99.5`
