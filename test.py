import cv2
import numpy as np
import urllib.request
from ultralytics import YOLO

# MODELİ YÜKLEME İŞLEMİ
try:
    model = YOLO('bestdeep100.pt')
    print("Model yüklendi.")
    
    # Modelin bildiği bütün arabalarr
    print(f"Modelin bildiği toplam sınıf sayısı: {len(model.names)}")
    if len(model.names) > 0:
        print(f"Örnekler: {list(model.names.values())[:5]} ...")
        
except Exception as e:
    print(f"HATA: Model dosyası bulunamadı! ({e})")
    exit()

# RESMİ İNTERNETTEN ALIR
url = "https://www.log.com.tr/wp-content/uploads/2024/10/2025-Porsche-911-Carrera-T-Coupe-13-copy.jpg" 

print(f"\nResim indiriliyor: {url}")
try:
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    if img is None: raise ValueError("Boş resim")
except Exception as e:
    print(f"Resim hatası: {e}")
    exit()

# TAHMİN 
# %0.1 ihtimal ve %0.9 IOU ile sadece 1 tahmin alır
results = model.predict(img, conf=0.001, iou=0.9, max_det=1) 

# SONUCU ANALİZ EDER 
result = results[0]
boxes = result.boxes

if len(boxes) == 0:
    print("\nModel buna araba diyemedi.")
    print("Sebep: Resimdeki açı çok ters olabilir veya model eğitimi henüz oturmamış.")
else:
    # En iyi tahmini alır
    box = boxes[0]
    class_id = int(box.cls)
    conf = float(box.conf)
    name = model.names[class_id]
    
    print(f"\nTAHMİN EDİLDİ: {name}")
    print(f"Güven Oranı: %{conf*100:.2f}")
    
    # Ekrana çizer
    resim_cizili = result.plot()
    
    # Ekrana sığdırır
    h, w = resim_cizili.shape[:2]
    if h > 800:
        scale = 800 / h
        resim_cizili = cv2.resize(resim_cizili, (int(w*scale), int(h*scale)))

    cv2.imshow("Model Tahmini", resim_cizili)
    print("Resim açıldı. Kapatmak için resme tıkla ve bir tuşa bas.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()