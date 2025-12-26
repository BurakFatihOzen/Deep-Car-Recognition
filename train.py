# Egitim kodlari Mehmet tarafindan optimize edildi
# Önce gerekli kütüphaneleri A100 makinesine kuralım
!pip install ultralytics kagglehub

import kagglehub
import os
import shutil
import numpy as np
from ultralytics import YOLO
import yaml
from collections import Counter
import torch

# DONANIM KONTROLÜ (A100 VAR MI?) 
print(f"GPU DURUMU: {torch.cuda.get_device_name(0)}")
if "A100" not in torch.cuda.get_device_name(0):
    print("UYARI: Şu an A100 görünmüyor.")
else:
    print("A100 Aktif.")

# VERİYİ İNDİRİR 
print("Tüm veri seti indiriliyor (Cache)...")
path = kagglehub.dataset_download("prondeau/the-car-connection-picture-dataset")

TARGET_CLASS_COUNT = 100   # En popüler 100 Model
IMAGES_PER_CLASS = 9999    

base_dir = "/content/ultimate_a100_project"
if os.path.exists(base_dir): shutil.rmtree(base_dir)
for split in ['train', 'valid']:
    os.makedirs(f"{base_dir}/{split}/images", exist_ok=True)
    os.makedirs(f"{base_dir}/{split}/labels", exist_ok=True)

#EN POPÜLER 100 MODELİ SEÇER
print("🔍 Veri madenciliği yapılıyor: En çok fotosu olan 100 model seçiliyor...")

model_names = []
all_files = []

for root, dirs, files in os.walk(path):
    for file in files:
        if file.lower().endswith('.jpg'):
            full_path = os.path.join(root, file)
            all_files.append(full_path)
            parts = file.split('_')
            if len(parts) >= 2:
                model_names.append(f"{parts[0]}_{parts[1]}")

# Sayım yapar ve Top 100'ü alır
class_counts = Counter(model_names)
top_models = [name for name, count in class_counts.most_common(TARGET_CLASS_COUNT)]
class_map = {name: i for i, name in enumerate(top_models)}

print(f"HEDEF BELİRLENDİ: {len(top_models)} Farklı Model.")
print(f"Örnekler: {top_models[:10]} ...")

# HIZLI ETİKETLEME 
print("Otomatik etiketleme başlıyor...")
# Etiketlemeyi 'X' (Extra Large) model ile yapıyoruz ki hata payı sıfıra yakın olsun
labeler = YOLO('yolov8x-seg.pt') 

counters = {name: 0 for name in top_models}
processed_count = 0

for file_path in all_files:
    filename = os.path.basename(file_path)
    parts = filename.split('_')
    if len(parts) < 2: continue
    car_name = f"{parts[0]}_{parts[1]}"

    if car_name in class_map: # Limit kontrolü yok
        
        # A100 olduğu için batch processing yapabiliriz ama kod karışmasın diye tek tek gidiyoruz.
        # A100 bunu milisaniyede yapar zaten.
        results = labeler.predict(file_path, classes=[2], verbose=False, conf=0.45, device=0)
        
        if results[0].masks:
            split = 'train' if np.random.rand() < 0.8 else 'valid'
            idx = counters[car_name]
            new_name = f"{car_name}_{idx}"
            
            # Kaydeder
            shutil.copy(file_path, f"{base_dir}/{split}/images/{new_name}.jpg")
            
            # Etiketler
            with open(f"{base_dir}/{split}/labels/{new_name}.txt", 'w') as f:
                for mask in results[0].masks.xyn:
                    line = f"{class_map[car_name]} " + " ".join(f"{x:.6f}" for x in mask.flatten())
                    f.write(line + "\n")
            
            counters[car_name] += 1
            processed_count += 1
            
            if processed_count % 500 == 0:
                print(f" {processed_count} resim işlendi...")

# --- 5. DATA.YAML ---
yaml_data = {
    'path': base_dir,
    'train': 'train/images',
    'val': 'valid/images',
    'nc': len(class_map),
    'names': [name.replace('_', ' ') for name in top_models]
}
with open(f"{base_dir}/data.yaml", 'w') as f:
    yaml.dump(yaml_data, f)

print(f"\nVERİ HAZIR! Toplam Resim: {processed_count}")
print("EĞİTİM BAŞLIYOR... (LARGE Model + Full Augmentation)")

# --- 6. ULTIMATE TRAINING (LARGE MODEL) ---
# A100 olduğu için 'Large' model kullanıyoruz.

model = YOLO('yolov8l-seg.pt') 

model.train(
    data=f"{base_dir}/data.yaml",
    epochs=50,       
    imgsz=640,
    batch=64,         # A100 gücü
    device=0,
    workers=16,     
    cache=True,       # RAM kullanımı açık
    
    augment=True,
    degrees=10.0,     # Biraz daha hafif çevirme
    mosaic=0.5,       # Mozaik etkisini azaltıldı
    mixup=0.0,       
    
    name='A100_Final_Model',
    patience=10       # 10 tur gelişmezse durur.
)
