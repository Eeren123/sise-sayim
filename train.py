import os
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch

# Windows Türkçe karakterli (Masaüstü vb.) yollar için OpenCV imread/imwrite yaması
_orig_imread = cv2.imread
_orig_imwrite = cv2.imwrite


def _unicode_imread(path, flags=cv2.IMREAD_COLOR):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return _orig_imread(path, flags)


def _unicode_imwrite(path, img, params=None):
    try:
        p = Path(str(path))
        p.parent.mkdir(parents=True, exist_ok=True)
        ext = p.suffix.lower() or ".jpg"
        ok, buf = cv2.imencode(ext, img, params or [])
        if ok:
            buf.tofile(str(p))
            return True
        return False
    except Exception:
        return _orig_imwrite(path, img, params)


cv2.imread = _unicode_imread
cv2.imwrite = _unicode_imwrite

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "dataset" / "data.yaml"
MODELS_DIR = BASE_DIR / "models"
RUNS_DIR = BASE_DIR / "runs"


def train_model(epochs: int = 20, imgsz: int = 640, batch: int = 8):
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Veri seti bulunamadı ({DATA_YAML}). Lütfen önce 'python prepare_dataset.py' çalıştırın."
        )

    device = "0" if torch.cuda.is_available() else "cpu"
    print("============================================================")
    print("Şeffaf + Kırmızı Şişe Tanıyan CNN (YOLOv8n) Eğitimi Başlıyor")
    print(f"Cihaz (Device): {'GPU (CUDA)' if device == '0' else 'CPU'}")
    print(f"Veri Seti     : {DATA_YAML}")
    print(f"Epoch Sayısı  : {epochs} | Görsel Boyutu: {imgsz}")
    print("============================================================")

    os.chdir(BASE_DIR)

    # Hazır (Pre-trained) YOLOv8n modelini yükle
    model = YOLO("yolov8n.pt")

    # Renkten bağımsız (şeffaf + kırmızı), ölçek ve perspektif çeşitlemeli eğitim
    results = model.train(
        data="dataset/data.yaml",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="runs",
        name="bottle_cnn_detector",
        exist_ok=True,
        degrees=180.0,       # 360 derece döndürme
        scale=0.55,          # Yakın-uzak kamera ölçek çeşitlemesi
        perspective=0.0004,  # Kamera eğim/perspektif toleransı
        hsv_h=0.5,           # Renk tonu değişimi (renk ezberini engeller)
        hsv_s=0.8,           # Doygunluk değişimi (şeffaf/renksiz şişelere uyum)
        hsv_v=0.4,           # Parlaklık değişimi
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.6,
        patience=15,
        workers=0,
        verbose=True
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    last_pt = Path(results.save_dir) / "weights" / "last.pt"
    target_pt = MODELS_DIR / "bottle_yolov8_best.pt"

    if best_pt.exists():
        shutil.copy2(best_pt, target_pt)
    elif last_pt.exists():
        shutil.copy2(last_pt, target_pt)

    print(f"\n[BAŞARILI] Şeffaf + Renkli Şişe Modeli Eğitimi Tamamlandı!")
    print(f"En iyi model ağırlıkları kaydedildi: {target_pt}")
    return target_pt


if __name__ == "__main__":
    has_gpu = torch.cuda.is_available()
    train_model(
        epochs=30 if has_gpu else 18,
        imgsz=640,
        batch=8
    )
