import os
import random
import shutil
from pathlib import Path
import cv2
import numpy as np

# Sabit rastgelelik tohumu (tekrarlanabilirlik için)
random.seed(42)
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
SOURCE_IMAGE = BASE_DIR / "original_bottles.jpg"
DATASET_DIR = BASE_DIR / "dataset"


def imread_unicode(path: Path) -> np.ndarray:
    """Windows Türkçe karakterli (Masaüstü vb.) dosya yollarından güvenli görsel okuma."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Görsel okunamadı: {path}")
    return img


def imwrite_unicode(path: Path, img: np.ndarray, quality: int = 95) -> None:
    """Windows Türkçe karakterli dosya yollarına güvenli görsel yazma."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in [".jpg", ".jpeg"] else []
    ok, buf = cv2.imencode(ext, img, params)
    if ok:
        buf.tofile(str(path))


def detect_initial_bottles(img: np.ndarray):
    """
    Orijinal görseldeki 83 şişe ağzını otomatik tespit eder.
    Döndürür: [(cx, cy, bw, bh), ...] piksel cinsinden kutu listesi.
    """
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_chan = lab[:, :, 1]
    a_local = a_chan - cv2.GaussianBlur(a_chan, (61, 61), 15)

    box_mask = np.zeros((h, w), dtype=np.uint8)
    poly = np.array([
        [116, 74],
        [568, 62],
        [642, 722],
        [115, 738]
    ], dtype=np.int32)
    cv2.fillPoly(box_mask, [poly], 255)

    mask = ((a_local > 6.2) & (box_mask > 0)).astype(np.uint8) * 255
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if 20 <= bw <= 55 and 20 <= bh <= 55 and area > 180:
            raw_boxes.append((x, y, bw, bh))

    raw_boxes.sort(key=lambda b: (b[1] // 50, b[0]))

    pad = 4
    boxes = []
    for (x, y, bw, bh) in raw_boxes:
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        boxes.append((cx, cy, float(x2 - x1), float(y2 - y1)))

    return boxes


# ==============================================================================
# MADDE 1: KIRMIZI HALKALARI SİLİP "YAPAY ŞEFFAF ŞİŞE" VE KARIŞIK RENK ÜRETİMİ
# ==============================================================================
def convert_bottles_to_transparent(img: np.ndarray, boxes: list, transparent_ratio: float = 1.0) -> np.ndarray:
    """
    Seçilen orandaki şişelerin kırmızı halkalarını LAB uzayında nötrleyerek
    şeffaf plastik/cam şişe ağzı görünümüne (parlak renksiz halka) dönüştürür.
    transparent_ratio = 1.0 -> Tüm şişeler şeffaf
    transparent_ratio = 0.5 -> Şişelerin rastgele yarısı şeffaf, yarısı kırmızı
    """
    out = img.copy()
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = img.shape[:2]

    # Hangi şişelerin şeffafa dönüştürüleceğini seç
    num_trans = int(round(len(boxes) * transparent_ratio))
    selected_indices = set(random.sample(range(len(boxes)), num_trans))

    for idx, (cx, cy, bw, bh) in enumerate(boxes):
        if idx not in selected_indices:
            continue

        x1 = max(0, int(cx - bw * 0.6))
        y1 = max(0, int(cy - bh * 0.6))
        x2 = min(w, int(cx + bw * 0.6))
        y2 = min(h, int(cy + bh * 0.6))

        patch_L = lab[y1:y2, x1:x2, 0]
        patch_a = lab[y1:y2, x1:x2, 1]
        patch_b = lab[y1:y2, x1:x2, 2]

        # Kırmızı halka piksellerini bul (a kanalı yerel ortalamadan yüksek olanlar)
        red_strength = np.clip((patch_a - 133.0) / 25.0, 0.0, 1.0)

        # Kırmızılığı tamamen nötr şeffaf plastik/koli yansıma tonuna (~131) çek
        patch_a_new = patch_a * (1.0 - red_strength * 0.92) + 130.5 * (red_strength * 0.92)
        patch_b_new = patch_b * (1.0 - red_strength * 0.50) + 134.0 * (red_strength * 0.50)

        # Şeffaf şişe ağzındaki ışık kırılmasını (parlamayı) simüle etmek için L kanalını hafif güçlendir
        patch_L_new = np.clip(patch_L + red_strength * 18.0, 0, 255)

        lab[y1:y2, x1:x2, 0] = patch_L_new
        lab[y1:y2, x1:x2, 1] = patch_a_new
        lab[y1:y2, x1:x2, 2] = patch_b_new

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def shift_bottle_colors(img: np.ndarray, hue_shift: int = 90, desaturate: float = 1.0) -> np.ndarray:
    """Renk ezberini kırmak için ton (Hue) ve doygunluk (Saturation) dönüşümü uygular."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * desaturate, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ==============================================================================
# MADDE 2: DİZİLİM EZBERİNİ KIRMA (RASTGELE EKSİK ŞİŞE / BOŞLUK SİMÜLASYONU)
# ==============================================================================
def apply_random_bottle_dropout(img: np.ndarray, boxes: list, drop_count: int):
    """
    Modelin 'her zaman 83 şişe yan yana dizilidir' ezberini bozmak için
    rastgele seçilen şişeleri arka plan dokusuyla kapatır ve etiket 리스트esinden siler.
    """
    if drop_count <= 0 or drop_count >= len(boxes) - 10:
        return img.copy(), list(boxes)

    out = img.copy()
    h, w = out.shape[:2]
    drop_indices = set(random.sample(range(len(boxes)), drop_count))
    kept_boxes = []

    for idx, (cx, cy, bw, bh) in enumerate(boxes):
        if idx not in drop_indices:
            kept_boxes.append((cx, cy, bw, bh))
            continue

        # Şişe bölgesini yumuşak arka plan (koli/naylon) dokusuyla doldur
        x1 = max(0, int(cx - bw * 0.52))
        y1 = max(0, int(cy - bh * 0.52))
        x2 = min(w, int(cx + bw * 0.52))
        y2 = min(h, int(cy + bh * 0.52))

        # Çevre piksellerin medyan rengini alıp hafif gürültü ve bulanıklıkla kapat
        ring_pad = 6
        rx1, ry1 = max(0, x1 - ring_pad), max(0, y1 - ring_pad)
        rx2, ry2 = min(w, x2 + ring_pad), min(h, y2 + ring_pad)
        med_color = np.median(out[ry1:ry2, rx1:rx2].reshape(-1, 3), axis=0)

        noise = np.random.normal(0, 6, (y2 - y1, x2 - x1, 3))
        patch = np.clip(med_color + noise, 0, 255).astype(np.uint8)
        out[y1:y2, x1:x2] = cv2.GaussianBlur(patch, (9, 9), 3)

    return out, kept_boxes


# ==============================================================================
# MADDE 3: ÇOKLU ÖLÇEK (YAKIN/UZAK KAMERA) VE PERSPEKTİF / DÖNDÜRME
# ==============================================================================
def rotate_image_and_boxes(img: np.ndarray, boxes: list, angle_deg: float, scale: float = 1.0):
    """Görseli ve dairesel şişe ağzı kutularını belirtilen açı ve ölçekle döndürür."""
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)

    rotated_img = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(114, 114, 114)
    )

    rad = np.deg2rad(angle_deg)
    cos2 = np.cos(rad) ** 2
    sin2 = np.sin(rad) ** 2

    new_boxes = []
    for (cx, cy, bw, bh) in boxes:
        nx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
        ny = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]
        nbw = (bw * cos2 + bh * sin2) * scale
        nbh = (bw * sin2 + bh * cos2) * scale

        margin = min(nbw, nbh) * 0.35
        if margin <= nx <= (w - margin) and margin <= ny <= (h - margin):
            x1 = np.clip(nx - nbw / 2.0, 0, w)
            y1 = np.clip(ny - nbh / 2.0, 0, h)
            x2 = np.clip(nx + nbw / 2.0, 0, w)
            y2 = np.clip(ny + nbh / 2.0, 0, h)
            if (x2 - x1) >= 10 and (y2 - y1) >= 10:
                new_boxes.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1))

    return rotated_img, new_boxes


def flip_image_and_boxes(img: np.ndarray, boxes: list, flip_code: int):
    h, w = img.shape[:2]
    flipped = cv2.flip(img, flip_code)
    new_boxes = []
    for (cx, cy, bw, bh) in boxes:
        nx = (w - cx) if flip_code in (1, -1) else cx
        ny = (h - cy) if flip_code in (0, -1) else cy
        new_boxes.append((nx, ny, bw, bh))
    return flipped, new_boxes


def random_crop_and_boxes(img: np.ndarray, boxes: list, crop_ratio: float):
    """Farklı kamera yakınlıklarını (büyük şişe / az şişe) öğretmek için rastgele kesit alır."""
    h, w = img.shape[:2]
    ch, cw = int(h * crop_ratio), int(w * crop_ratio)
    y0 = random.randint(0, max(0, h - ch))
    x0 = random.randint(0, max(0, w - cw))

    cropped = img[y0:y0 + ch, x0:x0 + cw].copy()
    new_boxes = []
    for (cx, cy, bw, bh) in boxes:
        nx, ny = cx - x0, cy - y0
        if (bw * 0.35) <= nx <= (cw - bw * 0.35) and (bh * 0.35) <= ny <= (ch - bh * 0.35):
            x1 = np.clip(nx - bw / 2.0, 0, cw)
            y1 = np.clip(ny - bh / 2.0, 0, ch)
            x2 = np.clip(nx + bw / 2.0, 0, cw)
            y2 = np.clip(ny + bh / 2.0, 0, ch)
            new_boxes.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1))

    return cropped, new_boxes


# ==============================================================================
# MADDE 4: ŞEFFAF PLASTİK İÇİN KONTRAST GÜÇLENDİRME (CLAHE) VE IŞIK ÇEŞİTLEMESİ
# ==============================================================================
def enhance_transparent_contrast(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Şeffaf şişe ağızlarındaki dairesel ışık yansımalarını belirginleştirir."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def adjust_photometrics(img: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def save_yolo_sample(img: np.ndarray, boxes: list, img_path: Path, label_path: Path):
    h, w = img.shape[:2]
    imwrite_unicode(img_path, img)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for (cx, cy, bw, bh) in boxes:
        ncx = np.clip(cx / w, 0.0, 1.0)
        ncy = np.clip(cy / h, 0.0, 1.0)
        nbw = np.clip(bw / w, 0.0, 1.0)
        nbh = np.clip(bh / h, 0.0, 1.0)
        lines.append(f"0 {ncx:.6f} {ncy:.6f} {nbw:.6f} {nbh:.6f}")
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dataset():
    print(f"[1/5] Kaynak görsel okunuyor: {SOURCE_IMAGE}")
    img_orig = imread_unicode(SOURCE_IMAGE)
    boxes_orig = detect_initial_bottles(img_orig)
    print(f"[2/5] Otomatik etiketleme tamamlandı: {len(boxes_orig)} adet şişe tespit edildi.")

    # Tamamen şeffaf şişe versiyonunu üret ve kaydet (kontrol için)
    img_full_transparent = convert_bottles_to_transparent(img_orig, boxes_orig, transparent_ratio=1.0)
    img_mixed_transparent = convert_bottles_to_transparent(img_orig, boxes_orig, transparent_ratio=0.55)
    imwrite_unicode(BASE_DIR / "outputs" / "ornek_seffaf_siseler.jpg", img_full_transparent)

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    for split in ["train", "val"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("[3/5] 4 aşamalı ezber bozucu veri seti üretiliyor (Şeffaf + Kırmızı + Boşluklu + Çoklu Ölçek)...")
    samples = []

    # GRUP A: Tamamen Şeffaf Şişeler (%0 Kırmızı) - Farklı Açılar ve Ölçekler (60 görsel)
    for idx, angle in enumerate(range(0, 360, 6)):
        base_t = convert_bottles_to_transparent(img_orig, boxes_orig, transparent_ratio=1.0)
        if idx % 2 == 0:
            base_t = enhance_transparent_contrast(base_t, clip_limit=random.uniform(1.6, 2.4))
        scale = random.uniform(0.78, 1.08)
        r_img, r_boxes = rotate_image_and_boxes(base_t, boxes_orig, float(angle), scale=scale)
        r_img = adjust_photometrics(r_img, alpha=random.uniform(0.88, 1.15), beta=random.uniform(-12, 12))
        is_val = (angle % 36 == 12)
        samples.append((f"trans_rot_{angle:03d}", r_img, r_boxes, is_val))

    # GRUP B: Yarı Şeffaf + Yarı Kırmızı Karışık Koliler + Rastgele Eksik Şişeler (45 görsel)
    for idx, angle in enumerate(range(0, 360, 8)):
        ratio = random.uniform(0.35, 0.75)
        mixed_img = convert_bottles_to_transparent(img_orig, boxes_orig, transparent_ratio=ratio)
        # Bazılarında rastgele 5-22 şişeyi silerek dizilim ezberini kır
        drop_n = random.randint(6, 22) if idx % 2 == 1 else 0
        m_img, m_boxes = apply_random_bottle_dropout(mixed_img, boxes_orig, drop_count=drop_n)
        if idx % 3 == 0:
            m_img, m_boxes = flip_image_and_boxes(m_img, m_boxes, random.choice([0, 1]))
        scale = random.uniform(0.80, 1.06)
        r_img, r_boxes = rotate_image_and_boxes(m_img, m_boxes, float(angle), scale=scale)
        is_val = (angle % 48 == 16)
        samples.append((f"mixed_drop_{idx:02d}_rot_{angle:03d}", r_img, r_boxes, is_val))

    # GRUP C: Orijinal Kırmızı ve Farklı Renk/Gri Tonlamalı Döndürmeler (45 görsel)
    for idx, angle in enumerate(range(0, 360, 8)):
        if idx % 3 == 0:
            # Tamamen orijinal kırmızı halkalı
            c_img = img_orig.copy()
        elif idx % 3 == 1:
            # Farklı kapak/halka rengi (mavi/yeşil/sarı)
            c_img = shift_bottle_colors(img_orig, hue_shift=random.choice([60, 90, 120]), desaturate=random.uniform(0.5, 1.0))
        else:
            # Gri tonlamaya yakın (renksiz/nötr)
            c_img = shift_bottle_colors(img_orig, hue_shift=0, desaturate=random.uniform(0.0, 0.25))

        scale = 1.0 if angle % 90 == 0 else random.uniform(0.82, 1.06)
        r_img, r_boxes = rotate_image_and_boxes(c_img, boxes_orig, float(angle), scale=scale)
        is_val = (angle % 48 == 24)
        samples.append((f"color_{idx:02d}_rot_{angle:03d}", r_img, r_boxes, is_val))

    # GRUP D: Yakın Çekim (Close-up / Büyük Şişeler) ve Farklı Mesafe Kesitleri (40 görsel)
    for i in range(40):
        mode = i % 3
        if mode == 0:
            src_img = convert_bottles_to_transparent(img_orig, boxes_orig, transparent_ratio=1.0)
            src_img = enhance_transparent_contrast(src_img, clip_limit=2.0)
        elif mode == 1:
            src_img = convert_bottles_to_transparent(img_orig, boxes_orig, transparent_ratio=0.5)
        else:
            src_img = img_orig.copy()

        angle = random.choice([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
        r_img, r_boxes = rotate_image_and_boxes(src_img, boxes_orig, float(angle), scale=1.0)
        # 0.32 ile 0.78 arası kesit (yakından 4-15 şişe veya orta mesafe 25-50 şişe)
        c_img, c_boxes = random_crop_and_boxes(r_img, r_boxes, crop_ratio=random.uniform(0.32, 0.78))
        if len(c_boxes) >= 3:
            c_img = adjust_photometrics(c_img, alpha=random.uniform(0.9, 1.12), beta=random.uniform(-10, 10))
            samples.append((f"zoom_{i:02d}_rot_{angle:03d}", c_img, c_boxes, i % 8 == 0))

    train_count, val_count = 0, 0
    for name, s_img, s_boxes, is_val in samples:
        split = "val" if is_val else "train"
        img_out = DATASET_DIR / "images" / split / f"{name}.jpg"
        lbl_out = DATASET_DIR / "labels" / split / f"{name}.txt"
        save_yolo_sample(s_img, s_boxes, img_out, lbl_out)
        if is_val:
            val_count += 1
        else:
            train_count += 1

    # Doğrulama setine hem orijinal 83 şişeyi hem de tamamen şeffaf 83 şişeyi ekle
    save_yolo_sample(
        img_orig, boxes_orig,
        DATASET_DIR / "images" / "val" / "val_original_83.jpg",
        DATASET_DIR / "labels" / "val" / "val_original_83.txt"
    )
    save_yolo_sample(
        img_full_transparent, boxes_orig,
        DATASET_DIR / "images" / "val" / "val_transparent_83.jpg",
        DATASET_DIR / "labels" / "val" / "val_transparent_83.txt"
    )
    val_count += 2

    yaml_content = (
        f"path: {DATASET_DIR.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: sise\n"
    )
    (DATASET_DIR / "data.yaml").write_text(yaml_content, encoding="utf-8")

    # 4 aşamayı gösteren görsel kolaj hazırla
    t1 = img_orig.copy()
    cv2.putText(t1, "1. Orijinal (Kirmizi Halkali)", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

    t2 = img_full_transparent.copy()
    cv2.putText(t2, "2. Yapay Seffaf Sise (Kirmizisiz)", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

    t3, b3 = apply_random_bottle_dropout(img_mixed_transparent, boxes_orig, drop_count=18)
    for (cx, cy, bw, bh) in b3:
        cv2.rectangle(t3, (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2)), (0, 255, 0), 2)
    cv2.putText(t3, f"3. Karisik + Bosluklu ({len(b3)} Sise)", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

    t4_crop, b4 = random_crop_and_boxes(img_full_transparent, boxes_orig, crop_ratio=0.45)
    for (cx, cy, bw, bh) in b4:
        cv2.rectangle(t4_crop, (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2)), (0, 255, 0), 2)
    t4 = cv2.resize(t4_crop, (img_orig.shape[1], img_orig.shape[0]))
    cv2.putText(t4, f"4. Yakin Cekim Seffaf ({len(b4)} Sise)", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

    tiles = [cv2.resize(x, (420, 450)) for x in [t1, t2, t3, t4]]
    collage = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    collage_path = BASE_DIR / "outputs" / "veri_cogaltma_ornekleri.jpg"
    imwrite_unicode(collage_path, collage)

    print(f"[4/5] Görsel kolaj kaydedildi: {collage_path}")
    print(f"[5/5] Yeni veri seti hazır!")
    print(f"      - Eğitim (Train) görsel sayısı : {train_count}")
    print(f"      - Doğrulama (Val) görsel sayısı: {val_count}")
    print(f"      - Toplam veri seti boyutu      : {train_count + val_count} görsel")


if __name__ == "__main__":
    build_dataset()
