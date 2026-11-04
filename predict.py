import argparse
from pathlib import Path
import cv2
import numpy as np

# Windows Türkçe karakterli (Masaüstü vb.) yollar için OpenCV imread/imwrite yaması
_orig_imread = cv2.imread
_orig_imwrite = cv2.imwrite


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return _orig_imread(str(path), flags)


def imwrite_unicode(path, img, params=None):
    try:
        p = Path(str(path))
        p.parent.mkdir(parents=True, exist_ok=True)
        ext = p.suffix.lower() or ".jpg"
        ok, buf = cv2.imencode(ext, img, params or [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            buf.tofile(str(p))
            return True
        return False
    except Exception:
        return _orig_imwrite(str(path), img, params)


cv2.imread = imread_unicode
cv2.imwrite = imwrite_unicode

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "bottle_yolov8_best.pt"
DEFAULT_IMAGE_PATH = BASE_DIR / "original_bottles.jpg"
OUTPUTS_DIR = BASE_DIR / "outputs"


def load_trained_model(model_path: Path = DEFAULT_MODEL_PATH) -> YOLO:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Eğitilmiş model bulunamadı: {model_path}\n"
            f"Lütfen önce 'python train.py' komutunu çalıştırın."
        )
    import os
    os.chdir(BASE_DIR)
    try:
        rel_path = model_path.resolve().relative_to(BASE_DIR.resolve())
        return YOLO(rel_path.as_posix())
    except ValueError:
        return YOLO(str(model_path))


FIXED_CONF_THRESH = 0.35
FIXED_IOU_THRESH = 0.45


def count_bottles(
    model: YOLO,
    img_bgr: np.ndarray,
    conf_thresh: float = FIXED_CONF_THRESH,
    iou_thresh: float = FIXED_IOU_THRESH,
    draw_numbers: bool = False,
    draw_boxes: bool = True,
    draw_centers: bool = False
):
    """
    Görsel üzerindeki şişeleri eğitilmiş YOLOv8 CNN modeliyle tespit eder ve sayar.
    Sadece çerçeveleri (bounding box) ve toplam şişe sayısını gösterir.
    Döndürür: (annotated_bgr, detections_list, stats_dict)
    """
    h, w = img_bgr.shape[:2]
    results = model.predict(
        source=img_bgr,
        conf=conf_thresh,
        iou=iou_thresh,
        imgsz=640,
        max_det=300,
        verbose=False
    )

    detections = []
    if len(results) > 0 and results[0].boxes is not None:
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), conf in zip(boxes_xyxy, confs):
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            detections.append({
                "x1": int(round(x1)),
                "y1": int(round(y1)),
                "x2": int(round(x2)),
                "y2": int(round(y2)),
                "cx": float(cx),
                "cy": float(cy),
                "conf": float(conf)
            })

    # Yukarıdan aşağıya, soldan sağa sırala
    detections.sort(key=lambda d: (int(d["cy"] // 50), d["cx"]))

    vis = img_bgr.copy()
    for idx, det in enumerate(detections, 1):
        det["id"] = idx
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        cx, cy = int(det["cx"]), int(det["cy"])

        if draw_boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 230, 64), 2)

        if draw_centers:
            cv2.circle(vis, (cx, cy), 4, (0, 255, 255), -1)

        if draw_numbers:
            label = str(idx)
            cv2.putText(vis, label, (x1 + 3, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, label, (x1 + 3, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)

    total_count = len(detections)

    # Üst bilgi şeridi (Sadece Toplam Şişe Sayısı)
    banner_h = max(48, int(h * 0.065))
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 25, 35), -1)
    cv2.addWeighted(overlay, 0.82, vis, 0.18, 0, vis)

    banner_text = f"Toplam Sise Sayisi: {total_count}"
    cv2.putText(
        vis, banner_text,
        (18, int(banner_h * 0.68)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 255, 180), 2, cv2.LINE_AA
    )

    stats = {
        "count": total_count,
    }
    return vis, detections, stats


def main():
    parser = argparse.ArgumentParser(description="Eğitilmiş YOLOv8 CNN modeli ile şişe sayımı")
    parser.add_argument("--image", type=str, default=str(DEFAULT_IMAGE_PATH), help="Test edilecek görsel yolu")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help="Eğitilmiş model (.pt) yolu")
    args = parser.parse_args()

    model = load_trained_model(Path(args.model))
    img = imread_unicode(Path(args.image))
    if img is None:
        raise FileNotFoundError(f"Görsel okunamadı: {args.image}")

    vis, detections, stats = count_bottles(model, img)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUTS_DIR / "sonuc_orijinal_sayim.jpg"
    imwrite_unicode(out_file, vis)

    print("============================================================")
    print(f"CNN ŞİŞE SAYIM SONUCU")
    print(f"Görsel            : {args.image}")
    print(f"Toplam Şişe Sayısı: {stats['count']} adet")
    print(f"Kaydedilen Sonuç  : {out_file}")
    print("============================================================")

    # Ayrıca 90, 180 ve 270 derece döndürülmüş görsellerde test kolajı üret
    test_tiles = []
    for angle, rot_code in [(0, None), (90, cv2.ROTATE_90_CLOCKWISE), (180, cv2.ROTATE_180), (270, cv2.ROTATE_90_COUNTERCLOCKWISE)]:
        r_img = img.copy() if rot_code is None else cv2.rotate(img, rot_code)
        r_vis, _, r_stats = count_bottles(model, r_img)
        tile = cv2.resize(r_vis, (420, 450))
        cv2.putText(
            tile, f"Dondurme: {angle} deg -> Sayim: {r_stats['count']}",
            (14, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA
        )
        test_tiles.append(tile)

    grid = np.vstack([np.hstack(test_tiles[:2]), np.hstack(test_tiles[2:])])
    rot_test_out = OUTPUTS_DIR / "sonuc_dondurulmus_testler.jpg"
    imwrite_unicode(rot_test_out, grid)
    print(f"Döndürülmüş test kolajı kaydedildi: {rot_test_out}")


if __name__ == "__main__":
    main()
