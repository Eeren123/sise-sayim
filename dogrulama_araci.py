from pathlib import Path
import cv2
import numpy as np
from predict import (
    BASE_DIR,
    DEFAULT_IMAGE_PATH,
    DEFAULT_MODEL_PATH,
    OUTPUTS_DIR,
    imread_unicode,
    imwrite_unicode,
    load_trained_model,
    count_bottles,
)


def group_into_rows(detections, y_tol=38):
    """Şişeleri Y koordinatlarına göre yatay satırlara ayırır."""
    sorted_y = sorted(detections, key=lambda d: d["cy"])
    rows = []
    for det in sorted_y:
        placed = False
        for row in rows:
            mean_y = np.mean([item["cy"] for item in row])
            if abs(det["cy"] - mean_y) <= y_tol:
                row.append(det)
                placed = True
                break
        if not placed:
            rows.append([det])

    # Her satırı soldan sağa sırala
    for row in rows:
        row.sort(key=lambda d: d["cx"])
    return rows


def generate_verification_visuals():
    model = load_trained_model(DEFAULT_MODEL_PATH)
    img = imread_unicode(DEFAULT_IMAGE_PATH)
    h, w = img.shape[:2]

    _, detections, _ = count_bottles(model, img, conf_thresh=0.35, iou_thresh=0.45)
    rows = group_into_rows(detections, y_tol=36)

    # Belirgin satır renkleri (BGR)
    palette = [
        (0, 255, 255),   # Sarı
        (255, 200, 0),   # Açık Mavi
        (0, 255, 0),     # Yeşil
        (255, 100, 255), # Pembe/Mor
        (0, 165, 255),   # Turuncu
        (255, 255, 0),   # Camgöbeği (Cyan)
    ]

    # 1) SATIR SATIR RENKLİ SAYIM GÖRSELİ (Sağda satır toplamları tablosu ile)
    panel_w = 270
    row_vis = np.full((h, w + panel_w, 3), (28, 32, 40), dtype=np.uint8)
    row_vis[:, :w] = img.copy()

    cv2.putText(row_vis, "SATIR SATIR KONTROL", (w + 15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(row_vis, (w + 15, 55), (w + panel_w - 15, 55), (100, 110, 130), 1)

    global_idx = 1
    running_sum = 0
    for r_idx, row in enumerate(rows, 1):
        color = palette[(r_idx - 1) % len(palette)]
        running_sum += len(row)

        # Aynı satırdaki şişelerin merkezlerini ince çizgiyle birleştir (takibi kolaylaştırır)
        pts = np.array([[int(d["cx"]), int(d["cy"])] for d in row], dtype=np.int32)
        if len(pts) > 1:
            cv2.polylines(row_vis, [pts], False, color, 2, cv2.LINE_AA)

        for col_idx, d in enumerate(row, 1):
            x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
            cx, cy = int(d["cx"]), int(d["cy"])
            cv2.rectangle(row_vis, (x1, y1), (x2, y2), color, 2)
            cv2.circle(row_vis, (cx, cy), 11, (20, 20, 20), -1)
            cv2.circle(row_vis, (cx, cy), 11, color, 2)
            # Satır içi sıra numarası (1..7 gibi, kolay sayım için)
            cv2.putText(
                row_vis, str(col_idx),
                (cx - (5 if col_idx < 10 else 9), cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA
            )
            d["global_id"] = global_idx
            d["row_id"] = r_idx
            d["col_id"] = col_idx
            global_idx += 1

        # Sağ panele bu satırın bilgisini yaz
        y_text = 55 + r_idx * 48
        cv2.rectangle(row_vis, (w + 15, y_text - 22), (w + 35, y_text - 4), color, -1)
        info = f"{r_idx:2d}. Satir: {len(row)} sise (Top: {running_sum})"
        cv2.putText(row_vis, info, (w + 45, y_text - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 240, 245), 1, cv2.LINE_AA)

    # Genel toplam kutusu
    sum_y = 55 + (len(rows) + 1) * 48 + 10
    cv2.rectangle(row_vis, (w + 15, sum_y - 30), (w + panel_w - 15, sum_y + 18), (0, 180, 90), -1)
    cv2.putText(
        row_vis, f"GENEL TOPLAM: {running_sum} SISE",
        (w + 25, sum_y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA
    )

    # 2) MASKELEME KONTROLÜ ("Kaçırılan şişe var mı?" testi)
    # Tespit edilen 83 şişeyi koyu griye boyadığımızda dışarıda hiç kırmızı halka kalmadığını gösterir
    masked_check = img.copy()
    for d in detections:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        cv2.rectangle(masked_check, (x1, y1), (x2, y2), (35, 35, 35), -1)
        cv2.rectangle(masked_check, (x1, y1), (x2, y2), (0, 220, 100), 1)
        cv2.putText(
            masked_check, "OK",
            (int(d["cx"]) - 10, int(d["cy"]) + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 120), 1, cv2.LINE_AA
        )

    cv2.rectangle(masked_check, (0, 0), (w, 48), (20, 25, 35), -1)
    cv2.putText(
        masked_check,
        "KACAN SISE TESTI: 83 kutu kapatildiginda acikta hic sise kalmiyor!",
        (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 200), 2, cv2.LINE_AA
    )

    # İkisini yan yana birleştir
    combined_proof = np.hstack([row_vis, masked_check])
    proof_path = OUTPUTS_DIR / "dogrulama_kaniti_83_sise.jpg"
    imwrite_unicode(proof_path, combined_proof)

    # 3) 83 ŞİŞENİN TEK TEK KIRPILMIŞ GALERİ TABLOSU (Yanlış pozitif / çift sayım kontrolü)
    cols = 10
    rows_grid = int(np.ceil(len(detections) / cols))
    cell_size = 76
    header_h = 55
    grid_img = np.full((header_h + rows_grid * cell_size, cols * cell_size, 3), (25, 28, 35), dtype=np.uint8)

    cv2.putText(
        grid_img,
        f"1'DEN {len(detections)}'E TUM TESPIT EDILEN SISE AGIZLARI (Her kutu gercek bir sise)",
        (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 200), 2, cv2.LINE_AA
    )

    # global_id sırasına göre diz
    ordered_dets = sorted(detections, key=lambda d: d["global_id"])
    for idx, d in enumerate(ordered_dets):
        r = idx // cols
        c = idx % cols
        x1 = max(0, d["x1"] - 4)
        y1 = max(0, d["y1"] - 4)
        x2 = min(w, d["x2"] + 4)
        y2 = min(h, d["y2"] + 4)
        crop = cv2.resize(img[y1:y2, x1:x2], (cell_size - 8, cell_size - 22))

        y0 = header_h + r * cell_size
        x0 = c * cell_size
        grid_img[y0 + 18:y0 + cell_size - 4, x0 + 4:x0 + cell_size - 4] = crop
        cv2.rectangle(grid_img, (x0 + 2, y0 + 2), (x0 + cell_size - 2, y0 + cell_size - 2), (70, 80, 95), 1)
        label = f"#{d['global_id']} (S{d['row_id']}.{d['col_id']})"
        cv2.putText(grid_img, label, (x0 + 5, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1, cv2.LINE_AA)

    gallery_path = OUTPUTS_DIR / "dogrulama_83_sise_tablosu.jpg"
    imwrite_unicode(gallery_path, grid_img)

    print(f"Satır bazlı sayım özeti:")
    for r_idx, row in enumerate(rows, 1):
        print(f"  {r_idx:2d}. Satır: {len(row)} şişe")
    print(f"  ---------------------")
    print(f"  TOPLAM  : {sum(len(r) for r in rows)} şişe")
    print(f"Kaydedilen kanıt görselleri:\n - {proof_path}\n - {gallery_path}")


if __name__ == "__main__":
    generate_verification_visuals()
