import argparse
import cv2
from predict import (
    DEFAULT_MODEL_PATH,
    FIXED_CONF_THRESH,
    FIXED_IOU_THRESH,
    count_bottles,
    load_trained_model,
)


def run_live_camera(camera_id: int = 0):
    model = load_trained_model(DEFAULT_MODEL_PATH)
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"[HATA] Kamera ({camera_id}) açılamadı! Lütfen kamera bağlantısını kontrol edin.")
        return

    print("============================================================")
    print("CANLI KAMERA ŞİŞE SAYIM TESTİ BAŞLATILDI")
    print("Çıkmak için kamera penceresindeyken 'q' veya 'ESC' tuşuna basın.")
    print("============================================================")

    window_name = "Canli Kamera - CNN Sise Sayimi (Cikis: Q / ESC)"
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        vis_frame, _, _ = count_bottles(
            model,
            frame,
            conf_thresh=FIXED_CONF_THRESH,
            iou_thresh=FIXED_IOU_THRESH,
            draw_numbers=False,
            draw_boxes=True,
            draw_centers=False,
        )

        cv2.imshow(window_name, vis_frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canlı Kamera ile Gerçek Zamanlı Şişe Sayımı")
    parser.add_argument("--cam", type=int, default=0, help="Kamera indeksi (varsayılan: 0)")
    args = parser.parse_args()
    run_live_camera(args.cam)
