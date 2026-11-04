from pathlib import Path
import time
import cv2
import numpy as np
import streamlit as st

from predict import (
    DEFAULT_IMAGE_PATH,
    DEFAULT_MODEL_PATH,
    FIXED_CONF_THRESH,
    FIXED_IOU_THRESH,
    OUTPUTS_DIR,
    count_bottles,
    imread_unicode,
    load_trained_model,
)

st.set_page_config(
    page_title="Şişe Sayım Sistemi",
    page_icon="🍾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Telefon, Tablet ve Masaüstü için Responsive (Duyarlı) CSS Tasarımı
st.markdown(
    """
    <style>
    /* Genel Konteyner ve Mobil Uyum */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1100px !important;
    }

    /* Başlık Kartı */
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        padding: 1.1rem 1.4rem;
        border-radius: 16px;
        border: 1px solid rgba(56, 189, 248, 0.25);
        margin-bottom: 1rem;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.18);
    }
    .app-header h1 {
        font-size: 1.55rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
        color: #38bdf8 !important;
    }
    .app-header p {
        font-size: 0.92rem !important;
        color: #cbd5e1 !important;
        margin: 0.35rem 0 0 0 !important;
    }

    /* Büyük Şişe Sayısı Sonuç Kartı */
    .count-card {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        color: #ffffff;
        padding: 1.15rem 1.2rem;
        border-radius: 16px;
        text-align: center;
        margin: 0.75rem 0 1.1rem 0;
        border: 2px solid #34d399;
        box-shadow: 0 10px 25px rgba(5, 150, 105, 0.28);
    }
    .count-card .count-label {
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #a7f3d0;
        margin-bottom: 0.2rem;
    }
    .count-card .count-value {
        font-size: 2.6rem;
        font-weight: 900;
        line-height: 1.1;
        color: #ffffff;
    }

    /* Dokunmatik Dostu Butonlar */
    .stButton > button, .stDownloadButton > button {
        width: 100% !important;
        min-height: 3.1rem !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }

    /* Sekme (Tab) Başlıkları */
    button[data-baseweb="tab"] {
        font-size: 0.98rem !important;
        font-weight: 700 !important;
        padding: 0.65rem 1rem !important;
    }

    /* Mobil Ekranlar İçin Özel Kurallar (<= 768px) */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-top: 0.8rem !important;
        }
        .app-header {
            padding: 0.85rem 0.9rem;
            border-radius: 12px;
        }
        .app-header h1 {
            font-size: 1.22rem !important;
        }
        .app-header p {
            font-size: 0.82rem !important;
        }
        .count-card {
            padding: 0.9rem 0.8rem;
            border-radius: 14px;
        }
        .count-card .count-value {
            font-size: 2.15rem;
        }
        /* Mobilde Radio Seçeneklerini Alt Alta ve Dokunmatik Kart Gibi Göster */
        div[role="radiogroup"] {
            gap: 0.45rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model(model_path_str: str):
    return load_trained_model(Path(model_path_str))


def render_count_card(count: int):
    st.markdown(
        f"""
        <div class="count-card">
            <div class="count-label">🍾 Tespit Edilen Toplam Şişe Sayısı</div>
            <div class="count-value">{count} Adet</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def rotate_bgr_image(img_bgr: np.ndarray, angle_deg: int) -> np.ndarray:
    if angle_deg == 0:
        return img_bgr
    if angle_deg == 90:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    if angle_deg == 180:
        return cv2.rotate(img_bgr, cv2.ROTATE_180)
    if angle_deg == 270:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle_deg), 1.0)
    return cv2.warpAffine(
        img_bgr, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(114, 114, 114),
    )


def process_and_display(model, img_bgr: np.ndarray, show_input: bool = False):
    vis_bgr, _, stats = count_bottles(
        model,
        img_bgr,
        conf_thresh=FIXED_CONF_THRESH,
        iou_thresh=FIXED_IOU_THRESH,
        draw_numbers=False,
        draw_boxes=True,
        draw_centers=False,
    )

    render_count_card(stats["count"])

    if show_input:
        c1, c2 = st.columns(2)
        with c1:
            st.caption("📷 Giriş Görseli")
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        with c2:
            st.caption(f"✅ Tespit Sonucu ({stats['count']} Şişe)")
            st.image(cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    else:
        st.image(cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    ok, buf = cv2.imencode(".jpg", vis_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if ok:
        st.download_button(
            label="📥 Sonuç Görselini Telefon / Bilgisayara İndir",
            data=buf.tobytes(),
            file_name=f"sise_sayim_{stats['count']}_adet.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )


def main():
    st.markdown(
        """
        <div class="app-header">
            <h1>🍾 Otomatik Şişe Sayım Sistemi (CNN)</h1>
            <p>Kameradan fotoğraf çekin veya görsel yükleyin — sistem şişeleri çerçeve içine alıp anında sayar.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not DEFAULT_MODEL_PATH.exists():
        st.warning(
            f"Eğitilmiş model henüz `{DEFAULT_MODEL_PATH}` konumunda bulunamadı. "
            "Lütfen `python train.py` komutunu çalıştırın."
        )
        return

    model = get_model(str(DEFAULT_MODEL_PATH))

    tab_test, tab_verify = st.tabs([
        "🔍 Canlı Şişe Testi",
        "✅ 83 Şişe Doğrulama",
    ])

    with tab_test:
        test_mode = st.radio(
            "Kullanım Yöntemi Seçin:",
            [
                "📱 Telefon Kamerası / Fotoğraf Yükle (Mobil İçin Önerilen)",
                "📸 Tarayıcı Kamerası ile Anlık Çekim",
                "🖼️ Örnek Koli Görselleri ile Test",
                "🎥 Bilgisayar Kamerası (Canlı Video Akışı)",
            ],
            horizontal=False,
        )

        # ------------------------------------------------------------------
        # 1) TELEFON KAMERASI VEYA GALERİDEN FOTOĞRAF YÜKLE (MOBİL UYUMLU)
        # ------------------------------------------------------------------
        if test_mode == "📱 Telefon Kamerası / Fotoğraf Yükle (Mobil İçin Önerilen)":
            st.info(
                "💡 **Telefondan kullanırken:** Aşağıdaki butona dokunduğunuzda telefonunuzun **Arka Kamerasını (Fotoğraf Çek)** "
                "veya **Galerisini** seçerek kutudaki şişeleri yüksek çözünürlükte çekip anında saydırabilirsiniz."
            )
            uploaded_file = st.file_uploader(
                "📷 Fotoğraf Çekmek veya Seçmek İçin Dokunun",
                type=["jpg", "jpeg", "png", "webp"],
            )
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    process_and_display(model, img_bgr, show_input=False)

        # ------------------------------------------------------------------
        # 2) TARAYICI İÇİ CANLI KAMERA VİZÖRÜ (st.camera_input)
        # ------------------------------------------------------------------
        elif test_mode == "📸 Tarayıcı Kamerası ile Anlık Çekim":
            st.caption("Kamerayı kutunun üzerine tutun ve **'Take Photo'** butonuna dokunun.")
            cam_photo = st.camera_input("Canlı Kamera Vizörü")
            if cam_photo is not None:
                file_bytes = np.asarray(bytearray(cam_photo.read()), dtype=np.uint8)
                frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if frame_bgr is not None:
                    process_and_display(model, frame_bgr, show_input=False)

        # ------------------------------------------------------------------
        # 3) ÖRNEK KOLİ GÖRSELLERİ İLE TEST (ORİJİNAL & ŞEFFAF)
        # ------------------------------------------------------------------
        elif test_mode == "🖼️ Örnek Koli Görselleri ile Test":
            col_s1, col_s2 = st.columns([1, 1])
            with col_s1:
                sample_choice = st.selectbox(
                    "Örnek Görsel Seçin:",
                    [
                        "Şeffaf Şişeler (Kırmızısız Test - 83 Şişe)",
                        "Orijinal Görsel (Kırmızı Halkalı - 83 Şişe)",
                    ],
                )
            with col_s2:
                test_angle = st.select_slider(
                    "Görseli Döndür (°):",
                    options=list(range(0, 360, 15)),
                    value=0,
                )

            if "Şeffaf" in sample_choice:
                trans_path = OUTPUTS_DIR / "ornek_seffaf_siseler.jpg"
                img_bgr = imread_unicode(trans_path if trans_path.exists() else DEFAULT_IMAGE_PATH)
            else:
                img_bgr = imread_unicode(DEFAULT_IMAGE_PATH)

            if img_bgr is not None:
                rotated_input = rotate_bgr_image(img_bgr, test_angle)
                process_and_display(model, rotated_input, show_input=True)

        # ------------------------------------------------------------------
        # 4) BİLGİSAYAR KAMERASI (LOCAL OPENCV CANLI AKIŞ)
        # ------------------------------------------------------------------
        elif test_mode == "🎥 Bilgisayar Kamerası (Canlı Video Akışı)":
            st.caption("Not: Bu mod uygulamanın çalıştığı bilgisayara bağlı yerel USB/Webcam kamerasını kullanır.")
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                start_cam = st.button("▶️ Canlı Kamerayı Başlat", type="primary", use_container_width=True)
            with col_b2:
                stop_cam = st.button("⏹️ Kamerayı Durdur", use_container_width=True)

            if "cam_running" not in st.session_state:
                st.session_state.cam_running = False
            if start_cam:
                st.session_state.cam_running = True
            if stop_cam:
                st.session_state.cam_running = False

            count_placeholder = st.empty()
            frame_placeholder = st.empty()

            if st.session_state.cam_running:
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(0)

                if not cap.isOpened():
                    st.error("Yerel kamera açılamadı! Telefondan bağlanıyorsanız lütfen yukarıdaki **'📱 Telefon Kamerası'** veya **'📸 Tarayıcı Kamerası'** seçeneğini kullanın.")
                    st.session_state.cam_running = False
                else:
                    try:
                        while st.session_state.cam_running:
                            ret, frame = cap.read()
                            if not ret or frame is None:
                                break

                            vis_frame, _, stats = count_bottles(
                                model,
                                frame,
                                conf_thresh=FIXED_CONF_THRESH,
                                iou_thresh=FIXED_IOU_THRESH,
                                draw_numbers=False,
                                draw_boxes=True,
                                draw_centers=False,
                            )

                            count_placeholder.markdown(
                                f"""
                                <div class="count-card">
                                    <div class="count-label">🍾 Canlı Toplam Şişe Sayısı</div>
                                    <div class="count-value">{stats['count']} Adet</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            frame_placeholder.image(
                                cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB),
                                channels="RGB",
                                use_container_width=True,
                            )
                            time.sleep(0.02)
                    finally:
                        cap.release()

    with tab_verify:
        st.subheader("Görselde Gerçekten 83 Şişe Olduğunun Kanıtı")
        st.markdown(
            """
            - **Satır Satır Renkli Sayım:** 11 yatay satır (`7 + 7 + 7 + 6 + 8 + 9 + 9 + 8 + 7 + 8 + 7 = 83 Şişe`).
            - **Kaçan Şişe Testi:** 83 kutu kapatıldığında açıkta hiçbir şişe ağzı kalmamaktadır.
            """
        )
        proof_img = OUTPUTS_DIR / "dogrulama_kaniti_83_sise.jpg"
        gallery_img = OUTPUTS_DIR / "dogrulama_83_sise_tablosu.jpg"
        if proof_img.exists():
            st.image(cv2.cvtColor(imread_unicode(proof_img), cv2.COLOR_BGR2RGB), use_container_width=True)
        if gallery_img.exists():
            st.image(cv2.cvtColor(imread_unicode(gallery_img), cv2.COLOR_BGR2RGB), use_container_width=True)


if __name__ == "__main__":
    main()
