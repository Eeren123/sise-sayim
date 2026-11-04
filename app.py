from pathlib import Path
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

try:
    import av
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False

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
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1100px !important;
    }

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

    .stButton > button, .stDownloadButton > button {
        width: 100% !important;
        min-height: 3.1rem !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }

    button[data-baseweb="tab"] {
        font-size: 0.98rem !important;
        font-weight: 700 !important;
        padding: 0.65rem 1rem !important;
    }

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
            <p>Cihazınızın kamerasını açın veya fotoğraf yükleyin — sistem şişeleri çerçeve içine alıp anında sayar.</p>
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
                "📸 Kamerayı Aç & Çek (Telefon ve Bilgisayar Kamerası)",
                "🎥 Canlı Video Akışı (Gerçek Zamanlı Kamera - WebRTC)",
                "📱 Yüksek Çözünürlüklü Fotoğraf Çek / Galeriden Yükle",
                "🖼️ Örnek Koli Görselleri ile Test",
            ],
            horizontal=False,
        )

        # ------------------------------------------------------------------
        # 1) TARAYICI KAMERASI İLE CANLI ÖNİZLEME & TEK TUŞLA SAYIM (%100 UYUMLU)
        # ------------------------------------------------------------------
        if test_mode == "📸 Kamerayı Aç & Çek (Telefon ve Bilgisayar Kamerası)":
            st.info(
                "📷 Tarayıcınız kamera izni istediğinde **'İzin Ver (Allow)'** seçeneğine dokunun. "
                "Kamerayı şişelere tutup **'Take Photo'** butonuna bastığınızda anında sayım yapılır."
            )
            cam_photo = st.camera_input("Cihaz Kamerası")
            if cam_photo is not None:
                file_bytes = np.asarray(bytearray(cam_photo.read()), dtype=np.uint8)
                frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if frame_bgr is not None:
                    process_and_display(model, frame_bgr, show_input=False)

        # ------------------------------------------------------------------
        # 2) CANLI VİDEO AKIŞI (WEBRTC - GİRİLEN CİHAZIN KAMERASINI CANLI İŞLER)
        # ------------------------------------------------------------------
        elif test_mode == "🎥 Canlı Video Akışı (Gerçek Zamanlı Kamera - WebRTC)":
            if not HAS_WEBRTC:
                st.warning("Canlı WebRTC modülü yükleniyor. Lütfen '📸 Kamerayı Aç & Çek' seçeneğini kullanın.")
            else:
                st.info(
                    "🎥 Aşağıdaki **'START'** butonuna bastığınızda **giriş yaptığınız telefonun veya bilgisayarın kamerası** "
                    "canlı olarak açılır ve video üzerinde gerçek zamanlı şişe sayımı yapılır."
                )
                cam_facing = st.radio(
                    "Kamera Yönü (Telefondan Giriyorsanız):",
                    ["Arka Kamera (Kutu Sayımı İçin)", "Ön Kamera / Bilgisayar Kamerası"],
                    horizontal=True,
                )
                facing_mode = "environment" if "Arka" in cam_facing else "user"

                def video_frame_callback(frame: "av.VideoFrame") -> "av.VideoFrame":
                    img_bgr = frame.to_ndarray(format="bgr24")
                    vis_bgr, _, _ = count_bottles(
                        model,
                        img_bgr,
                        conf_thresh=FIXED_CONF_THRESH,
                        iou_thresh=FIXED_IOU_THRESH,
                        draw_numbers=False,
                        draw_boxes=True,
                        draw_centers=False,
                    )
                    return av.VideoFrame.from_ndarray(vis_bgr, format="bgr24")

                webrtc_streamer(
                    key=f"bottle-counter-{facing_mode}",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration={
                        "iceServers": [
                            {"urls": ["stun:stun.l.google.com:19302"]},
                            {"urls": ["stun:stun1.l.google.com:19302"]},
                            {"urls": ["stun:stun2.l.google.com:19302"]},
                        ]
                    },
                    media_stream_constraints={
                        "video": {"facingMode": facing_mode},
                        "audio": False,
                    },
                    video_frame_callback=video_frame_callback,
                    async_processing=True,
                )

        # ------------------------------------------------------------------
        # 3) TELEFON ARKA KAMERASI (TAM ÇÖZÜNÜRLÜK) VEYA GALERİDEN YÜKLE
        # ------------------------------------------------------------------
        elif test_mode == "📱 Yüksek Çözünürlüklü Fotoğraf Çek / Galeriden Yükle":
            st.info(
                "💡 Telefondan aşağıdaki kutuya dokunduğunuzda **'Fotoğraf Çek (Kamera)'** veya **'Galeri'** "
                "seçeneği açılır. Çok sayıda küçük şişe içeren kolilerde en net sonucu verir."
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
        # 4) ÖRNEK KOLİ GÖRSELLERİ İLE TEST (ORİJİNAL & ŞEFFAF)
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
