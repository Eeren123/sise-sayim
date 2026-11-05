from pathlib import Path
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

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
    .block-container {
        padding-top: 1.0rem !important;
        padding-bottom: 2rem !important;
        max-width: 1050px !important;
    }

    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        padding: 1.0rem 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(56, 189, 248, 0.25);
        margin-bottom: 0.9rem;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.18);
    }
    .app-header h1 {
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
        color: #38bdf8 !important;
    }
    .app-header p {
        font-size: 0.9rem !important;
        color: #cbd5e1 !important;
        margin: 0.3rem 0 0 0 !important;
    }

    .count-card {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        color: #ffffff;
        padding: 1.1rem 1.2rem;
        border-radius: 16px;
        text-align: center;
        margin: 0.7rem 0 1.0rem 0;
        border: 2px solid #34d399;
        box-shadow: 0 10px 25px rgba(5, 150, 105, 0.28);
    }
    .count-card .count-label {
        font-size: 0.92rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #a7f3d0;
        margin-bottom: 0.2rem;
    }
    .count-card .count-value {
        font-size: 2.5rem;
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
        padding: 0.6rem 1rem !important;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.65rem !important;
            padding-right: 0.65rem !important;
            padding-top: 0.6rem !important;
        }
        .app-header {
            padding: 0.8rem 0.8rem;
            border-radius: 12px;
        }
        .app-header h1 {
            font-size: 1.2rem !important;
        }
        .app-header p {
            font-size: 0.8rem !important;
        }
        .count-card .count-value {
            font-size: 2.1rem;
        }
        div[role="radiogroup"] {
            gap: 0.4rem !important;
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


def render_live_onnx_camera():
    """
    Tarayıcı içinde doğrudan telefonun/bilgisayarın kamerasına bağlanan (getUserMedia),
    canlı görüntü üzerinde şişeleri anlık çerçeveleyip sayan ve '📸 Fotoğraf Çek' butonuyla
    görüntüyü dondurup kaydedebilen bileşen.
    """
    html_code = f"""
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js"></script>
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: system-ui, -apple-system, sans-serif; }}
            body {{ background: transparent; color: #f8fafc; }}
            .live-wrapper {{
                display: flex;
                flex-direction: column;
                gap: 10px;
                width: 100%;
                max-width: 720px;
                margin: 0 auto;
            }}
            .live-counter {{
                background: linear-gradient(135deg, #065f46 0%, #047857 100%);
                border: 2px solid #34d399;
                border-radius: 14px;
                padding: 12px 16px;
                text-align: center;
                box-shadow: 0 6px 18px rgba(5, 150, 105, 0.25);
            }}
            .live-counter .lbl {{
                font-size: 0.85rem;
                font-weight: 700;
                text-transform: uppercase;
                color: #a7f3d0;
                letter-spacing: 0.05em;
            }}
            .live-counter .val {{
                font-size: 2.3rem;
                font-weight: 900;
                color: #ffffff;
                line-height: 1.15;
            }}
            .controls {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 8px;
            }}
            .btn {{
                padding: 13px 12px;
                border-radius: 12px;
                border: none;
                font-size: 0.96rem;
                font-weight: 700;
                cursor: pointer;
                transition: transform 0.1s, opacity 0.2s;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 6px;
            }}
            .btn:active {{ transform: scale(0.98); }}
            .btn-start {{ background: #2563eb; color: #ffffff; }}
            .btn-switch {{ background: #334155; color: #f8fafc; }}
            .btn-snap {{ background: #059669; color: #ffffff; grid-column: span 2; display: none; font-size: 1.05rem; border: 2px solid #34d399; }}
            .btn-stop {{ background: #dc2626; color: #ffffff; grid-column: span 2; display: none; }}
            .status-bar {{
                font-size: 0.85rem;
                text-align: center;
                padding: 8px 12px;
                border-radius: 10px;
                background: #1e293b;
                color: #38bdf8;
                border: 1px solid rgba(56, 189, 248, 0.25);
            }}
            .cam-stage {{
                position: relative;
                width: 100%;
                min-height: 320px;
                background: #0f172a;
                border-radius: 16px;
                overflow: hidden;
                border: 2px solid #334155;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            #camVideo {{
                width: 100%;
                height: auto;
                display: block;
            }}
            #overlayCanvas {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
            }}
            .placeholder-msg {{
                position: absolute;
                text-align: center;
                padding: 20px;
                color: #94a3b8;
                font-size: 0.95rem;
            }}
        </style>
    </head>
    <body>
        <div class="live-wrapper">
            <div class="live-counter">
                <div class="lbl">🍾 Kamerada Tespit Edilen Şişe Sayısı</div>
                <div class="val" id="bottleCount">0 Adet</div>
            </div>

            <div class="controls">
                <button class="btn btn-start" id="btnStart" onclick="startCamera('environment')">
                    📷 Arka Kamerayı Aç
                </button>
                <button class="btn btn-switch" id="btnFront" onclick="startCamera('user')">
                    💻 Ön / PC Kamerasını Aç
                </button>
                <button class="btn btn-snap" id="btnSnap" onclick="toggleFreeze()">
                    📸 Fotoğraf Çek / Görüntüyü Sabitle
                </button>
                <button class="btn btn-stop" id="btnStop" onclick="stopCamera()">
                    ⏹️ Kamerayı Kapat
                </button>
            </div>

            <div class="status-bar" id="statusBar">
                🤖 CNN Modeli hazırlanıyor, lütfen bekleyin...
            </div>

            <div class="cam-stage">
                <div class="placeholder-msg" id="placeholderMsg">
                    Kamerayı açmak için yukarıdaki <b>"📷 Arka Kamerayı Aç"</b> veya <b>"💻 Ön / PC Kamerasını Aç"</b> butonuna dokunun.<br>
                    Kameranın gördüğü alandaki şişeler canlı olarak çerçevelenip sayılacaktır.
                </div>
                <video id="camVideo" autoplay playsinline muted></video>
                <canvas id="overlayCanvas"></canvas>
            </div>
        </div>

        <script>
            const MODEL_URLS = [
                "https://raw.githubusercontent.com/Eeren123/sise-sayim/main/models/bottle_yolov8_best.onnx",
                "https://cdn.jsdelivr.net/gh/Eeren123/sise-sayim@main/models/bottle_yolov8_best.onnx"
            ];
            const CONF_THRESH = {FIXED_CONF_THRESH};
            const IOU_THRESH = {FIXED_IOU_THRESH};
            const INPUT_SIZE = 640;

            let session = null;
            let videoStream = null;
            let isRunning = false;
            let isFrozen = false;

            const video = document.getElementById("camVideo");
            const overlay = document.getElementById("overlayCanvas");
            const overlayCtx = overlay.getContext("2d");
            const countEl = document.getElementById("bottleCount");
            const statusEl = document.getElementById("statusBar");
            const placeholderEl = document.getElementById("placeholderMsg");
            const btnStop = document.getElementById("btnStop");
            const btnSnap = document.getElementById("btnSnap");

            const offCanvas = document.createElement("canvas");
            offCanvas.width = INPUT_SIZE;
            offCanvas.height = INPUT_SIZE;
            const offCtx = offCanvas.getContext("2d", {{ willReadFrequently: true }});

            async function initModel() {{
                try {{
                    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
                    ort.env.wasm.numThreads = 1;
                    statusEl.innerText = "⏳ Eğitilmiş CNN modeli indiriliyor (ilk açılışta ~3 sn sürer)...";
                    let loaded = false;
                    for (const url of MODEL_URLS) {{
                        try {{
                            session = await ort.InferenceSession.create(url, {{ executionProviders: ["wasm"] }});
                            loaded = true;
                            break;
                        }} catch (e) {{
                            console.warn("Model URL denemesi basarisiz:", url, e);
                        }}
                    }}
                    if (loaded) {{
                        statusEl.innerText = "✅ CNN Modeli Hazır! Yukarıdaki mavi butonla kamerayı açabilirsiniz.";
                        statusEl.style.color = "#34d399";
                    }} else {{
                        statusEl.innerText = "⚠️ Model yüklenemedi. Lütfen sayfayı yenileyin.";
                        statusEl.style.color = "#f87171";
                    }}
                }} catch (err) {{
                    statusEl.innerText = "⚠️ Model başlatma hatası: " + err.message;
                }}
            }}

            async function startCamera(facingMode) {{
                stopCamera();
                try {{
                    statusEl.innerText = "📷 Kamera açılıyor, izin istenirse 'İzin Ver'e dokunun...";
                    const constraints = {{
                        audio: false,
                        video: {{
                            facingMode: {{ ideal: facingMode }},
                            width: {{ ideal: 1280 }},
                            height: {{ ideal: 720 }}
                        }}
                    }};
                    videoStream = await navigator.mediaDevices.getUserMedia(constraints);
                    video.srcObject = videoStream;
                    await video.play();

                    placeholderEl.style.display = "none";
                    btnSnap.style.display = "flex";
                    btnSnap.innerText = "📸 Fotoğraf Çek / Görüntüyü Sabitle";
                    btnStop.style.display = "flex";
                    isRunning = true;
                    isFrozen = false;
                    statusEl.innerText = "🟢 Canlı Kamera Aktif — Şişeler anlık olarak sayılıyor";
                    statusEl.style.color = "#34d399";

                    requestAnimationFrame(processLoop);
                }} catch (err) {{
                    statusEl.innerText = "❌ Kamera açılamadı: Lütfen tarayıcıdan kamera izni verin (" + err.message + ")";
                    statusEl.style.color = "#f87171";
                }}
            }}

            function toggleFreeze() {{
                if (!videoStream) return;
                if (!isFrozen) {{
                    isFrozen = true;
                    video.pause();
                    btnSnap.innerText = "▶️ Canlı Sayıma Devam Et";
                    btnSnap.style.background = "#2563eb";
                    statusEl.innerText = "📸 Fotoğraf çekildi ve sabitlendi! Sonuç yukarıda gösteriliyor.";
                }} else {{
                    isFrozen = false;
                    video.play();
                    btnSnap.innerText = "📸 Fotoğraf Çek / Görüntüyü Sabitle";
                    btnSnap.style.background = "#059669";
                    statusEl.innerText = "🟢 Canlı Kamera Aktif — Şişeler anlık olarak sayılıyor";
                    requestAnimationFrame(processLoop);
                }}
            }}

            function stopCamera() {{
                isRunning = false;
                isFrozen = false;
                if (videoStream) {{
                    videoStream.getTracks().forEach(t => t.stop());
                    videoStream = null;
                }}
                video.srcObject = null;
                overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
                btnSnap.style.display = "none";
                btnStop.style.display = "none";
                placeholderEl.style.display = "block";
                countEl.innerText = "0 Adet";
            }}

            function computeIoU(a, b) {{
                const xx1 = Math.max(a.x1, b.x1);
                const yy1 = Math.max(a.y1, b.y1);
                const xx2 = Math.min(a.x2, b.x2);
                const yy2 = Math.min(a.y2, b.y2);
                const w = Math.max(0, xx2 - xx1);
                const h = Math.max(0, yy2 - yy1);
                const inter = w * h;
                const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
                const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
                return inter / (areaA + areaB - inter + 1e-6);
            }}

            function nms(boxes, iouThresh) {{
                boxes.sort((a, b) => b.score - a.score);
                const selected = [];
                const active = new Array(boxes.length).fill(true);
                for (let i = 0; i < boxes.length; i++) {{
                    if (!active[i]) continue;
                    selected.push(boxes[i]);
                    for (let j = i + 1; j < boxes.length; j++) {{
                        if (active[j] && computeIoU(boxes[i], boxes[j]) > iouThresh) {{
                            active[j] = false;
                        }}
                    }}
                }}
                return selected;
            }}

            async function processLoop() {{
                if (!isRunning || isFrozen) return;

                if (video.readyState >= 2 && session && video.videoWidth > 0) {{
                    const vw = video.videoWidth;
                    const vh = video.videoHeight;

                    if (overlay.width !== vw || overlay.height !== vh) {{
                        overlay.width = vw;
                        overlay.height = vh;
                    }}

                    const scale = Math.min(INPUT_SIZE / vw, INPUT_SIZE / vh);
                    const nw = Math.round(vw * scale);
                    const nh = Math.round(vh * scale);
                    const padX = Math.floor((INPUT_SIZE - nw) / 2);
                    const padY = Math.floor((INPUT_SIZE - nh) / 2);

                    offCtx.fillStyle = "rgb(114, 114, 114)";
                    offCtx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
                    offCtx.drawImage(video, padX, padY, nw, nh);

                    const imgData = offCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
                    const numPixels = INPUT_SIZE * INPUT_SIZE;
                    const floatData = new Float32Array(3 * numPixels);

                    for (let i = 0; i < numPixels; i++) {{
                        floatData[i] = imgData[i * 4] / 255.0;
                        floatData[numPixels + i] = imgData[i * 4 + 1] / 255.0;
                        floatData[2 * numPixels + i] = imgData[i * 4 + 2] / 255.0;
                    }}

                    try {{
                        const inputTensor = new ort.Tensor("float32", floatData, [1, 3, INPUT_SIZE, INPUT_SIZE]);
                        const feeds = {{ [session.inputNames[0]]: inputTensor }};
                        const results = await session.run(feeds);
                        const out = results[session.outputNames[0]].data;
                        const numAnchors = 8400;

                        const rawBoxes = [];
                        for (let i = 0; i < numAnchors; i++) {{
                            const score = out[4 * numAnchors + i];
                            if (score >= CONF_THRESH) {{
                                const cx = out[0 * numAnchors + i];
                                const cy = out[1 * numAnchors + i];
                                const w = out[2 * numAnchors + i];
                                const h = out[3 * numAnchors + i];

                                const x1 = Math.max(0, (cx - w / 2 - padX) / scale);
                                const y1 = Math.max(0, (cy - h / 2 - padY) / scale);
                                const x2 = Math.min(vw, (cx + w / 2 - padX) / scale);
                                const y2 = Math.min(vh, (cy + h / 2 - padY) / scale);

                                if (x2 - x1 > 6 && y2 - y1 > 6) {{
                                    rawBoxes.push({{ x1, y1, x2, y2, score }});
                                }}
                            }}
                        }}

                        const finalBoxes = nms(rawBoxes, IOU_THRESH);
                        countEl.innerText = finalBoxes.length + " Adet";

                        overlayCtx.clearRect(0, 0, vw, vh);
                        overlayCtx.strokeStyle = "#00e640";
                        overlayCtx.lineWidth = Math.max(3, Math.round(vw / 240));
                        for (const b of finalBoxes) {{
                            overlayCtx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
                        }}
                    }} catch (e) {{
                        console.error("Inference error:", e);
                    }}
                }}

                if (isRunning && !isFrozen) {{
                    setTimeout(() => requestAnimationFrame(processLoop), 40);
                }}
            }}

            initModel();
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=720, scrolling=False)


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


def main():
    st.markdown(
        """
        <div class="app-header">
            <h1>🍾 Otomatik Şişe Sayım Sistemi (CNN)</h1>
            <p>Kameraya bağlanarak canlı alanda şişeleri sayın veya fotoğraf çekerek test edin.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not DEFAULT_MODEL_PATH.exists():
        st.warning("Eğitilmiş model bulunamadı.")
        return

    model = get_model(str(DEFAULT_MODEL_PATH))

    tab_test, tab_verify = st.tabs([
        "🔍 Canlı Şişe Testi",
        "✅ 83 Şişe Doğrulama",
    ])

    with tab_test:
        test_mode = st.radio(
            "Test Yöntemi Seçin:",
            [
                "📷 Canlı Kameraya Bağlan (Anlık Canlı Sayım)",
                "📸 Fotoğraf Çek / Galeriden Yükle",
                "🖼️ Örnek Koli Görselleri ile Test",
            ],
            horizontal=True,
        )

        # ------------------------------------------------------------------
        # 1) CANLI KAMERA VİZÖRÜ (DOĞRUDAN TELEFON/PC KAMERASINA BAĞLANIR VE CANLI SAYAR)
        # ------------------------------------------------------------------
        if test_mode == "📷 Canlı Kameraya Bağlan (Anlık Canlı Sayım)":
            render_live_onnx_camera()

        # ------------------------------------------------------------------
        # 2) KAMERADAN FOTOĞRAF ÇEK (TAKE PHOTO) VEYA DOSYA YÜKLE (UPLOAD)
        # ------------------------------------------------------------------
        elif test_mode == "📸 Fotoğraf Çek / Galeriden Yükle":
            st.markdown("##### 📸 1. Kameradan Doğrudan Fotoğraf Çek")
            cam_photo = st.camera_input("Kamerayı açıp 'Take Photo' butonuna basarak fotoğraf çekin")
            if cam_photo is not None:
                file_bytes = np.asarray(bytearray(cam_photo.read()), dtype=np.uint8)
                frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if frame_bgr is not None:
                    process_and_display(model, frame_bgr, show_input=False)

            st.markdown("---")
            st.markdown("##### 📁 2. Veya Cihazdan / Galeriden Fotoğraf Yükle")
            uploaded_file = st.file_uploader(
                "Telefon galerisinden veya bilgisayardan fotoğraf seçin",
                type=["jpg", "jpeg", "png", "webp"],
            )
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    process_and_display(model, img_bgr, show_input=False)

        # ------------------------------------------------------------------
        # 3) ÖRNEK KOLİ GÖRSELLERİ İLE TEST
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
