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
        padding-top: 0.8rem !important;
        padding-bottom: 1.5rem !important;
        max-width: 1000px !important;
    }

    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        padding: 0.95rem 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(56, 189, 248, 0.25);
        margin-bottom: 0.8rem;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.18);
    }
    .app-header h1 {
        font-size: 1.45rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
        color: #38bdf8 !important;
    }
    .app-header p {
        font-size: 0.88rem !important;
        color: #cbd5e1 !important;
        margin: 0.25rem 0 0 0 !important;
    }

    .count-card {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        color: #ffffff;
        padding: 1.05rem 1.2rem;
        border-radius: 16px;
        text-align: center;
        margin: 0.6rem 0 0.9rem 0;
        border: 2px solid #34d399;
        box-shadow: 0 10px 25px rgba(5, 150, 105, 0.28);
    }
    .count-card .count-label {
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #a7f3d0;
        margin-bottom: 0.2rem;
    }
    .count-card .count-value {
        font-size: 2.4rem;
        font-weight: 900;
        line-height: 1.1;
        color: #ffffff;
    }

    button[data-baseweb="tab"] {
        font-size: 0.96rem !important;
        font-weight: 700 !important;
        padding: 0.55rem 1rem !important;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.55rem !important;
            padding-right: 0.55rem !important;
            padding-top: 0.5rem !important;
        }
        .app-header {
            padding: 0.75rem 0.75rem;
            border-radius: 12px;
        }
        .app-header h1 {
            font-size: 1.18rem !important;
        }
        .app-header p {
            font-size: 0.78rem !important;
        }
        .count-card .count-value {
            font-size: 2.05rem;
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


def render_in_app_camera():
    """
    Uygulama dışına çıkmadan, doğrudan uygulama ekranı içinde kamerayı 60 FPS akıcı (donmadan) açan,
    arka plan Web Worker (ort.env.wasm.proxy = true) ile arayüzü hiç kilitlemeden çalışan
    ve uygulama içinden tek tuşla fotoğraf çekip şişeleri sayan bileşen.
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
                font-size: 0.84rem;
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
                font-size: 0.95rem;
                font-weight: 700;
                cursor: pointer;
                transition: transform 0.1s, opacity 0.2s;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 6px;
                text-align: center;
            }}
            .btn:active {{ transform: scale(0.98); }}
            .btn-start {{ background: #2563eb; color: #ffffff; }}
            .btn-switch {{ background: #334155; color: #f8fafc; }}
            .btn-snap {{
                background: linear-gradient(135deg, #059669 0%, #10b981 100%);
                color: #ffffff;
                grid-column: span 2;
                display: none;
                font-size: 1.08rem;
                padding: 15px 14px;
                border: 2px solid #6ee7b7;
                box-shadow: 0 6px 16px rgba(16, 185, 129, 0.3);
            }}
            .btn-live-toggle {{
                background: #475569;
                color: #f8fafc;
                 display: none;
            }}
            .btn-stop {{
                background: #dc2626;
                color: #ffffff;
                display: none;
            }}
            .status-bar {{
                font-size: 0.85rem;
                text-align: center;
                padding: 8px 12px;
                border-radius: 10px;
                background: #1e293b;
                color: #38bdf8;
                border: 1px solid rgba(56, 189, 248, 0.25);
                line-height: 1.35;
            }}
            .cam-stage {{
                position: relative;
                width: 100%;
                min-height: 360px;
                background: #0f172a;
                border-radius: 16px;
                overflow: hidden;
                border: 2px solid #334155;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            #camVideo, #photoCanvas {{
                width: 100%;
                height: auto;
                display: none;
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
                text-align: center;
                padding: 24px;
                color: #94a3b8;
                font-size: 0.94rem;
                line-height: 1.5;
            }}
        </style>
    </head>
    <body>
        <div class="live-wrapper">
            <div class="live-counter">
                <div class="lbl">🍾 Tespit Edilen Toplam Şişe Sayısı</div>
                <div class="val" id="bottleCount">0 Adet</div>
            </div>

            <div class="controls">
                <button class="btn btn-start" id="btnStartBack" onclick="startCamera('environment')">
                    📷 Arka Kamerayı Aç
                </button>
                <button class="btn btn-switch" id="btnStartFront" onclick="startCamera('user')">
                    💻 Ön / PC Kamerasını Aç
                </button>

                <!-- Kamera açılınca uygulama içinde fotoğraf çeken ana buton -->
                <button class="btn btn-snap" id="btnSnap" onclick="captureOrResume()">
                    📸 Fotoğraf Çek ve Şişeleri Say
                </button>

                <button class="btn btn-live-toggle" id="btnLiveToggle" onclick="toggleLiveDetect()">
                    🔄 Canlı Sayım: AÇIK
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
                    Uygulama içinde kamerayı açmak için yukarıdaki <b>"📷 Arka Kamerayı Aç"</b> (Telefon) veya <b>"💻 Ön / PC Kamerasını Aç"</b> (Bilgisayar) butonuna dokunun.
                </div>
                <video id="camVideo" autoplay playsinline muted></video>
                <canvas id="photoCanvas"></canvas>
                <canvas id="overlayCanvas"></canvas>
            </div>
        </div>

        <script>
            // Streamlit iframe'ine kamera izni yetkisi ekle
            try {{
                if (window.parent && window.parent.document) {{
                    const iframes = window.parent.document.querySelectorAll("iframe");
                    iframes.forEach(f => {{
                        const cur = f.getAttribute("allow") || "";
                        if (!cur.includes("camera")) {{
                            f.setAttribute("allow", (cur ? cur + "; " : "") + "camera *; microphone *");
                        }}
                    }});
                }}
            }} catch (e) {{}}

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
            let isCaptured = false;
            let liveDetectEnabled = true;
            let isBusy = false;

            const video = document.getElementById("camVideo");
            const photoCanvas = document.getElementById("photoCanvas");
            const photoCtx = photoCanvas.getContext("2d");
            const overlay = document.getElementById("overlayCanvas");
            const overlayCtx = overlay.getContext("2d");
            const countEl = document.getElementById("bottleCount");
            const statusEl = document.getElementById("statusBar");
            const placeholderEl = document.getElementById("placeholderMsg");
            const btnStop = document.getElementById("btnStop");
            const btnSnap = document.getElementById("btnSnap");
            const btnLiveToggle = document.getElementById("btnLiveToggle");

            const offCanvas = document.createElement("canvas");
            offCanvas.width = INPUT_SIZE;
            offCanvas.height = INPUT_SIZE;
            const offCtx = offCanvas.getContext("2d", {{ willReadFrequently: true }});

            async function initModel() {{
                try {{
                    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
                    ort.env.wasm.numThreads = 1;
                    // KRİTİK: proxy = true modeli arka planda Web Worker içinde çalıştırır,
                    // böylece kamera görüntüsü ASLA donmaz (60 FPS akıcı kalır)!
                    ort.env.wasm.proxy = true;

                    statusEl.innerText = "⏳ Eğitilmiş CNN modeli hazırlanıyor (~2 sn)...";
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
                        statusEl.innerText = "✅ CNN Modeli Hazır! Yukarıdaki butonla kamerayı açabilirsiniz.";
                        statusEl.style.color = "#34d399";
                    }} else {{
                        statusEl.innerText = "⚠️ Model yüklenemedi. Lütfen sayfayı yenileyin.";
                        statusEl.style.color = "#f87171";
                    }}
                }} catch (err) {{
                    statusEl.innerText = "⚠️ Model başlatma hatası: " + err.message;
                }}
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

            async function detectOnSource(sourceEl, vw, vh) {{
                if (!session) return [];
                const scale = Math.min(INPUT_SIZE / vw, INPUT_SIZE / vh);
                const nw = Math.round(vw * scale);
                const nh = Math.round(vh * scale);
                const padX = Math.floor((INPUT_SIZE - nw) / 2);
                const padY = Math.floor((INPUT_SIZE - nh) / 2);

                offCtx.fillStyle = "rgb(114, 114, 114)";
                offCtx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
                offCtx.drawImage(sourceEl, padX, padY, nw, nh);

                const imgData = offCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
                const numPixels = INPUT_SIZE * INPUT_SIZE;
                const floatData = new Float32Array(3 * numPixels);

                for (let i = 0; i < numPixels; i++) {{
                    floatData[i] = imgData[i * 4] / 255.0;
                    floatData[numPixels + i] = imgData[i * 4 + 1] / 255.0;
                    floatData[2 * numPixels + i] = imgData[i * 4 + 2] / 255.0;
                }}

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
                return nms(rawBoxes, IOU_THRESH);
            }}

            function drawBoxes(boxes, vw, vh) {{
                overlayCtx.clearRect(0, 0, vw, vh);
                overlayCtx.strokeStyle = "#00e640";
                overlayCtx.lineWidth = Math.max(3, Math.round(vw / 220));
                for (const b of boxes) {{
                    overlayCtx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
                }}
            }}

            async function startCamera(facingMode) {{
                stopCamera();
                try {{
                    statusEl.innerText = "📷 Kamera açılıyor...";
                    statusEl.style.color = "#38bdf8";

                    const constraints = {{
                        audio: false,
                        video: {{
                            facingMode: {{ ideal: facingMode }},
                            width: {{ ideal: 960 }},
                            height: {{ ideal: 720 }},
                            frameRate: {{ ideal: 30 }}
                        }}
                    }};
                    try {{
                        videoStream = await navigator.mediaDevices.getUserMedia(constraints);
                    }} catch (firstErr) {{
                        videoStream = await navigator.mediaDevices.getUserMedia({{ video: true, audio: false }});
                    }}

                    photoCanvas.style.display = "none";
                    placeholderEl.style.display = "none";
                    video.style.display = "block";
                    video.srcObject = videoStream;
                    await video.play();

                    btnSnap.style.display = "flex";
                    btnSnap.innerText = "📸 Fotoğraf Çek ve Şişeleri Say";
                    btnSnap.style.background = "linear-gradient(135deg, #059669 0%, #10b981 100%)";
                    btnLiveToggle.style.display = "flex";
                    btnStop.style.display = "flex";
                    isRunning = true;
                    isCaptured = false;
                    statusEl.innerText = "🟢 Kamera Açık (Akıcı Mod) — İstediğiniz anda '📸 Fotoğraf Çek ve Şişeleri Say' butonuna basın.";
                    statusEl.style.color = "#34d399";

                    scheduleNextLiveDetect();
                }} catch (err) {{
                    statusEl.innerHTML = "⚠️ <b>Kamera açılamadı:</b> Tarayıcı adres çubuğunun solundaki 🔒/⚙️ simgesinden Kamera iznini <b>'İzin Ver (Allow)'</b> yapıp sayfayı yenileyin.";
                    statusEl.style.color = "#fbbf24";
                }}
            }}

            function toggleLiveDetect() {{
                liveDetectEnabled = !liveDetectEnabled;
                if (liveDetectEnabled) {{
                    btnLiveToggle.innerText = "🔄 Canlı Sayım: AÇIK";
                    btnLiveToggle.style.background = "#475569";
                    scheduleNextLiveDetect();
                }} else {{
                    btnLiveToggle.innerText = "🔄 Canlı Sayım: KAPALI (Sadece Fotoğrafta)";
                    btnLiveToggle.style.background = "#334155";
                    if (!isCaptured) {{
                        overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
                    }}
                }}
            }}

            async function captureOrResume() {{
                if (!videoStream) return;
                if (!isCaptured) {{
                    // Uygulama içindeki kameradan anlık fotoğrafı yakala ve say
                    isCaptured = true;
                    const vw = video.videoWidth;
                    const vh = video.videoHeight;
                    photoCanvas.width = vw;
                    photoCanvas.height = vh;
                    overlay.width = vw;
                    overlay.height = vh;

                    photoCtx.drawImage(video, 0, 0, vw, vh);
                    video.style.display = "none";
                    photoCanvas.style.display = "block";

                    btnSnap.innerText = "⏳ Fotoğraftaki Şişeler Sayılıyor...";
                    statusEl.innerText = "⏳ Çekilen fotoğraf analiz ediliyor...";

                    const boxes = await detectOnSource(photoCanvas, vw, vh);
                    countEl.innerText = boxes.length + " Adet";
                    drawBoxes(boxes, vw, vh);

                    btnSnap.innerText = "🔄 Yeni Fotoğraf Çek (Kameraya Dön)";
                    btnSnap.style.background = "#2563eb";
                    statusEl.innerText = "✅ Fotoğraf Çekildi! Tespit Edilen Toplam Şişe: " + boxes.length + " Adet";
                    statusEl.style.color = "#34d399";
                }} else {{
                    // Tekrar canlı kameraya dön
                    isCaptured = false;
                    photoCanvas.style.display = "none";
                    video.style.display = "block";
                    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
                    btnSnap.innerText = "📸 Fotoğraf Çek ve Şişeleri Say";
                    btnSnap.style.background = "linear-gradient(135deg, #059669 0%, #10b981 100%)";
                    statusEl.innerText = "🟢 Kamera Açık — İstediğiniz anda '📸 Fotoğraf Çek ve Şişeleri Say' butonuna basın.";
                    scheduleNextLiveDetect();
                }}
            }}

            function stopCamera() {{
                isRunning = false;
                isCaptured = false;
                if (videoStream) {{
                    videoStream.getTracks().forEach(t => t.stop());
                    videoStream = null;
                }}
                video.srcObject = null;
                video.style.display = "none";
                photoCanvas.style.display = "none";
                overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
                btnSnap.style.display = "none";
                btnLiveToggle.style.display = "none";
                btnStop.style.display = "none";
                placeholderEl.style.display = "block";
                countEl.innerText = "0 Adet";
            }}

            function scheduleNextLiveDetect() {{
                if (!isRunning || isCaptured || !liveDetectEnabled) return;
                setTimeout(runLiveStep, 220);
            }}

            async function runLiveStep() {{
                if (!isRunning || isCaptured || !liveDetectEnabled) return;
                if (isBusy || video.readyState < 2 || !session || video.videoWidth === 0) {{
                    scheduleNextLiveDetect();
                    return;
                }}

                isBusy = true;
                try {{
                    const vw = video.videoWidth;
                    const vh = video.videoHeight;
                    if (overlay.width !== vw || overlay.height !== vh) {{
                        overlay.width = vw;
                        overlay.height = vh;
                    }}
                    const boxes = await detectOnSource(video, vw, vh);
                    if (isRunning && !isCaptured && liveDetectEnabled) {{
                        countEl.innerText = boxes.length + " Adet";
                        drawBoxes(boxes, vw, vh);
                    }}
                }} catch (e) {{
                    console.error("Live detect error:", e);
                }} finally {{
                    isBusy = false;
                    scheduleNextLiveDetect();
                }}
            }}

            initModel();
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=740, scrolling=False)


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
            <p>Uygulama içinden kamerayı açıp canlı sayım yapın veya fotoğraf çekerek şişeleri sayın.</p>
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
                "📷 Uygulama İçi Kamera (Canlı Sayım & Fotoğraf Çek)",
                "🖼️ Örnek Koli Görselleri ile Test",
            ],
            horizontal=True,
        )

        if test_mode == "📷 Uygulama İçi Kamera (Canlı Sayım & Fotoğraf Çek)":
            render_in_app_camera()

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
