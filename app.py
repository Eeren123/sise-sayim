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

    .stButton > button {
        width: 100% !important;
        min-height: 3.3rem !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1.02rem !important;
    }

    button[data-baseweb="tab"] {
        font-size: 1.0rem !important;
        font-weight: 700 !important;
        padding: 0.65rem 1.1rem !important;
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
        button[data-baseweb="tab"] {
            font-size: 0.88rem !important;
            padding: 0.5rem 0.65rem !important;
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


def render_live_counting_page():
    """
    1. SAYFA: Sadece Canlı Kamera ile Anlık Sayım.
    Kamera açılır ve gördüğü alandaki şişeleri gerçek zamanlı çerçeveleyip sayar.
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
                padding: 14px 12px;
                border-radius: 12px;
                border: none;
                font-size: 0.96rem;
                font-weight: 700;
                cursor: pointer;
                transition: transform 0.1s;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 6px;
                text-align: center;
            }}
            .btn:active {{ transform: scale(0.98); }}
            .btn-start {{ background: #2563eb; color: #ffffff; }}
            .btn-switch {{ background: #334155; color: #f8fafc; }}
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
                min-height: 360px;
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
                font-size: 0.95rem;
                line-height: 1.5;
            }}
        </style>
    </head>
    <body>
        <div class="live-wrapper">
            <div class="live-counter">
                <div class="lbl">🍾 Canlı Kamerada Tespit Edilen Şişe Sayısı</div>
                <div class="val" id="bottleCount">0 Adet</div>
            </div>

            <div class="controls">
                <button class="btn btn-start" onclick="startCamera('environment')">
                    📷 Canlı Kamerayı Aç (Arka)
                </button>
                <button class="btn btn-switch" onclick="startCamera('user')">
                    💻 Canlı Kamerayı Aç (Ön / PC)
                </button>
                <button class="btn btn-stop" id="btnStop" onclick="stopCamera()">
                    ⏹️ Canlı Kamerayı Kapat
                </button>
            </div>

            <div class="status-bar" id="statusBar">
                🤖 CNN Modeli hazırlanıyor, lütfen bekleyin...
            </div>

            <div class="cam-stage">
                <div class="placeholder-msg" id="placeholderMsg">
                    Canlı sayımı başlatmak için yukarıdaki <b>"📷 Canlı Kamerayı Aç"</b> butonuna dokunun.<br>
                    Kameranın gördüğü alandaki şişeler anlık olarak çerçevelenip sayılacaktır.
                </div>
                <video id="camVideo" autoplay playsinline muted></video>
                <canvas id="overlayCanvas"></canvas>
            </div>
        </div>

        <script>
            try {{
                if (window.parent && window.parent.document) {{
                    window.parent.document.querySelectorAll("iframe").forEach(f => {{
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
            let isBusy = false;

            const video = document.getElementById("camVideo");
            const overlay = document.getElementById("overlayCanvas");
            const overlayCtx = overlay.getContext("2d");
            const countEl = document.getElementById("bottleCount");
            const statusEl = document.getElementById("statusBar");
            const placeholderEl = document.getElementById("placeholderMsg");
            const btnStop = document.getElementById("btnStop");

            const offCanvas = document.createElement("canvas");
            offCanvas.width = INPUT_SIZE;
            offCanvas.height = INPUT_SIZE;
            const offCtx = offCanvas.getContext("2d", {{ willReadFrequently: true }});

            async function initModel() {{
                try {{
                    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
                    ort.env.wasm.numThreads = 1;
                    ort.env.wasm.proxy = true;

                    statusEl.innerText = "⏳ Eğitilmiş CNN modeli hazırlanıyor (~2 sn)...";
                    for (const url of MODEL_URLS) {{
                        try {{
                            session = await ort.InferenceSession.create(url, {{ executionProviders: ["wasm"] }});
                            break;
                        }} catch (e) {{}}
                    }}
                    if (session) {{
                        statusEl.innerText = "✅ CNN Modeli Hazır! Yukarıdaki butonla canlı kamerayı açabilirsiniz.";
                        statusEl.style.color = "#34d399";
                    }} else {{
                        statusEl.innerText = "⚠️ Model yüklenemedi. Lütfen sayfayı yenileyin.";
                        statusEl.style.color = "#f87171";
                    }}
                }} catch (err) {{
                    statusEl.innerText = "⚠️ Model hatası: " + err.message;
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
                    }} catch (e) {{
                        videoStream = await navigator.mediaDevices.getUserMedia({{ video: true, audio: false }});
                    }}

                    placeholderEl.style.display = "none";
                    video.style.display = "block";
                    video.srcObject = videoStream;
                    await video.play();

                    btnStop.style.display = "flex";
                    isRunning = true;
                    statusEl.innerText = "🟢 Canlı Sayım Aktif — Kameranın gördüğü şişeler anlık sayılıyor.";
                    statusEl.style.color = "#34d399";

                    scheduleNextLiveDetect();
                }} catch (err) {{
                    statusEl.innerHTML = "⚠️ <b>Kamera açılamadı:</b> Tarayıcı adres çubuğunun solundaki 🔒/⚙️ simgesinden Kamera iznini <b>'İzin Ver'</b> yapın.";
                    statusEl.style.color = "#fbbf24";
                }}
            }}

            function stopCamera() {{
                isRunning = false;
                if (videoStream) {{
                    videoStream.getTracks().forEach(t => t.stop());
                    videoStream = null;
                }}
                video.srcObject = null;
                video.style.display = "none";
                overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
                btnStop.style.display = "none";
                placeholderEl.style.display = "block";
                countEl.innerText = "0 Adet";
            }}

            function scheduleNextLiveDetect() {{
                if (!isRunning) return;
                setTimeout(runLiveStep, 200);
            }}

            async function runLiveStep() {{
                if (!isRunning) return;
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
                    if (isRunning) {{
                        countEl.innerText = boxes.length + " Adet";
                        overlayCtx.clearRect(0, 0, vw, vh);
                        overlayCtx.strokeStyle = "#00e640";
                        overlayCtx.lineWidth = Math.max(3, Math.round(vw / 220));
                        for (const b of boxes) {{
                            overlayCtx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
                        }}
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
    components.html(html_code, height=700, scrolling=False)


def render_in_app_photo_capture():
    """
    2. SAYFADAKİ '📸 Fotoğraf Çek' butonu seçildiğinde uygulama içinde kamerayı açıp
    fotoğraf çeken ve şişeleri sayan bileşen.
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
            .snap-wrapper {{
                display: flex;
                flex-direction: column;
                gap: 10px;
                width: 100%;
                max-width: 720px;
                margin: 0 auto;
            }}
            .snap-counter {{
                background: linear-gradient(135deg, #065f46 0%, #047857 100%);
                border: 2px solid #34d399;
                border-radius: 14px;
                padding: 12px 16px;
                text-align: center;
                box-shadow: 0 6px 18px rgba(5, 150, 105, 0.25);
            }}
            .snap-counter .lbl {{
                font-size: 0.84rem;
                font-weight: 700;
                text-transform: uppercase;
                color: #a7f3d0;
            }}
            .snap-counter .val {{
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
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 6px;
            }}
            .btn-cam {{ background: #2563eb; color: #ffffff; }}
            .btn-cam-front {{ background: #334155; color: #f8fafc; }}
            .btn-capture {{
                background: linear-gradient(135deg, #059669 0%, #10b981 100%);
                color: #ffffff;
                grid-column: span 2;
                font-size: 1.08rem;
                padding: 15px;
                border: 2px solid #6ee7b7;
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
            }}
            .stage {{
                position: relative;
                width: 100%;
                min-height: 350px;
                background: #0f172a;
                border-radius: 16px;
                overflow: hidden;
                border: 2px solid #334155;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            #camVideo, #photoCanvas {{ width: 100%; height: auto; display: none; }}
            #overlayCanvas {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }}
            .placeholder-msg {{ text-align: center; padding: 24px; color: #94a3b8; font-size: 0.94rem; }}
        </style>
    </head>
    <body>
        <div class="snap-wrapper">
            <div class="snap-counter">
                <div class="lbl">🍾 Çekilen Fotoğraftaki Toplam Şişe Sayısı</div>
                <div class="val" id="bottleCount">0 Adet</div>
            </div>

            <div class="controls">
                <button class="btn btn-cam" onclick="openCam('environment')">📷 Arka Kamerayı Aç</button>
                <button class="btn btn-cam-front" onclick="openCam('user')">💻 Ön / PC Kamerasını Aç</button>
                <button class="btn btn-capture" id="btnCapture" onclick="takePhotoAndCount()">📸 Fotoğrafı Çek ve Şişeleri Say</button>
            </div>

            <div class="status-bar" id="statusBar">⏳ CNN Modeli hazırlanıyor...</div>

            <div class="stage">
                <div class="placeholder-msg" id="placeholderMsg">
                    Kamerayı açıp uygulama içinden fotoğraf çekmek için yukarıdaki <b>"📷 Kamerayı Aç"</b> butonuna basın.
                </div>
                <video id="camVideo" autoplay playsinline muted></video>
                <canvas id="photoCanvas"></canvas>
                <canvas id="overlayCanvas"></canvas>
            </div>
        </div>

        <script>
            try {{
                if (window.parent && window.parent.document) {{
                    window.parent.document.querySelectorAll("iframe").forEach(f => {{
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
            let stream = null;
            let photoTaken = false;

            const video = document.getElementById("camVideo");
            const photoCanvas = document.getElementById("photoCanvas");
            const photoCtx = photoCanvas.getContext("2d");
            const overlay = document.getElementById("overlayCanvas");
            const overlayCtx = overlay.getContext("2d");
            const countEl = document.getElementById("bottleCount");
            const statusEl = document.getElementById("statusBar");
            const placeholderEl = document.getElementById("placeholderMsg");
            const btnCapture = document.getElementById("btnCapture");

            const offCanvas = document.createElement("canvas");
            offCanvas.width = INPUT_SIZE;
            offCanvas.height = INPUT_SIZE;
            const offCtx = offCanvas.getContext("2d", {{ willReadFrequently: true }});

            async function init() {{
                ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
                ort.env.wasm.numThreads = 1;
                ort.env.wasm.proxy = true;
                for (const url of MODEL_URLS) {{
                    try {{
                        session = await ort.InferenceSession.create(url, {{ executionProviders: ["wasm"] }});
                        break;
                    }} catch (e) {{}}
                }}
                if (session) {{
                    statusEl.innerText = "✅ Kamera ile fotoğraf çekmeye hazır!";
                    statusEl.style.color = "#34d399";
                    // Otomatik olarak kamerayı açmayı dene
                    openCam('environment');
                }}
            }}

            function computeIoU(a, b) {{
                const xx1 = Math.max(a.x1, b.x1), yy1 = Math.max(a.y1, b.y1);
                const xx2 = Math.min(a.x2, b.x2), yy2 = Math.min(a.y2, b.y2);
                const inter = Math.max(0, xx2 - xx1) * Math.max(0, yy2 - yy1);
                return inter / ((a.x2 - a.x1)*(a.y2 - a.y1) + (b.x2 - b.x1)*(b.y2 - b.y1) - inter + 1e-6);
            }}

            function nms(boxes, iouThresh) {{
                boxes.sort((a, b) => b.score - a.score);
                const selected = [], active = new Array(boxes.length).fill(true);
                for (let i = 0; i < boxes.length; i++) {{
                    if (!active[i]) continue;
                    selected.push(boxes[i]);
                    for (let j = i + 1; j < boxes.length; j++) {{
                        if (active[j] && computeIoU(boxes[i], boxes[j]) > iouThresh) active[j] = false;
                    }}
                }}
                return selected;
            }}

            async function openCam(facingMode) {{
                if (stream) stream.getTracks().forEach(t => t.stop());
                try {{
                    try {{
                        stream = await navigator.mediaDevices.getUserMedia({{
                            audio: false,
                            video: {{ facingMode: {{ ideal: facingMode }}, width: {{ ideal: 1280 }}, height: {{ ideal: 720 }} }}
                        }});
                    }} catch (e) {{
                        stream = await navigator.mediaDevices.getUserMedia({{ video: true, audio: false }});
                    }}
                    photoTaken = false;
                    placeholderEl.style.display = "none";
                    photoCanvas.style.display = "none";
                    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
                    video.style.display = "block";
                    video.srcObject = stream;
                    await video.play();
                    btnCapture.style.display = "flex";
                    btnCapture.innerText = "📸 Fotoğrafı Çek ve Şişeleri Say";
                    btnCapture.style.background = "linear-gradient(135deg, #059669 0%, #10b981 100%)";
                    statusEl.innerText = "🟢 Kamera Açık (60 FPS Akıcı) — Hazır olduğunuzda '📸 Fotoğrafı Çek' butonuna basın.";
                }} catch (err) {{
                    statusEl.innerText = "⚠️ Kamera izni bekleniyor. Yukarıdaki '📷 Kamerayı Aç' butonuna dokunun.";
                }}
            }}

            async function takePhotoAndCount() {{
                if (!stream) return;
                if (!photoTaken) {{
                    photoTaken = true;
                    const vw = video.videoWidth, vh = video.videoHeight;
                    photoCanvas.width = vw; photoCanvas.height = vh;
                    overlay.width = vw; overlay.height = vh;
                    photoCtx.drawImage(video, 0, 0, vw, vh);
                    video.style.display = "none";
                    photoCanvas.style.display = "block";

                    statusEl.innerText = "⏳ Fotoğraf analiz ediliyor...";
                    btnCapture.innerText = "⏳ Sayılıyor...";

                    const scale = Math.min(INPUT_SIZE / vw, INPUT_SIZE / vh);
                    const nw = Math.round(vw * scale), nh = Math.round(vh * scale);
                    const padX = Math.floor((INPUT_SIZE - nw) / 2), padY = Math.floor((INPUT_SIZE - nh) / 2);

                    offCtx.fillStyle = "rgb(114, 114, 114)";
                    offCtx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
                    offCtx.drawImage(photoCanvas, padX, padY, nw, nh);

                    const imgData = offCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
                    const numPixels = INPUT_SIZE * INPUT_SIZE;
                    const floatData = new Float32Array(3 * numPixels);
                    for (let i = 0; i < numPixels; i++) {{
                        floatData[i] = imgData[i * 4] / 255.0;
                        floatData[numPixels + i] = imgData[i * 4 + 1] / 255.0;
                        floatData[2 * numPixels + i] = imgData[i * 4 + 2] / 255.0;
                    }}

                    const inputTensor = new ort.Tensor("float32", floatData, [1, 3, INPUT_SIZE, INPUT_SIZE]);
                    const results = await session.run({{ [session.inputNames[0]]: inputTensor }});
                    const out = results[session.outputNames[0]].data;
                    const numAnchors = 8400;
                    const rawBoxes = [];
                    for (let i = 0; i < numAnchors; i++) {{
                        const score = out[4 * numAnchors + i];
                        if (score >= CONF_THRESH) {{
                            const cx = out[0 * numAnchors + i], cy = out[1 * numAnchors + i];
                            const w = out[2 * numAnchors + i], h = out[3 * numAnchors + i];
                            const x1 = Math.max(0, (cx - w / 2 - padX) / scale);
                            const y1 = Math.max(0, (cy - h / 2 - padY) / scale);
                            const x2 = Math.min(vw, (cx + w / 2 - padX) / scale);
                            const y2 = Math.min(vh, (cy + h / 2 - padY) / scale);
                            if (x2 - x1 > 6 && y2 - y1 > 6) rawBoxes.push({{ x1, y1, x2, y2, score }});
                        }}
                    }}
                    const boxes = nms(rawBoxes, IOU_THRESH);
                    countEl.innerText = boxes.length + " Adet";

                    overlayCtx.clearRect(0, 0, vw, vh);
                    overlayCtx.strokeStyle = "#00e640";
                    overlayCtx.lineWidth = Math.max(3, Math.round(vw / 220));
                    for (const b of boxes) overlayCtx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);

                    btnCapture.innerText = "🔄 Yeni Fotoğraf Çek";
                    btnCapture.style.background = "#2563eb";
                    statusEl.innerText = "✅ Sayım Tamamlandı: Toplam " + boxes.length + " Adet Şişe Tespit Edildi.";
                }} else {{
                    photoTaken = false;
                    photoCanvas.style.display = "none";
                    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
                    video.style.display = "block";
                    countEl.innerText = "0 Adet";
                    btnCapture.innerText = "📸 Fotoğrafı Çek ve Şişeleri Say";
                    btnCapture.style.background = "linear-gradient(135deg, #059669 0%, #10b981 100%)";
                    statusEl.innerText = "🟢 Kamera Açık — Hazır olduğunuzda '📸 Fotoğrafı Çek' butonuna basın.";
                }}
            }}

            init();
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=700, scrolling=False)


def process_and_display(model, img_bgr: np.ndarray):
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
    st.image(cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)


def main():
    st.markdown(
        """
        <div class="app-header">
            <h1>🍾 Otomatik Şişe Sayım Sistemi (CNN)</h1>
            <p>Canlı kamera ile anlık sayım yapın veya fotoğraf çekip/ekleyerek şişeleri sayın.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not DEFAULT_MODEL_PATH.exists():
        st.warning("Eğitilmiş model bulunamadı.")
        return

    model = get_model(str(DEFAULT_MODEL_PATH))

    # İSTENEN SAYFA YAPISI:
    # 1. Sayfa (İlk Açılan): Canlı Sayım
    # 2. Sayfa (Sonraki Sayfa): Ayrı ayrı butonlarla Fotoğraf Çek ve Fotoğraf Ekle
    # 3. Sayfa: 83 Şişe Doğrulama
    tab_live, tab_photo, tab_verify = st.tabs([
        "🎥 Canlı Sayım",
        "📸 Fotoğraf Çek & Fotoğraf Ekle",
        "✅ 83 Şişe Doğrulama",
    ])

    # ------------------------------------------------------------------
    # 1. SAYFA (İLK AÇILAN): CANLI SAYIM
    # ------------------------------------------------------------------
    with tab_live:
        render_live_counting_page()

    # ------------------------------------------------------------------
    # 2. SAYFA (SONRAKİ SAYFA): AYRI AYRI BUTONLARLA FOTOĞRAF ÇEK VE FOTOĞRAF EKLE
    # ------------------------------------------------------------------
    with tab_photo:
        if "photo_action" not in st.session_state:
            st.session_state.photo_action = "capture"

        col_btn_capture, col_btn_upload = st.columns(2)
        with col_btn_capture:
            if st.button(
                "📸 Fotoğraf Çek",
                type="primary" if st.session_state.photo_action == "capture" else "secondary",
                use_container_width=True,
            ):
                st.session_state.photo_action = "capture"
                st.rerun()

        with col_btn_upload:
            if st.button(
                "🖼️ Fotoğraf Ekle",
                type="primary" if st.session_state.photo_action == "upload" else "secondary",
                use_container_width=True,
            ):
                st.session_state.photo_action = "upload"
                st.rerun()

        st.markdown("---")

        if st.session_state.photo_action == "capture":
            render_in_app_photo_capture()
        else:
            uploaded_file = st.file_uploader(
                "📁 Cihazınızdan veya Galeriden Bir Şişe Fotoğrafı Seçin",
                type=["jpg", "jpeg", "png", "webp"],
            )
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    process_and_display(model, img_bgr)
            else:
                st.info("Yukarıdaki kutudan bir fotoğraf seçtiğinizde şişeler otomatik olarak çerçeve içine alınıp sayılacaktır.")

    # ------------------------------------------------------------------
    # 3. SAYFA: 83 ŞİŞE DOĞRULAMA
    # ------------------------------------------------------------------
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
