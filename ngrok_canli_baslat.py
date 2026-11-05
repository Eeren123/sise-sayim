import subprocess
import sys
import time
import urllib.request
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def get_existing_ngrok_url():
    try:
        with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for t in data.get("tunnels", []):
                if t.get("public_url", "").startswith("https://"):
                    return t["public_url"]
    except Exception:
        return None
    return None


def start_streamlit_and_ngrok(port: int = 8501):
    print("============================================================")
    print("[1/2] Streamlit Sunucusu Baslatiliyor (Port: 8501)...")
    st_proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", str(port),
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
        ],
        cwd=str(BASE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)

    public_url = get_existing_ngrok_url()
    ngrok_proc = None
    if not public_url:
        print("[2/2] Ngrok HTTPS Tuneli Aciliyor...")
        ngrok_proc = subprocess.Popen(
            ["ngrok", "http", str(port)],
            cwd=str(BASE_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(15):
            time.sleep(1)
            public_url = get_existing_ngrok_url()
            if public_url:
                break

    if public_url:
        (BASE_DIR / "canli_ngrok_linki.txt").write_text(public_url + "\n", encoding="utf-8")
        print("\n============================================================")
        print("[HAZIR] CANLI NGROK HTTPS LINKINIZ:")
        print(f"  -->  {public_url}")
        print("============================================================")
        print("Bu linki arkadasiniza gonderebilirsiniz (HTTPS oldugu icin telefon kamerasi dogrudan calisir).")
        print("Durdurmak icin bu pencereyi kapatin veya Ctrl+C tuslarina basin.\n")
    else:
        print("[UYARI] Ngrok tunel URL'si alinamadi. Lutfen 'ngrok http 8501' komutunu kontrol edin.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nSunucu kapatiliyor...")
        if ngrok_proc:
            ngrok_proc.terminate()
        st_proc.terminate()


if __name__ == "__main__":
    start_streamlit_and_ngrok()
