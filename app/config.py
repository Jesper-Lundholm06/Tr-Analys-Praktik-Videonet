import os
import sys
import configparser


def _base_dir():
    """
    Returnera bas-sökvägen för modeller och resurser.
    - Frozen (PyInstaller): mappen där service.exe ligger
    - Dev: mappen där config.py ligger
    """
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


BASE_DIR = _base_dir()

# ============================================================
# Läs settings.ini (ligger bredvid exe:n / config.py)
# ============================================================
_ini = configparser.ConfigParser()
_ini_path = os.path.join(BASE_DIR, "settings.ini")
_ini.read(_ini_path, encoding="utf-8")


def _ini_get(section, key, fallback=""):
    """Hämta ett värde från settings.ini med fallback."""
    return _ini.get(section, key, fallback=fallback).strip()


# ============================================================
# DATA_DIR – dit alla skrivbara filer hamnar
# ============================================================
_data_dir_raw = _ini_get("paths", "data_dir", "")
if _data_dir_raw:
    DATA_DIR = _data_dir_raw
else:
    DATA_DIR = BASE_DIR

# Skapa data-mappen om den inte finns
os.makedirs(DATA_DIR, exist_ok=True)


def _find_model(name):
    """Hitta modellfil: bredvid .exe först, sen i _internal."""
    if getattr(sys, 'frozen', False):
        p = os.path.join(BASE_DIR, name)
        if os.path.isfile(p):
            return p
        return os.path.join(sys._MEIPASS, name)
    return os.path.join("..", name)


class Config:
    # === MODELLER ===
    MODEL_PATH = _find_model("best.pt")
    CRACK_MODEL_PATH = _find_model("best_cracks.pt")
    CORNER_MODEL_PATH = _find_model("Best_Corners.pt")

    # === KAMERA ===
    RTSP_URL = _ini_get("camera", "rtsp_url", "")
    CAMERA_RECONNECT_BASE = 2
    CAMERA_RECONNECT_MAX = 30
    CAMERA_TIMEOUT = 5

    # === OUTPUT (under DATA_DIR) ===
    OUTPUT_DIR_CRACKS = os.path.join(DATA_DIR, "defekta_plankor")
    OUTPUT_DIR_CORNERS = os.path.join(DATA_DIR, "defekta_horn")

    # === JUSTERBARA PARAMETRAR ===
    CONFIDENCE = 0.8
    CRACK_CONFIDENCE = 0.2

    # Hörndetektering – 2-zon (V/H) med bästa-per-zon-logik
    CORNER_SCAN_CONF = 0.01
    CORNER_MIN_CONF  = 0.1
    ZONE_MARGIN = 0.35
    DIST_THRESHOLD = 30

    # Marginaler
    MARGIN_LEFT = 50
    MARGIN_RIGHT = 120
    MARGIN_Y = 15

    # Crop-padding
    PAD = 80

    # === LOGGNING (under DATA_DIR) ===
    LOG_DIR = os.path.join(DATA_DIR, "logs")
    LOG_ROTATE_DAYS = 7

    # === HASTIGHET ===
    FRAME_DELAY = 0.3

    # === WEBSERVER (från settings.ini) ===
    HOST = _ini_get("server", "host", "0.0.0.0")
    PORT = int(_ini_get("server", "port", "8014"))


config = Config()