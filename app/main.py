import glob as file_glob
import json
import os
import sys
import logging
from datetime import datetime
from pathlib import Path


def resource_path(relative_path):
    """Returnera korrekt path både i dev och PyInstaller exe."""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# ==========================
# LOGG-SETUP
# ==========================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

_log_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# --- system.log ---
system_log = logging.getLogger("system")
system_log.setLevel(logging.INFO)
_sys_handler = logging.FileHandler(os.path.join(LOG_DIR, "system.log"), encoding="utf-8")
_sys_handler.setFormatter(_log_fmt)
system_log.addHandler(_sys_handler)

# --- camera.log ---
camera_log = logging.getLogger("camera")
camera_log.setLevel(logging.INFO)
_cam_handler = logging.FileHandler(os.path.join(LOG_DIR, "camera.log"), encoding="utf-8")
_cam_handler.setFormatter(_log_fmt)
camera_log.addHandler(_cam_handler)

system_log.info("System started")

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from pipeline_web import generate_frames, last_defects, startup_time
from config import config, Config

app = FastAPI()

# === PRESET-SYSTEM ===
PRESETS_FILE = resource_path("presets.json")

PRESET_PARAMS = [
    "CONFIDENCE", "CRACK_CONFIDENCE",
    "CORNER_SCAN_CONF", "CORNER_MIN_CONF", "ZONE_MARGIN", "DIST_THRESHOLD",
    "MARGIN_LEFT", "MARGIN_RIGHT", "MARGIN_Y",
    "PAD",
]

DEFAULTS = {p: getattr(Config, p) for p in PRESET_PARAMS}


def _load_presets() -> dict:
    if os.path.exists(PRESETS_FILE):
        with open(PRESETS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_presets(presets: dict):
    with open(PRESETS_FILE, "w", encoding="utf-8") as f:
        json.dump(presets, f, indent=2, ensure_ascii=False)


def _current_values() -> dict:
    return {p: getattr(config, p) for p in PRESET_PARAMS}


INT_PARAMS = {"MARGIN_LEFT", "MARGIN_RIGHT", "MARGIN_Y", "DIST_THRESHOLD", "PAD"}


def _apply_values(values: dict):
    import pipeline_web
    for k, v in values.items():
        if k in INT_PARAMS:
            v = int(round(float(v)))
        else:
            v = float(v)
        setattr(config, k, v)
    pipeline_web.current_confidence = config.CONFIDENCE


# === STATISKA FILER ===
# I frozen-läge: bredvid exe:n. I dev: relativa sökvägar.
_static_dir = resource_path("static")
_templates_dir = resource_path("templates")

app.mount("/static", StaticFiles(directory=_static_dir), name="static")
templates = Jinja2Templates(directory=_templates_dir)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/defects")
def get_defects():
    return JSONResponse(
        {"images": [d.hex() for d in last_defects]}
    )


@app.post("/update_config")
async def update_config(data: dict):
    param = data["param"]
    value = float(data["value"])
    if param in INT_PARAMS:
        value = int(round(value))
    setattr(config, param, value)
    return {"status": "ok"}


@app.post("/pause")
def pause():
    import pipeline_web
    pipeline_web.is_paused = True
    return {"status": "paused"}


@app.post("/resume")
def resume():
    import pipeline_web
    pipeline_web.is_paused = False
    return {"status": "running"}


@app.post("/set_confidence/{value}")
def set_confidence(value: float):
    import pipeline_web
    pipeline_web.current_confidence = value
    config.CONFIDENCE = value
    return {"new_confidence": value}


# ==========================================
# HÄLSOKONTROLL
# ==========================================

@app.get("/health")
def health():
    """Hälsokontroll — visar om systemet lever."""
    import pipeline_web
    now = datetime.now()
    uptime = now - startup_time
    hours, remainder = divmod(int(uptime.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    last_frame_age = None
    if pipeline_web.last_camera_time > 0:
        last_frame_age = round(
            now.timestamp() - pipeline_web.last_camera_time, 1
        )

    return {
        "status": "ok" if pipeline_web.camera_connected else "degraded",
        "uptime": f"{hours}h {minutes}m {seconds}s",
        "camera_connected": pipeline_web.camera_connected,
        "camera_reconnects": pipeline_web.camera_reconnect_count,
        "last_frame_age_s": last_frame_age,
        "rtsp_url": config.RTSP_URL,
        "total_planks": pipeline_web.stats["total_planks"],
        "is_paused": pipeline_web.is_paused,
    }


# ==========================================
# STATISTIK
# ==========================================

@app.get("/stats")
def get_stats():
    """Returnera live-statistik."""
    import pipeline_web
    s = pipeline_web.stats.copy()
    total = s["total_planks"]
    defekt_total = s["defekt_cracks"] + s["defekt_corners"]
    s["defekt_total"] = defekt_total
    s["defekt_pct"] = round((defekt_total / total * 100), 1) if total > 0 else 0.0
    s["good_pct"] = round((s["good_planks"] / total * 100), 1) if total > 0 else 0.0
    return s


@app.post("/reset_stats")
def reset_stats():
    """Nollställ statistiken."""
    import pipeline_web
    pipeline_web.reset_stats()
    system_log.info("Stats reset by user")
    return {"status": "reset"}


# ==========================================
# KAMERA
# ==========================================

@app.get("/get_status")
def get_status():
    """Returnera aktuellt kamerastatus."""
    import pipeline_web
    return {
        "rtsp_url": config.RTSP_URL,
        "camera_connected": pipeline_web.camera_connected,
        "is_paused": pipeline_web.is_paused,
    }


@app.post("/set_rtsp_url")
async def set_rtsp_url(data: dict):
    """Uppdatera RTSP-URL och starta om kameran."""
    import pipeline_web
    url = data.get("url", "").strip()
    if not url:
        return JSONResponse({"error": "URL får inte vara tom"}, status_code=400)

    config.RTSP_URL = url
    system_log.info("RTSP URL changed to: %s", url)
    pipeline_web.stop_camera()

    import time
    time.sleep(0.5)
    pipeline_web.start_camera()

    return {"status": "ok", "url": url}


# ==========================================
# PRESET-ENDPOINTS
# ==========================================

@app.get("/presets")
def list_presets():
    return {
        "presets": _load_presets(),
        "current": _current_values(),
        "defaults": DEFAULTS,
    }


@app.post("/presets/{name}")
def save_preset(name: str):
    presets = _load_presets()
    presets[name] = _current_values()
    _save_presets(presets)
    return {"status": "saved", "name": name}


@app.post("/load_preset/{name}")
def load_preset(name: str):
    presets = _load_presets()
    if name not in presets:
        return JSONResponse({"error": "Preset finns inte"}, status_code=404)
    _apply_values(presets[name])
    system_log.info("Preset loaded: %s", name)
    return {"status": "loaded", "name": name, "values": presets[name]}


@app.delete("/presets/{name}")
def delete_preset(name: str):
    presets = _load_presets()
    if name in presets:
        del presets[name]
        _save_presets(presets)
    return {"status": "deleted", "name": name}


@app.post("/reset_defaults")
def reset_defaults():
    _apply_values(DEFAULTS)
    return {"status": "reset", "values": DEFAULTS}


# ==========================================
# BILDANALYS (mapp-läge)
# ==========================================

@app.post("/process_image")
async def process_image(image: UploadFile = File(...)):
    """Kör detektionspipelinen på en uppladdad bild (mapp-läge)."""
    import numpy as np
    import cv2
    from pipeline_web import process_plank, draw_margins

    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if raw is None:
        return JSONResponse({"error": "Ogiltig bild"}, status_code=400)

    img_clean = raw.copy()
    img = draw_margins(raw)
    annotated, _status = process_plank(img, img_clean, image.filename or "folder.jpg")

    _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


# ==========================================
# BILDMAPP-LÄGE
# ==========================================

_SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

_folder_state: dict = {
    "mode": "live",       # "live" | "folder"
    "folder_path": "",
    "images": [],         # sorted list of absolute paths
}


@app.post("/set_mode")
async def set_mode(data: dict):
    mode = data.get("mode", "live")
    if mode not in ("live", "folder"):
        return JSONResponse({"error": "Ogiltigt läge"}, status_code=400)
    _folder_state["mode"] = mode
    system_log.info("Mode changed to: %s", mode)
    return {"status": "ok", "mode": mode}


@app.post("/set_folder")
async def set_folder(data: dict):
    path = data.get("path", "").strip()
    if not path:
        return JSONResponse({"error": "Sökvägen är tom"}, status_code=400)
    if not os.path.isdir(path):
        return JSONResponse({"error": f"Mappen finns inte: {path}"}, status_code=400)

    images = []
    for ext in _SUPPORTED_EXT:
        images.extend(file_glob.glob(os.path.join(path, f"*{ext}")))
        images.extend(file_glob.glob(os.path.join(path, f"*{ext.upper()}")))
    images = sorted(set(images))

    _folder_state["folder_path"] = path
    _folder_state["images"] = images
    system_log.info("Folder loaded: %s (%d images)", path, len(images))
    return {"status": "ok", "count": len(images), "path": path}


@app.get("/folder_image/{index}")
def get_folder_image(index: int):
    images = _folder_state["images"]
    if not images:
        return JSONResponse({"error": "Inga bilder laddade"}, status_code=404)
    img_path = images[index % len(images)]
    try:
        content = Path(img_path).read_bytes()
        ext = Path(img_path).suffix.lower()
        media_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
        return Response(content=content, media_type=media_type)
    except OSError:
        return JSONResponse({"error": "Kunde inte läsa bild"}, status_code=500)


@app.get("/folder_status")
def get_folder_status():
    return {
        "mode": _folder_state["mode"],
        "folder_path": _folder_state["folder_path"],
        "image_count": len(_folder_state["images"]),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app",
                host=config.HOST,
                port=config.PORT,
                reload=False)