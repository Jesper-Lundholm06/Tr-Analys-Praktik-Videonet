import json
import os

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from pipeline_web import generate_frames, last_defects
from config import config, Config

app = FastAPI()

# === PRESET-SYSTEM ===
PRESETS_FILE = "presets.json"

PRESET_PARAMS = [
    "CONFIDENCE", "CRACK_CONFIDENCE",
    "CORNER_SCAN_CONF", "CORNER_MIN_CONF", "ZONE_MARGIN", "DIST_THRESHOLD",
    "MARGIN_LEFT", "MARGIN_RIGHT", "MARGIN_Y",
    "FRAME_DELAY",
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


INT_PARAMS = {"MARGIN_LEFT", "MARGIN_RIGHT", "MARGIN_Y", "DIST_THRESHOLD"}


def _apply_values(values: dict):
    import pipeline_web
    for k, v in values.items():
        if k in INT_PARAMS:
            v = int(round(float(v)))
        else:
            v = float(v)
        setattr(config, k, v)
    pipeline_web.current_confidence = config.CONFIDENCE


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


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


@app.post("/restart")
def restart():
    import pipeline_web
    pipeline_web.current_index = 0
    pipeline_web.is_paused = False
    return {"status": "restarted"}


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
    return {"status": "reset"}


# ==========================================
# KÄLLA: BILD / KAMERA
# ==========================================

@app.get("/get_status")
def get_status():
    """Returnera aktuellt läge och kamerastatus."""
    import pipeline_web
    return {
        "mode": pipeline_web.source_mode,
        "rtsp_url": config.RTSP_URL,
        "camera_connected": pipeline_web.camera_connected,
        "is_paused": pipeline_web.is_paused,
    }


@app.post("/set_mode/{mode}")
def set_mode(mode: str):
    """Byt mellan 'images' och 'camera'."""
    import pipeline_web
    if mode not in ("images", "camera"):
        return JSONResponse({"error": "Ogiltigt läge"}, status_code=400)

    pipeline_web.source_mode = mode
    pipeline_web.is_paused = False

    # Återställ bildindex vid byte till bilder
    if mode == "images":
        pipeline_web.current_index = 0

    return {"status": "ok", "mode": mode}


@app.post("/set_rtsp_url")
async def set_rtsp_url(data: dict):
    """Uppdatera RTSP-URL. Om kameran är aktiv, stoppa och starta om."""
    import pipeline_web
    url = data.get("url", "").strip()
    if not url:
        return JSONResponse({"error": "URL får inte vara tom"}, status_code=400)

    config.RTSP_URL = url

    # Om vi är i kameraläge, starta om kameran med ny URL
    if pipeline_web.source_mode == "camera":
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app",
                host="127.0.0.1",
                port=8011,
                reload=True)