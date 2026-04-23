# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PlankInspection** is a Windows Service application with a web control panel that uses YOLO-based AI to detect defects (cracks and corner damage) in wooden planks from an industrial RTSP camera feed.

## Commands

**Development (no build required):**
```bash
cd app
python main.py          # FastAPI server on http://0.0.0.0:8014
python service.py debug # Run as foreground service for testing
```

**Batch/offline testing:**
```bash
python plank_pipeline.py  # Process images in input_images/ directories
```

**Build Windows executable:**
```bash
cd app
./build.bat  # Installs deps, runs PyInstaller, optionally creates Inno Setup installer
             # Output: dist/PlankInspection/
```

**Windows Service management (after install):**
```bash
service.exe install | start | stop | remove
net start PlankInspection
```

## Architecture

Three-tier design:

**1. Windows Service Layer** — [app/service.py](app/service.py)
Wraps the FastAPI app as a Windows Service (via pywin32). Handles service lifecycle and sets working directory correctly for the PyInstaller frozen executable.

**2. Web Server** — [app/main.py](app/main.py)
FastAPI app exposing REST endpoints for: MJPEG video stream, live statistics, parameter tuning, preset management (load/save/delete), camera URL updates, and pause/resume.

**3. Detection Pipeline** — [app/pipeline_web.py](app/pipeline_web.py)
Runs in a background daemon thread. Pulls RTSP frames, then chains three YOLO models:
1. `best.pt` — detects the plank itself (bounding box + crop)
2. `best_cracks.pt` — crack detection on the cropped plank
3. `Best_Corners.pt` — corner defect detection, split into left/right zones by `ZONE_MARGIN`

Camera reconnects with exponential backoff (2s → 30s max). Thread-safe frame handoff via `camera_lock`.

**Configuration:**
- [app/config.py](app/config.py) — runtime config object; reads from `settings.ini`
- [app/settings.ini](app/settings.ini) — deployment settings (RTSP URL, port, data paths)
- [app/presets.json](app/presets.json) — named parameter snapshots for quick switching

All detection thresholds (confidence, zone margins, crop padding, frame delay) are adjustable at runtime via the web dashboard or `/update_config` API without restarting.

## Key Details

- **Language**: Python 3; UI text and comments are in Swedish
- **ML runtime**: CPU-only PyTorch (no CUDA) — avoids large CUDA dependency in installer
- **Models** (`.pt` files): stored at repo root and referenced from `settings.ini`; the three active models are `best.pt`, `best_cracks.pt`, `Best_Corners.pt`
- **Logs**: daily-rotating files in `app/logs/` (`system.log`, `camera.log`, `pipeline_*.log`), 7-day retention
- **Defect images**: saved to the output path configured in `settings.ini`; browsable via `/defects` endpoint
- **PyInstaller build**: `build.spec` bundles models and static files; `rthook_torch.py` is a runtime hook required for frozen torch to work correctly
- **Windows-only**: Windows Service integration means the service-layer code will not run on Linux/macOS