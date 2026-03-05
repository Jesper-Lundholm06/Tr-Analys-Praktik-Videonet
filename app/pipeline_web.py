import cv2
import numpy as np
import os
import time
import threading
from datetime import datetime
from ultralytics import YOLO
from collections import deque
from config import config

# ==========================
# MODELLER
# ==========================

model = YOLO(config.MODEL_PATH)
crack_model = YOLO(config.CRACK_MODEL_PATH)
corner_model = YOLO(config.CORNER_MODEL_PATH)

print("Modeller laddade")

# ==========================
# GLOBALT TILLSTÅND
# ==========================

is_paused = False
current_index = 0
image_list = []

current_confidence = config.CONFIDENCE
source_mode = config.SOURCE_MODE  # "images" eller "camera"

last_defects = deque(maxlen=4)

# === STATISTIK ===
stats = {
    "total_planks": 0,
    "good_planks": 0,
    "defekt_cracks": 0,
    "defekt_corners": 0,
    "partial": 0,
    "no_plank": 0,
}


def reset_stats():
    """Nollställ statistik + skriv sammanfattning till logg."""
    log_stats_summary()
    for k in stats:
        stats[k] = 0


def log_stats_summary():
    """Skriv en sammanfattningsrad till loggfilen."""
    s = stats
    total = s["total_planks"]
    if total == 0:
        return
    defekt = s["defekt_cracks"] + s["defekt_corners"]
    pct = (defekt / total * 100) if total > 0 else 0
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"\n--- STATISTIK [{ts}] ---\n"
        f"  Plankor totalt: {total}\n"
        f"  Godkända:       {s['good_planks']}\n"
        f"  Defekta:        {defekt}  ({pct:.1f}%)\n"
        f"    - Sprickor:   {s['defekt_cracks']}\n"
        f"    - Hörn:       {s['defekt_corners']}\n"
        f"  Delvis:         {s['partial']}\n"
        f"  Ingen planka:   {s['no_plank']}\n"
        f"{'='*40}\n"
    )
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)

# === KAMERA-TILLSTÅND ===
camera_frame = None
camera_lock = threading.Lock()
camera_thread = None
camera_running = False
camera_connected = False
last_camera_time = 0.0

os.makedirs(config.OUTPUT_DIR_CRACKS, exist_ok=True)
os.makedirs(config.OUTPUT_DIR_CORNERS, exist_ok=True)
os.makedirs(config.LOG_DIR, exist_ok=True)

LOG_PATH = os.path.join(config.LOG_DIR, config.LOG_FILE)


def log_entry(filename, status, plank_conf=None, l_conf=None, r_conf=None,
              n_cracks=0, crack_conf_max=None, verdict=None, reason=""):
    """Skriv en rad till loggfilen."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if verdict:
        l_str = f"{l_conf:.3f}" if l_conf is not None else "—"
        r_str = f"{r_conf:.3f}" if r_conf is not None else "—"
        p_str = f"{plank_conf:.2f}" if plank_conf is not None else "—"
        cr_str = f"{n_cracks}"
        if crack_conf_max is not None and n_cracks > 0:
            cr_str += f" (max {crack_conf_max:.2f})"
        line = (f"[{ts}] [{filename}] "
                f"Plank: {p_str} | L: {l_str}, R: {r_str} | "
                f"Cracks: {cr_str} | -> {verdict}"
                f"{' (' + reason + ')' if reason else ''}")
    else:
        p_str = f"{plank_conf:.2f}" if plank_conf is not None else "—"
        line = f"[{ts}] [{filename}] {status} (conf: {p_str})"

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ==========================
# KAMERA-TRÅD
# ==========================

def camera_reader():
    """Läser frames från RTSP-kameran i en separat tråd."""
    global camera_frame, camera_running, camera_connected, last_camera_time

    RECONNECT_DELAY = 3
    TIMEOUT = 5
    cap = None

    while camera_running:
        # Anslut om inte ansluten
        if cap is None or not cap.isOpened():
            camera_connected = False
            rtsp_url = config.RTSP_URL
            print(f"📷 Ansluter till kamera: {rtsp_url}")

            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(1)

            if not cap.isOpened():
                print("❌ Kunde inte ansluta kamera")
                cap = None
                time.sleep(RECONNECT_DELAY)
                continue

            print("✅ Kamera ansluten")
            camera_connected = True
            last_camera_time = time.time()

        # Läs frame
        ret, new_frame = cap.read()
        if ret:
            with camera_lock:
                camera_frame = new_frame
            last_camera_time = time.time()
            camera_connected = True
        else:
            time.sleep(0.05)

        # Watchdog – ingen frame på TIMEOUT sekunder → reconnect
        if time.time() - last_camera_time > TIMEOUT:
            print("⚠️ Kamera timeout → reconnect")
            camera_connected = False
            if cap is not None:
                cap.release()
                cap = None
            time.sleep(1)

    # Rensa vid stopp
    if cap is not None:
        cap.release()
    camera_connected = False
    print("📷 Kameratråd stoppad")


def start_camera():
    """Starta kameraläsartråden."""
    global camera_thread, camera_running
    if camera_running:
        return  # Redan igång
    camera_running = True
    camera_thread = threading.Thread(target=camera_reader, daemon=True)
    camera_thread.start()
    print("📷 Kameratråd startad")


def stop_camera():
    """Stoppa kameraläsartråden."""
    global camera_running, camera_frame, camera_connected
    camera_running = False
    camera_connected = False
    camera_frame = None
    print("📷 Stoppar kameratråd...")


# ==========================
# HÖRNZON-KONTROLL (2-zon: V/H)
# ==========================

def best_conf_per_zone(boxes, crop_w):
    zone_margin = config.ZONE_MARGIN
    x_thresh_low = crop_w * zone_margin
    x_thresh_high = crop_w * (1 - zone_margin)

    best = {
        "L": {"conf": 0.0, "center": None},
        "R": {"conf": 0.0, "center": None}
    }

    unique_centers = []
    dist_threshold = config.DIST_THRESHOLD

    for b in boxes:
        cx1, cy1, cx2, cy2 = map(int, b.xyxy[0])
        center_x = (cx1 + cx2) // 2
        center_y = (cy1 + cy2) // 2
        conf = float(b.conf[0])

        is_new = True
        for ux, uy, _ in unique_centers:
            distance = ((center_x - ux)**2 + (center_y - uy)**2) ** 0.5
            if distance < dist_threshold:
                is_new = False
                break
        if is_new:
            unique_centers.append((center_x, center_y, conf))

        if center_x < x_thresh_low:
            if conf > best["L"]["conf"]:
                best["L"] = {"conf": conf, "center": (center_x, center_y)}
        elif center_x > x_thresh_high:
            if conf > best["R"]["conf"]:
                best["R"] = {"conf": conf, "center": (center_x, center_y)}

    return best, unique_centers


def check_zones(best):
    min_conf = config.CORNER_MIN_CONF
    zones = {
        "L": best["L"]["conf"] >= min_conf,
        "R": best["R"]["conf"] >= min_conf
    }
    verdict = "GOOD" if all(zones.values()) else "DEFEKT"
    return verdict, zones


def draw_corner_zones(img):
    h, w = img.shape[:2]
    zone_margin = config.ZONE_MARGIN
    x_low = int(w * zone_margin)
    x_high = int(w * (1 - zone_margin))

    overlay = img.copy()
    alpha = 0.12
    zone_color = (255, 200, 0)

    cv2.rectangle(overlay, (0, 0), (x_low, h), zone_color, -1)
    cv2.rectangle(overlay, (x_high, 0), (w, h), zone_color, -1)

    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def draw_best_corners(analysis_img, best):
    BOX_SIZE = 30
    min_conf = config.CORNER_MIN_CONF

    for zone_name, zone_data in best.items():
        if zone_data["center"] is None:
            continue
        center_x, center_y = zone_data["center"]
        conf = zone_data["conf"]

        dot_color = (0, 255, 0) if conf >= min_conf else (0, 0, 255)

        bx1 = center_x - BOX_SIZE // 2
        by1 = center_y - BOX_SIZE // 2
        bx2 = center_x + BOX_SIZE // 2
        by2 = center_y + BOX_SIZE // 2
        cv2.rectangle(analysis_img, (bx1, by1), (bx2, by2), dot_color, 2)
        cv2.circle(analysis_img, (center_x, center_y), 5, dot_color, -1)
        cv2.putText(analysis_img, f"{conf:.0%}", (bx2 + 3, center_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, dot_color, 1)


# ==========================
# PLANK-ANALYS (gemensam för båda lägen)
# ==========================

def process_plank(img, img_clean, filename="live"):
    """
    Kör hela plank-pipelinen på en bild.
    Returnerar (annotated_img, status_text).
    img = bild att rita på (med margin-overlay)
    img_clean = ren kopia för crop
    """
    global current_confidence

    img_h, img_w = img.shape[:2]

    results = model(img_clean, conf=current_confidence, verbose=False)

    full_count = 0
    status = "INGEN"

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        ml = int(config.MARGIN_LEFT)
        mr = int(config.MARGIN_RIGHT)
        my = int(config.MARGIN_Y)

        is_full = (
            x1 > ml and
            x2 < img_w - mr and
            y1 > my and
            y2 < img_h - my
        )

        if is_full:
            full_count += 1

            pad = config.PAD
            crop = img_clean[
                max(0, y1 - pad):min(img_h, y2 + pad),
                max(0, x1 - pad):min(img_w, x2 + pad)
            ]
            crop_h, crop_w = crop.shape[:2]

            analysis_img = crop.copy()
            analysis_img = draw_corner_zones(analysis_img)

            verdict = "GOOD"
            n_cracks = 0
            n_corners = 0
            defect_reason = ""

            # STEG 1: Sprickanalys
            crack_results = crack_model(crop, conf=config.CRACK_CONFIDENCE, verbose=False)
            n_cracks = len(crack_results[0].boxes)

            for cr_box in crack_results[0].boxes:
                crx1, cry1, crx2, cry2 = map(int, cr_box.xyxy[0])
                cv2.rectangle(analysis_img, (crx1, cry1), (crx2, cry2), (0, 0, 255), 2)
                cv2.putText(analysis_img, f"crack {float(cr_box.conf[0]):.0%}",
                            (crx1, cry1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            if n_cracks > 0:
                verdict = "DEFEKT"
                defect_reason = f"sprickor: {n_cracks}"

                corner_results = corner_model(crop, conf=config.CORNER_SCAN_CONF, verbose=False)
                best, unique_centers = best_conf_per_zone(corner_results[0].boxes, crop_w)
                n_corners = len(unique_centers)
                draw_best_corners(analysis_img, best)
            else:
                # STEG 2: Hörnanalys
                corner_results = corner_model(crop, conf=config.CORNER_SCAN_CONF, verbose=False)
                best, unique_centers = best_conf_per_zone(corner_results[0].boxes, crop_w)
                n_corners = len(unique_centers)

                corner_verdict, zones = check_zones(best)

                if corner_verdict == "GOOD":
                    verdict = "GOOD"
                    defect_reason = f"L:{best['L']['conf']:.0%} R:{best['R']['conf']:.0%}"
                else:
                    missing = [z for z, found in zones.items() if not found]
                    missing_str = ", ".join(missing)
                    verdict = "DEFEKT"
                    defect_reason = f"saknad sida: {missing_str} (L:{best['L']['conf']:.0%} R:{best['R']['conf']:.0%})"

                draw_best_corners(analysis_img, best)

            # Text på analysbild
            c_verdict = (0, 0, 255) if verdict == "DEFEKT" else (0, 255, 0)
            c_crack = (0, 255, 0) if n_cracks == 0 else (0, 0, 255)

            cv2.putText(analysis_img, f"Resultat: {verdict}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c_verdict, 2)
            cv2.putText(analysis_img, f"Sprickor: {n_cracks}", (5, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c_crack, 2)
            cv2.putText(analysis_img, f"Sidor: {n_corners}", (5, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(analysis_img, defect_reason, (5, analysis_img.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

            # Spara defekter
            if verdict == "DEFEKT":
                if n_cracks > 0:
                    save_path = os.path.join(config.OUTPUT_DIR_CRACKS, f"defekt_{filename}")
                else:
                    save_path = os.path.join(config.OUTPUT_DIR_CORNERS, f"defekt_{filename}")

                cv2.imwrite(save_path, crop)

                _, jpg_bytes = cv2.imencode('.jpg', analysis_img)
                last_defects.append(jpg_bytes.tobytes())

            # Färga plank-rutan + STATISTIK
            stats["total_planks"] += 1
            if verdict == "DEFEKT":
                if n_cracks > 0:
                    stats["defekt_cracks"] += 1
                else:
                    stats["defekt_corners"] += 1
                color = (0, 165, 255)
                label = f"DEFEKT ({defect_reason})"
            else:
                stats["good_planks"] += 1
                color = (0, 255, 0)
                label = f"HEL OK {conf:.0%}"

            # Logg
            crack_conf_max = None
            if n_cracks > 0:
                crack_conf_max = max(float(cb.conf[0]) for cb in crack_results[0].boxes)
            log_entry(
                filename=filename, status="HEL",
                plank_conf=conf,
                l_conf=best["L"]["conf"], r_conf=best["R"]["conf"],
                n_cracks=n_cracks, crack_conf_max=crack_conf_max,
                verdict=verdict, reason=defect_reason
            )
        else:
            stats["partial"] += 1
            color = (0, 0, 255)
            label = f"DELVIS {conf:.0%}"
            log_entry(filename=filename, status="DELVIS", plank_conf=conf)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if full_count > 0:
        status = "HEL PLANKA"
    elif len(results[0].boxes) > 0:
        status = "DELVIS"
    else:
        status = "INGEN"
        stats["no_plank"] += 1
        log_entry(filename=filename, status="INGEN")

    return img, status


# ==========================
# HJÄLPFUNKTION: Lägg på margin-overlay
# ==========================

def draw_margins(img):
    """Rita margin-zoner (röda kanter) på bilden."""
    img_h, img_w = img.shape[:2]
    ml = int(config.MARGIN_LEFT)
    mr = int(config.MARGIN_RIGHT)

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (ml, img_h), (0, 0, 255), -1)
    cv2.rectangle(overlay, (img_w - mr, 0), (img_w, img_h), (0, 0, 255), -1)
    return cv2.addWeighted(overlay, 0.3, img, 0.7, 0)


# ==========================
# SKAPA ERROR-FRAME
# ==========================

def create_error_frame(message, sub=""):
    """Skapar en svart bild med felmeddelande."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, message, (30, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    if sub:
        cv2.putText(img, sub, (30, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return img


# ==========================
# VIDEO STREAM – BILDLÄGE
# ==========================

def generate_frames_images():
    """Generator för bildläge (bläddrar genom mapp)."""
    global is_paused, current_index, image_list

    image_list = sorted([
        f for f in os.listdir(config.INPUT_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    total_images = len(image_list)

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n{'='*70}\n")
        f.write(f"  Pipeline startad (BILDER): {ts} | {total_images} bilder\n")
        f.write(f"{'='*70}\n")

    while True:
        # Kolla om vi bytt till kameraläge
        if source_mode != "images":
            return

        if total_images == 0:
            time.sleep(0.5)
            continue

        if is_paused:
            time.sleep(0.1)
            continue

        if current_index >= total_images:
            current_index = total_images - 1
            is_paused = True
            time.sleep(0.1)
            continue

        filename = image_list[current_index]
        filepath = os.path.join(config.INPUT_DIR, filename)

        img = cv2.imread(filepath)
        if img is None:
            current_index += 1
            continue

        img_clean = img.copy()
        img = draw_margins(img)

        img, status = process_plank(img, img_clean, filename)

        cv2.putText(img,
                    f"[BILDER] {current_index + 1}/{total_images} - {status}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        _, frame = cv2.imencode('.jpg', img)
        current_index += 1

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame.tobytes() +
               b'\r\n')

        time.sleep(config.FRAME_DELAY)


# ==========================
# VIDEO STREAM – KAMERALÄGE
# ==========================

def generate_frames_camera():
    """Generator för kameraläge (live RTSP)."""
    global is_paused, camera_frame

    start_camera()

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n{'='*70}\n")
        f.write(f"  Pipeline startad (KAMERA): {ts}\n")
        f.write(f"{'='*70}\n")

    last_processed = None
    frame_counter = 0

    while True:
        # Kolla om vi bytt till bildläge
        if source_mode != "camera":
            stop_camera()
            return

        if is_paused:
            # Visa senaste bearbetade bilden
            if last_processed is not None:
                _, jpg = cv2.imencode('.jpg', last_processed)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       jpg.tobytes() +
                       b'\r\n')
            time.sleep(0.1)
            continue

        # Hämta senaste kamerabild
        with camera_lock:
            raw = camera_frame.copy() if camera_frame is not None else None

        if raw is None:
            err = create_error_frame("Ansluter till kamera...", config.RTSP_URL)
            _, jpg = cv2.imencode('.jpg', err)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   jpg.tobytes() +
                   b'\r\n')
            time.sleep(0.5)
            continue

        frame_counter += 1
        ts_label = datetime.now().strftime("%H:%M:%S")

        img_clean = raw.copy()
        img = draw_margins(raw)

        img, status = process_plank(img, img_clean,
                                    f"live_{ts_label.replace(':', '')}_{frame_counter}.jpg")

        # Status-text
        conn_text = "LIVE" if camera_connected else "RECONNECTING"
        cv2.putText(img,
                    f"[KAMERA {conn_text}] {ts_label} - {status}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        last_processed = img.copy()

        _, jpg = cv2.imencode('.jpg', img)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               jpg.tobytes() +
               b'\r\n')

        time.sleep(config.FRAME_DELAY)


# ==========================
# HUVUD-GENERATOR (router)
# ==========================

def generate_frames():
    """Väljer rätt generator baserat på source_mode."""
    while True:
        if source_mode == "camera":
            yield from generate_frames_camera()
        else:
            yield from generate_frames_images()
        # Liten paus vid lägesbyte
        time.sleep(0.2)