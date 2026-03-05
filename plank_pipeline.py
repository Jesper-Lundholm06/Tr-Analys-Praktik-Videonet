import cv2 
import numpy as np
import os
import torch
from ultralytics import YOLO

# === KONFIGURATION ===
INPUT_DIR = "input_images"
MODEL_PATH = "best.pt"
CRACK_MODEL_PATH = "best_cracks.pt"
CORNER_MODEL_PATH = "Best_Corners.pt"
CONFIDENCE = 0.6
CRACK_CONFIDENCE = 0.2

# Hörndetektering – "bästa per zon"-logik
CORNER_SCAN_CONF = 0.01       # Låg tröskel – fånga ALLA detektioner
CORNER_MIN_CONF  = 0.1       # Golv – bästa i zonen måste överstiga detta

# Margin
MARGIN_LEFT = 50
MARGIN_RIGHT = 120
MARGIN_Y = 15

# Zonbaserad hörndetektering – vänster/höger
ZONE_MARGIN = 0.35

# Output – två separata mappar
OUTPUT_DIR_CRACKS = "defekta_plankor"
OUTPUT_DIR_CORNERS = "defekta_horn"
os.makedirs(OUTPUT_DIR_CRACKS, exist_ok=True)
os.makedirs(OUTPUT_DIR_CORNERS, exist_ok=True)

# Ladda modeller
model = YOLO(MODEL_PATH)
crack_model = YOLO(CRACK_MODEL_PATH)
corner_model = YOLO(CORNER_MODEL_PATH)
print("Modeller laddade (planka, sprickor, hörn)")


def best_conf_per_zone(boxes, crop_w, zone_margin=ZONE_MARGIN):
    """
    Hitta bästa (högsta) confidence per zon (L/R) från ALLA detektioner.
    Returnerar best-dict och dedupade unika centers för visualisering.
    """
    x_thresh_low  = crop_w * zone_margin
    x_thresh_high = crop_w * (1 - zone_margin)

    best = {
        "L": {"conf": 0.0, "center": None},
        "R": {"conf": 0.0, "center": None}
    }

    # Dedup + spåra bästa per zon
    unique_centers = []
    dist_threshold = 30

    for b in boxes:
        cx1, cy1, cx2, cy2 = map(int, b.xyxy[0])
        center_x = (cx1 + cx2) // 2
        center_y = (cy1 + cy2) // 2
        conf = float(b.conf[0])

        # Dedup
        is_new = True
        for ux, uy, _ in unique_centers:
            distance = ((center_x - ux)**2 + (center_y - uy)**2) ** 0.5
            if distance < dist_threshold:
                is_new = False
                break
        if is_new:
            unique_centers.append((center_x, center_y, conf))

        # Bästa per zon (innan dedup – vi vill ha högsta conf oavsett)
        if center_x < x_thresh_low:
            if conf > best["L"]["conf"]:
                best["L"] = {"conf": conf, "center": (center_x, center_y)}
        elif center_x > x_thresh_high:
            if conf > best["R"]["conf"]:
                best["R"] = {"conf": conf, "center": (center_x, center_y)}

    return best, unique_centers


def check_zones(best, min_conf=None):
    """Kolla om båda zoner har en detektion över minimigolvet."""
    if min_conf is None:
        min_conf = CORNER_MIN_CONF
    zones = {
        "L": best["L"]["conf"] >= min_conf,
        "R": best["R"]["conf"] >= min_conf
    }
    verdict = "GOOD" if all(zones.values()) else "DEFEKT"
    return verdict, zones


def draw_corner_zones(img, zone_margin=ZONE_MARGIN):
    """Rita vänster- och högerzoner som overlay."""
    h, w = img.shape[:2]
    x_low  = int(w * zone_margin)
    x_high = int(w * (1 - zone_margin))

    overlay = img.copy()
    alpha = 0.12
    zone_color = (255, 200, 0)

    cv2.rectangle(overlay, (0, 0), (x_low, h), zone_color, -1)
    cv2.rectangle(overlay, (x_high, 0), (w, h), zone_color, -1)

    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def draw_best_corners(analysis_img, best, crop_w, zone_margin=ZONE_MARGIN):
    """Rita bästa detektion per zon – grön om godkänd, röd om under golv."""
    BOX_SIZE = 30

    for zone_name, zone_data in best.items():
        if zone_data["center"] is None:
            continue
        center_x, center_y = zone_data["center"]
        conf = zone_data["conf"]

        dot_color = (0, 255, 0) if conf >= CORNER_MIN_CONF else (0, 0, 255)

        bx1 = center_x - BOX_SIZE // 2
        by1 = center_y - BOX_SIZE // 2
        bx2 = center_x + BOX_SIZE // 2
        by2 = center_y + BOX_SIZE // 2
        cv2.rectangle(analysis_img, (bx1, by1), (bx2, by2), dot_color, 2)
        cv2.circle(analysis_img, (center_x, center_y), 5, dot_color, -1)
        cv2.putText(analysis_img, f"{conf:.0%}", (bx2 + 3, center_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, dot_color, 1)


# Hitta alla bilder
images = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])
print(f"Hittade {len(images)} bilder")

for i, filename in enumerate(images):
    filepath = os.path.join(INPUT_DIR, filename)
    img = cv2.imread(filepath)

    if img is None:
        continue

    results = model(filepath, conf=CONFIDENCE, verbose=False)

    img_h, img_w = img.shape[:2]
    img_clean = img.copy()

    # Rita margin-zoner
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (MARGIN_LEFT, img_h), (0, 0, 255), -1)
    cv2.rectangle(overlay, (img_w - MARGIN_RIGHT, 0), (img_w, img_h), (0, 0, 255), -1)
    img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

    full_count = 0
    analysis_img = None
    verdict = "GOOD"
    n_cracks = 0
    n_corners = 0
    defect_reason = ""

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        is_full = (
            x1 > MARGIN_LEFT and
            x2 < img_w - MARGIN_RIGHT and
            y1 > MARGIN_Y and
            y2 < img_h - MARGIN_Y
        )

        if is_full:
            color = (0, 255, 0)
            label = f"HEL {conf:.0%}"
            full_count += 1

            pad = 80
            crop = img_clean[
                max(0, y1-pad):min(img_h, y2+pad),
                max(0, x1-pad):min(img_w, x2+pad)
            ]
            crop_h, crop_w = crop.shape[:2]

            analysis_img = crop.copy()
            analysis_img = draw_corner_zones(analysis_img)

            # ============================================================
            # STEG 1: Sprickanalys – om sprickor → DEFEKT direkt
            # ============================================================
            crack_results = crack_model(crop, conf=CRACK_CONFIDENCE, verbose=False)
            n_cracks = len(crack_results[0].boxes)

            for cr_box in crack_results[0].boxes:
                crx1, cry1, crx2, cry2 = map(int, cr_box.xyxy[0])
                cv2.rectangle(analysis_img, (crx1, cry1), (crx2, cry2), (0, 0, 255), 2)
                cv2.putText(analysis_img, f"crack {float(cr_box.conf[0]):.0%}",
                            (crx1, cry1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            if n_cracks > 0:
                # >>> Sprickor → DEFEKT → sparas i defekta_plankor/ <<<
                verdict = "DEFEKT"
                defect_reason = f"sprickor: {n_cracks}"

                # Best_Corners enbart för visualisering (påverkar EJ verdict)
                corner_results = corner_model(crop, conf=CORNER_SCAN_CONF, verbose=False)
                best, unique_centers = best_conf_per_zone(corner_results[0].boxes, crop_w)
                n_corners = len(unique_centers)
                draw_best_corners(analysis_img, best, crop_w)

            else:
                # ============================================================
                # STEG 2: Hörnanalys – bästa confidence per zon (V + H)
                # ============================================================
                corner_results = corner_model(crop, conf=CORNER_SCAN_CONF, verbose=False)
                best, unique_centers = best_conf_per_zone(corner_results[0].boxes, crop_w)
                n_corners = len(unique_centers)

                corner_verdict, zones = check_zones(best)

                # === DEBUG ===
                print(f"  [{filename}] L: {best['L']['conf']:.3f}, R: {best['R']['conf']:.3f} -> {corner_verdict}")
                # === SLUT DEBUG ===

                if corner_verdict == "GOOD":
                    verdict = "GOOD"
                    defect_reason = f"L:{best['L']['conf']:.0%} R:{best['R']['conf']:.0%}"
                    draw_best_corners(analysis_img, best, crop_w)
                else:
                    missing = [z for z, found in zones.items() if not found]
                    missing_str = ", ".join(missing)
                    verdict = "DEFEKT"
                    defect_reason = f"saknad sida: {missing_str} (L:{best['L']['conf']:.0%} R:{best['R']['conf']:.0%})"
                    draw_best_corners(analysis_img, best, crop_w)

            # --- Text på analysbild ---
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

            # === SPARA TILL RÄTT MAPP ===
            if verdict == "DEFEKT":
                if n_cracks > 0:
                    cv2.imwrite(os.path.join(OUTPUT_DIR_CRACKS, f"defekt_{filename}"), analysis_img)
                else:
                    cv2.imwrite(os.path.join(OUTPUT_DIR_CORNERS, f"defekt_{filename}"), analysis_img)

        else:
            color = (0, 0, 255)
            label = f"DELVIS {conf:.0%}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    n_det = len(results[0].boxes)
    if full_count > 0:
        status = "HEL PLANKA"
    elif n_det > 0:
        status = "delvis"
    else:
        status = "ingen planka"
    cv2.putText(img, f"{i+1}/{len(images)} - {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Plankanalys (Q=avsluta, SPACE=paus)", img)

    if analysis_img is not None:
        cv2.imshow("Analys: sidor + sprickor", analysis_img)
    else:
        try: cv2.destroyWindow("Analys: sidor + sprickor")
        except: pass

    key = cv2.waitKey(100)
    if key == ord('q'):
        break
    elif key == ord(' '):
        cv2.waitKey(0)

cv2.destroyAllWindows()
print("Klart!")