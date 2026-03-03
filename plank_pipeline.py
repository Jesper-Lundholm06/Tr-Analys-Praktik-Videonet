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
CONFIDENCE = 0.8
CRACK_CONFIDENCE = 0.2
CORNER_CONFIDENCE = 0.2

# Margin
MARGIN_LEFT = 50
MARGIN_RIGHT = 120
MARGIN_Y = 15

# Zonbaserad hörndetektering – hur stor del av bilden utgör varje zon (0.0–1.0)
# Hörn måste hamna inom ZONE_MARGIN från kanten för att räknas som ett giltigt hörn
ZONE_MARGIN = 0.35  # 35% av croppens bredd/höjd räknas som "hörn-zon"

# Output
OUTPUT_DIR = "defekta_plankor"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ladda modeller
model = YOLO(MODEL_PATH)
crack_model = YOLO(CRACK_MODEL_PATH)
corner_model = YOLO(CORNER_MODEL_PATH)
print("Modeller laddade")


def check_corners_by_zone(unique_centers, crop_w, crop_h, zone_margin=ZONE_MARGIN):
    """
    Delar upp croppen i 4 zoner (kvadranter) och kontrollerar om det finns
    minst ett hörn i varje zon. Returnerar (verdict, zone_hits) där
    zone_hits är en dict med True/False per zon.

    Zonerna är:
      TL = övre-vänster  (x < zone_margin*w, y < zone_margin*h)
      TR = övre-höger    (x > (1-zone_margin)*w, y < zone_margin*h)
      BL = nedre-vänster (x < zone_margin*w, y > (1-zone_margin)*h)
      BR = nedre-höger   (x > (1-zone_margin)*w, y > (1-zone_margin)*h)
    """
    x_thresh_low  = crop_w * zone_margin
    x_thresh_high = crop_w * (1 - zone_margin)
    y_thresh_low  = crop_h * zone_margin
    y_thresh_high = crop_h * (1 - zone_margin)

    zones = {"TL": False, "TR": False, "BL": False, "BR": False}

    for cx, cy in unique_centers:
        if cx < x_thresh_low and cy < y_thresh_low:
            zones["TL"] = True
        if cx > x_thresh_high and cy < y_thresh_low:
            zones["TR"] = True
        if cx < x_thresh_low and cy > y_thresh_high:
            zones["BL"] = True
        if cx > x_thresh_high and cy > y_thresh_high:
            zones["BR"] = True

    all_found = all(zones.values())
    verdict = "GOOD" if all_found else "SUSPECT"
    return verdict, zones


def draw_corner_zones(img, zone_margin=ZONE_MARGIN):
    """Ritar halvtransparenta zoner i hörnen på analysbilden för felsökning."""
    h, w = img.shape[:2]
    x_low  = int(w * zone_margin)
    x_high = int(w * (1 - zone_margin))
    y_low  = int(h * zone_margin)
    y_high = int(h * (1 - zone_margin))

    overlay = img.copy()
    alpha = 0.12
    zone_color = (255, 200, 0)  # blågrön ton

    # Rita de fyra hörnzonerna
    cv2.rectangle(overlay, (0, 0),       (x_low, y_low),  zone_color, -1)
    cv2.rectangle(overlay, (x_high, 0),  (w, y_low),      zone_color, -1)
    cv2.rectangle(overlay, (0, y_high),  (x_low, h),      zone_color, -1)
    cv2.rectangle(overlay, (x_high, y_high), (w, h),      zone_color, -1)

    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


# Hitta alla bilder
images = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])
print(f"Hittade {len(images)} bilder")

# Visa en i taget (som en film)
for i, filename in enumerate(images):
    filepath = os.path.join(INPUT_DIR, filename)
    img = cv2.imread(filepath)

    if img is None:
        continue

    # Kör YOLO (plankdetektion)
    results = model(filepath, conf=CONFIDENCE, verbose=False)

    img_h, img_w = img.shape[:2]
    img_clean = img.copy()

    # Rita margin-zoner (mörka kanter)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (MARGIN_LEFT, img_h), (0, 0, 255), -1)
    cv2.rectangle(overlay, (img_w - MARGIN_RIGHT, 0), (img_w, img_h), (0, 0, 255), -1)
    img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

    full_count = 0
    analysis_img = None
    verdict = "GOOD"
    n_cracks = 0
    n_corners = 0

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

            pad = 30
            crop = img_clean[
                max(0, y1-pad):min(img_h, y2+pad),
                max(0, x1-pad):min(img_w, x2+pad)
            ]
            crop_h, crop_w = crop.shape[:2]

            analysis_img = crop.copy()

            # Rita hörnzoner på analysbilden (felsökning)
            analysis_img = draw_corner_zones(analysis_img)

            # --- 1. Hörnanalys med YOLO ---
            corner_results = corner_model(crop, conf=CORNER_CONFIDENCE, verbose=False)

            unique_centers = []
            DIST_THRESHOLD = 30

            for c_box in corner_results[0].boxes:
                cx1, cy1, cx2, cy2 = map(int, c_box.xyxy[0])
                center_x = (cx1 + cx2) // 2
                center_y = (cy1 + cy2) // 2

                is_new = True
                for ux, uy in unique_centers:
                    distance = ((center_x - ux)**2 + (center_y - uy)**2) ** 0.5
                    if distance < DIST_THRESHOLD:
                        is_new = False
                        break

                if is_new:
                    unique_centers.append((center_x, center_y))

            n_corners = len(unique_centers)

            # === ZONBASERAD HÖRNBEDÖMNING ===
            verdict, zones = check_corners_by_zone(unique_centers, crop_w, crop_h)

            # Rita ut hörn – färga efter om de hamnade i en giltig zon
            BOX_SIZE = 30
            zone_names = ["TL", "TR", "BL", "BR"]
            x_thresh_low  = crop_w * ZONE_MARGIN
            x_thresh_high = crop_w * (1 - ZONE_MARGIN)
            y_thresh_low  = crop_h * ZONE_MARGIN
            y_thresh_high = crop_h * (1 - ZONE_MARGIN)

            for center_x, center_y in unique_centers:
                # Kontrollera om detta hörn bidrar till en zon
                in_zone = (
                    (center_x < x_thresh_low  or center_x > x_thresh_high) and
                    (center_y < y_thresh_low   or center_y > y_thresh_high)
                )
                dot_color = (0, 255, 0) if in_zone else (0, 165, 255)  # grön = giltig, orange = utanför zon

                bx1 = center_x - BOX_SIZE // 2
                by1 = center_y - BOX_SIZE // 2
                bx2 = center_x + BOX_SIZE // 2
                by2 = center_y + BOX_SIZE // 2
                cv2.rectangle(analysis_img, (bx1, by1), (bx2, by2), dot_color, 2)
                cv2.circle(analysis_img, (center_x, center_y), 5, dot_color, -1)

            # Visa vilka zoner som saknas
            missing = [z for z, found in zones.items() if not found]
            missing_str = "saknas: " + ",".join(missing) if missing else "alla hörn OK"

            # --- 2. Sprickmodell ---
            crack_results = crack_model(crop, conf=CRACK_CONFIDENCE, verbose=False)
            n_cracks = len(crack_results[0].boxes)
            for cr_box in crack_results[0].boxes:
                crx1, cry1, crx2, cry2 = map(int, cr_box.xyxy[0])
                cv2.rectangle(analysis_img, (crx1, cry1), (crx2, cry2), (0, 0, 255), 2)
                cv2.putText(analysis_img, f"crack {float(cr_box.conf[0]):.0%}",
                            (crx1, cry1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # --- Text på analysbild ---
            c1 = (0, 255, 0) if verdict == "GOOD" else (0, 0, 255)
            c2 = (0, 255, 0) if n_cracks == 0 else (0, 0, 255)
            cv2.putText(analysis_img, f"Horn: {verdict} ({missing_str})", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c1, 2)
            cv2.putText(analysis_img, f"Sprickor: {n_cracks}", (5, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c2, 2)

            # Spara om defekt
            if verdict != "GOOD" or n_cracks > 0:
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"defekt_{filename}"), analysis_img)

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
        cv2.imshow("Analys: horn + sprickor", analysis_img)
    else:
        try: cv2.destroyWindow("Analys: horn + sprickor")
        except: pass

    key = cv2.waitKey(100)
    if key == ord('q'):
        break
    elif key == ord(' '):
        cv2.waitKey(0)

cv2.destroyAllWindows()
print("Klart!")
