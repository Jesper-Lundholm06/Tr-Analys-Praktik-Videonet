import cv2 
import numpy as np
import os
import torch
from ultralytics import YOLO
# Fix: ultralytics patchar cv2.imshow, återställ originalet
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


# Output
OUTPUT_DIR = "defekta_plankor"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Ladda modeller
model = YOLO(MODEL_PATH)
crack_model = YOLO(CRACK_MODEL_PATH)
corner_model = YOLO(CORNER_MODEL_PATH)
print("Modeller laddade")


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

    # Margin-parametrar
    img_h, img_w = img.shape[:2]
    img_clean = img.copy()

    # Rita margin-zoner (mörka kanter)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (MARGIN_LEFT, img_h), (0, 0, 255), -1)
    cv2.rectangle(overlay, (img_w - MARGIN_RIGHT, 0), (img_w, img_h), (0, 0, 255), -1)
    img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

    # Analysera varje detektion
    full_count = 0
    analysis_img = None
    verdict = "GOOD"
    n_cracks = 0
    n_corners = 0

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # Margin-check: ligger boxen helt innanför?
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

            # Croppa ut plankan
            pad = 30
            crop = img_clean[
                max(0, y1-pad):min(img_h, y2+pad),
                max(0, x1-pad):min(img_w, x2+pad)
            ]

            # Skapa analysbild från croppen
            analysis_img = crop.copy()

   # --- 1. Hörnanalys med YOLO ---
            corner_results = corner_model(crop, conf=CORNER_CONFIDENCE, verbose=False)

            unique_centers = []
            DIST_THRESHOLD = 30  # justera mellan 20–40 vid behov

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
            

            # Rita ut hittade hörnn_corners = len(unique_centers)
            BOX_SIZE = 30  # storlek på rutan runt hörnet
            for center_x, center_y in unique_centers:
                x1 = center_x - BOX_SIZE // 2
                y1 = center_y - BOX_SIZE // 2
                x2 = center_x + BOX_SIZE // 2
                y2 = center_y + BOX_SIZE // 2
                cv2.rectangle(analysis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(analysis_img, (center_x, center_y), 5, (0, 255, 0), -1)
            # Bedöm: mindre än 4 hörn = defekt
            if n_corners < 4:
                verdict = "SUSPECT"
            else:
                verdict = "GOOD"

            # --- 2. Sprickmodell ---
            crack_results = crack_model(crop, conf=CRACK_CONFIDENCE, verbose=False)
            n_cracks = len(crack_results[0].boxes)
            for cr_box in crack_results[0].boxes:
                crx1, cry1, crx2, cry2 = map(int, cr_box.xyxy[0])
                cv2.rectangle(analysis_img, (crx1, cry1), (crx2, cry2), (0, 0, 255), 2)
                cv2.putText(analysis_img, f"crack {float(cr_box.conf[0]):.0%}",
                            (crx1, cry1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # --- Text på analysbild ---
            # --- Text på analysbild ---
            c1 = (0, 255, 0) if verdict == "GOOD" else (0, 0, 255)
            c2 = (0, 255, 0) if n_cracks == 0 else (0, 0, 255)
            cv2.putText(analysis_img, f"Horn: {verdict} ({n_corners}/4)", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c1, 2)
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

    # Bildinfo
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