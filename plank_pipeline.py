import cv2
import os
from ultralytics import YOLO

# === KONFIGURATION ===
INPUT_DIR = "input_images"
MODEL_PATH = "best.pt"
CONFIDENCE = 0.8

# Ladda modell
model = YOLO(MODEL_PATH)
print("Modell laddad")

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

    # Kör YOLO
    results = model(filepath, conf=CONFIDENCE, verbose=False)

    # Margin-parametrar
    img_h, img_w = img.shape[:2]
    MARGIN_LEFT = 50     # pixlar från vänster kant
    MARGIN_RIGHT = 120   # pixlar från höger kant
    MARGIN_Y = 15        # pixlar från topp/botten

    # Rita margin-zoner (mörka kanter)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (MARGIN_LEFT, img_h), (0, 0, 255), -1)
    cv2.rectangle(overlay, (img_w - MARGIN_RIGHT, 0), (img_w, img_h), (0, 0, 255), -1)
    img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

    # Analysera varje detektion
    full_count = 0
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
            color = (0, 255, 0)    # Grön = hel planka
            label = f"HEL {conf:.0%}"
            full_count += 1
        else:
            color = (0, 0, 255)    # Röd = delvis
            label = f"DELVIS {conf:.0%}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Bildinfo
    n_det = len(results[0].boxes)
    if full_count > 0:
        status = f"HEL PLANKA"
    elif n_det > 0:
        status = f"delvis"
    else:
        status = "ingen planka"
    cv2.putText(img, f"{i+1}/{len(images)} - {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Plankanalys (Q=avsluta, SPACE=paus)", img)

    key = cv2.waitKey(200)
    if key == ord('q'):
        break
    elif key == ord(' '):
        cv2.waitKey(0)

cv2.destroyAllWindows()
print("Klart!")