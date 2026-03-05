class Config:
    # === MODELLER & INPUT ===
    MODEL_PATH = "../best.pt"
    CRACK_MODEL_PATH = "../best_cracks.pt"
    CORNER_MODEL_PATH = "../Best_Corners.pt"
    INPUT_DIR = "../input_images"

    # === KÄLLA ===
    SOURCE_MODE = "images"  # "images" eller "camera"
    RTSP_URL = "rtsp://service:Praktik26!@172.16.1.25:554/"

    # === OUTPUT – separata mappar ===
    OUTPUT_DIR_CRACKS = "defekta_plankor"
    OUTPUT_DIR_CORNERS = "defekta_horn"

    # === JUSTERBARA PARAMETRAR ===
    CONFIDENCE = 0.8
    CRACK_CONFIDENCE = 0.2

    # Hörndetektering – 2-zon (V/H) med bästa-per-zon-logik
    CORNER_SCAN_CONF = 0.01    # Låg tröskel – fånga ALLA detektioner
    CORNER_MIN_CONF  = 0.1     # Golv – bästa i zonen måste överstiga detta
    ZONE_MARGIN = 0.35
    DIST_THRESHOLD = 30

    # Marginaler
    MARGIN_LEFT = 50
    MARGIN_RIGHT = 120
    MARGIN_Y = 15

    # Crop-padding
    PAD = 80

    # === LOGGNING ===
    LOG_DIR = "logs"
    LOG_FILE = "pipeline_log.txt"

    # === HASTIGHET ===
    FRAME_DELAY = 0.3  # sekunder per bild

config = Config()