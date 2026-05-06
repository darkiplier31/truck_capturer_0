"""
Truck/Vehicle Plate Capture System
===================================
Detects vehicles using YOLOv8, captures the best frame where the plate
is most visible (sharpest, centered, closest), enhances contrast, converts
to grayscale, then publishes the image to MQTT.

Flow:
  RTSP stream → YOLOv8 detection → frame scoring → enhance → MQTT publish
"""

import os
import cv2
import time
import logging
import numpy as np
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
from ultralytics import YOLO
from datetime import datetime

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
load_dotenv()

STREAM_URL  = os.getenv("STREAM_URL",  "./truk.mp4")  # RTSP stream or local video file
MQTT_BROKER = os.getenv("MQTT_BROKER","192.168.110.12")
MQTT_PORT   = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER   = os.getenv("MQTT_USER")
MQTT_PASS   = os.getenv("MQTT_PASS")
MQTT_TOPIC  = "/truck/plat4"

# Detection settings
MODEL_PATH          = os.getenv("MODEL_PATH", "yolov8n.pt")   # or full path to weights
VEHICLE_CLASSES     = {2, 3, 5, 7}                            # car, motorcycle, bus, truck
CONFIDENCE_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.4"))

# Capture / scoring settings
FRAME_BUFFER_SIZE   = int(os.getenv("FRAME_BUFFER_SIZE", "30"))  # frames to collect before scoring
COOLDOWN_SECONDS    = float(os.getenv("COOLDOWN_SECONDS", "5"))   # min seconds between publishes
MIN_VEHICLE_AREA    = float(os.getenv("MIN_VEHICLE_AREA", "0.05")) # min fraction of frame area

# Image enhancement
CONTRAST_ALPHA  = float(os.getenv("CONTRAST_ALPHA", "1.8"))   # contrast multiplier (1.0=no change)
CONTRAST_BETA   = float(os.getenv("CONTRAST_BETA",  "10"))    # brightness offset
CLAHE_CLIP      = float(os.getenv("CLAHE_CLIP",     "3.0"))   # CLAHE clip limit
CLAHE_TILE      = int(os.getenv("CLAHE_TILE",       "8"))     # CLAHE tile grid size

# JPEG quality for MQTT payload
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("truck-detector")


# ─────────────────────────────────────────────
# Image Enhancement
# ─────────────────────────────────────────────

def enhance_for_ocr(frame: np.ndarray) -> np.ndarray:
    """
    Convert BGR frame to grayscale and apply contrast enhancement
    optimised for downstream plate OCR:
      1. Convert to grayscale
      2. CLAHE (adaptive histogram equalisation) for local contrast
      3. Linear alpha/beta contrast stretch for global brightness control
    Returns a single-channel (grayscale) image.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE – improves local contrast without blowing out highlights
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP,
        tileGridSize=(CLAHE_TILE, CLAHE_TILE)
    )
    gray = clahe.apply(gray)

    # Global contrast stretch
    gray = cv2.convertScaleAbs(gray, alpha=CONTRAST_ALPHA, beta=CONTRAST_BETA)

    return gray


# ─────────────────────────────────────────────
# Frame Scoring
# ─────────────────────────────────────────────

def sharpness_score(img: np.ndarray) -> float:
    """Laplacian variance – higher means sharper (less motion blur)."""
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def score_detection(box_xyxy, frame_shape, conf: float) -> float:
    """
    Score a single vehicle detection for plate readability.

    Criteria (all normalised 0–1, multiplied together so any zero kills the score):

    1. **Area ratio**   – vehicle occupies a meaningful portion of the frame
                          (larger = closer = plate more readable).
    2. **Confidence**   – YOLOv8 detection confidence.
    3. **Horizontal centre penalty** – vehicle centred in frame is preferred.
    4. **Lower-half bonus** – typical dashcam / IP-cam angle means the front
                              plate is visible when the vehicle is in the lower
                              2/3 of the frame.
    5. **Sharpness**    – computed on the cropped vehicle ROI.
    6. **Aspect penalty** – extremely wide or tall boxes are usually false positives.
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box_xyxy

    box_w = x2 - x1
    box_h = y2 - y1
    area  = box_w * box_h
    frame_area = h * w

    area_ratio = area / frame_area
    if area_ratio < MIN_VEHICLE_AREA:
        return 0.0

    # Aspect ratio check (vehicles are roughly 1:0.5 – 1:2)
    aspect = box_w / max(box_h, 1)
    if aspect < 0.3 or aspect > 4.0:
        return 0.0

    # Horizontal centering (0=perfect centre, 1=at edge)
    cx = (x1 + x2) / 2 / w
    centre_score = 1.0 - abs(cx - 0.5) * 2   # 1 at centre, 0 at edges

    # Lower-half presence (y2 near bottom is good for front-plate visibility)
    # Normalised: 0 if box top at very top, 1 if box bottom reaches 80 %+ of frame
    y_presence = min(y2 / h, 1.0)

    score = (
        area_ratio ** 0.5     # square-root dampens extreme size dominance
        * conf
        * centre_score
        * y_presence
    )
    return score


def pick_best_frame(buffer: list) -> tuple:
    """
    Given a list of (frame, detections) tuples, return
    (best_frame, best_box) with the highest aggregated score.
    """
    best_score = -1
    best_frame = None
    best_box   = None

    for frame, boxes in buffer:
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf)
            s = score_detection(xyxy, frame.shape, conf)
            if s > best_score:
                best_score = s
                best_frame = frame
                best_box   = xyxy

    return best_frame, best_box, best_score


# ─────────────────────────────────────────────
# MQTT Client
# ─────────────────────────────────────────────

def build_mqtt_client() -> mqtt.Client:
    client = mqtt.Client(client_id=f"truck-detector-{int(time.time())}")
    client.username_pw_set(MQTT_USER, MQTT_PASS)

    def on_connect(c, userdata, flags, rc):
        if rc == 0:
            log.info(f"MQTT connected to {MQTT_BROKER}:{MQTT_PORT}")
        else:
            log.error(f"MQTT connect failed, rc={rc}")

    def on_disconnect(c, userdata, rc):
        log.warning(f"MQTT disconnected (rc={rc}), will reconnect…")

    def on_publish(c, userdata, mid):
        log.info(f"MQTT published message id={mid} → {MQTT_TOPIC}")

    client.on_connect    = on_connect
    client.on_disconnect = on_disconnect
    client.on_publish    = on_publish

    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.loop_start()
    return client


def publish_image(client: mqtt.Client, image: np.ndarray) -> bool:
    """Encode image as JPEG bytes and publish to MQTT broker."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    ret, buf = cv2.imencode(".jpg", image, encode_params)
    if not ret:
        log.error("Failed to JPEG-encode image")
        return False

    payload = buf.tobytes()
    info = client.publish(MQTT_TOPIC, payload=payload, qos=1)
    log.info(f"Published {len(payload)/1024:.1f} KB to '{MQTT_TOPIC}'")
    return info.rc == mqtt.MQTT_ERR_SUCCESS


# ─────────────────────────────────────────────
# Main Detection Loop
# ─────────────────────────────────────────────

def is_local_file(url: str) -> bool:
    """Return True if url is a local file path rather than a network stream."""
    lower = url.lower().strip()
    return not (lower.startswith("rtsp://") or
                lower.startswith("http://") or
                lower.startswith("https://"))


def open_stream(url: str) -> cv2.VideoCapture:
    if is_local_file(url):
        if not os.path.exists(url):
            raise FileNotFoundError(f"Video file not found: {url}")
        cap = cv2.VideoCapture(url)
        log.info(f"Opened local file: {url}")
    else:
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimal buffer → low latency
        log.info(f"Opened stream: {url}")

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {url}")
    return cap


def run():
    log.info("Loading YOLOv8 model …")
    model = YOLO(MODEL_PATH)
    log.info(f"Model loaded: {MODEL_PATH}")

    mqtt_client = build_mqtt_client()
    time.sleep(1)   # let MQTT connect

    frame_buffer: list = []      # [(frame, vehicle_boxes), ...]
    last_publish_time = 0.0
    consecutive_empty = 0

    log.info(f"Opening source: {STREAM_URL}")
    cap = open_stream(STREAM_URL)
    local_file = is_local_file(STREAM_URL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if local_file:
                    log.info("End of video file – looping back to start …")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_buffer.clear()
                    consecutive_empty = 0
                    continue
                log.warning("Stream read failed – reconnecting in 3 s …")
                cap.release()
                time.sleep(3)
                cap = open_stream(STREAM_URL)
                frame_buffer.clear()
                consecutive_empty = 0
                continue

            # ── YOLOv8 inference ──────────────────────────────────────
            results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)[0]
            vehicle_boxes = [
                b for b in results.boxes
                if int(b.cls) in VEHICLE_CLASSES
            ]

            if vehicle_boxes:
                consecutive_empty = 0
                frame_buffer.append((frame.copy(), vehicle_boxes))
                log.debug(f"Detected {len(vehicle_boxes)} vehicle(s); buffer={len(frame_buffer)}")

                # Flush buffer when full
                if len(frame_buffer) >= FRAME_BUFFER_SIZE:
                    now = time.time()
                    if now - last_publish_time >= COOLDOWN_SECONDS:
                        best_frame, best_box, score = pick_best_frame(frame_buffer)
                        if best_frame is not None and score > 0:
                            enhanced = enhance_for_ocr(best_frame)
                            publish_image(mqtt_client, enhanced)

                            # Optional: save locally for debugging
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            save_path = f"captures/capture_{ts}.jpg"
                            os.makedirs("captures", exist_ok=True)
                            cv2.imwrite(save_path, enhanced)
                            log.info(f"Saved capture: {save_path}  (score={score:.4f})")

                            last_publish_time = now
                    frame_buffer.clear()

            else:
                consecutive_empty += 1
                # If we stop seeing vehicles, flush buffer if it has content
                if consecutive_empty >= 10 and frame_buffer:
                    now = time.time()
                    if now - last_publish_time >= COOLDOWN_SECONDS:
                        best_frame, best_box, score = pick_best_frame(frame_buffer)
                        if best_frame is not None and score > 0:
                            enhanced = enhance_for_ocr(best_frame)
                            publish_image(mqtt_client, enhanced)

                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            save_path = f"captures/capture_{ts}.jpg"
                            os.makedirs("captures", exist_ok=True)
                            cv2.imwrite(save_path, enhanced)
                            log.info(f"Saved capture (on-exit): {save_path}  (score={score:.4f})")

                            last_publish_time = now
                    frame_buffer.clear()
                    consecutive_empty = 0

    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        cap.release()
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        log.info("Cleanup done.")


if __name__ == "__main__":
    run()