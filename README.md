# 🚛 Truck Plate Capture System

Detects vehicles from an RTSP stream using **YOLOv8**, selects the best frame where the plate is most visible, applies contrast enhancement + grayscale conversion (optimised for OCR), and publishes the image to an **MQTT** broker.

---

## Architecture

```
RTSP Camera
    │
    ▼
YOLOv8 Detection (vehicle classes: car / motorcycle / bus / truck)
    │
    ▼
Frame Buffer (collect N frames with vehicles)
    │
    ▼
Best-Frame Scoring
    ├── Area ratio      (vehicle size relative to frame)
    ├── Confidence      (YOLO detection score)
    ├── Centre score    (prefer vehicle centred horizontally)
    ├── Y-position      (prefer vehicle in lower portion for front-plate view)
    └── Aspect ratio    (reject obviously wrong boxes)
    │
    ▼
Image Enhancement
    ├── Grayscale conversion
    ├── CLAHE (adaptive local contrast)
    └── Linear contrast stretch (alpha / beta)
    │
    ▼
MQTT Publish (JPEG bytes → broker)
    │
    ▼
[Future] OCR / Plate Recognition
```

---

## Quick Start

```bash
# 1. Clone & enter the project
git clone <your-repo-url>
cd truck-detector

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure credentials
cp .env.example .env
# Edit .env with your RTSP URL and MQTT credentials

# 5. Run (weights download automatically on first run)
python detector.py
```

---

## Test with a Local Video

```bash
# Basic test (no MQTT, no preview window)
python test_video.py --video sample.mp4 --no-mqtt

# With live preview (requires a display)
python test_video.py --video sample.mp4 --no-mqtt --show

# Full test including MQTT publish
python test_video.py --video sample.mp4
```

Captured frames are saved to `captures/` as JPEG files.

---

## Configuration (`.env`)

| Variable            | Default          | Description                                    |
|---------------------|-----------------|------------------------------------------------|
| `RTSP_URL`          | —               | Full RTSP URL of the IP camera                 |
| `MQTT_BROKER`       | —               | Hostname of the MQTT broker                    |
| `MQTT_PORT`         | `1883`          | MQTT port                                      |
| `MQTT_USER`         | —               | MQTT username                                  |
| `MQTT_PASS`         | —               | MQTT password                                  |
| `MQTT_TOPIC`        | `/truck/plat4`  | Topic to publish images to                     |
| `MODEL_PATH`        | `yolov8n.pt`    | Path to YOLOv8 weights                         |
| `CONF_THRESHOLD`    | `0.4`           | Minimum detection confidence                   |
| `MIN_VEHICLE_AREA`  | `0.05`          | Minimum vehicle area as fraction of frame      |
| `FRAME_BUFFER_SIZE` | `30`            | Frames to collect before picking best          |
| `COOLDOWN_SECONDS`  | `5`             | Min seconds between publishes                  |
| `CONTRAST_ALPHA`    | `1.8`           | Linear contrast multiplier                     |
| `CONTRAST_BETA`     | `10`            | Brightness offset                              |
| `CLAHE_CLIP`        | `3.0`           | CLAHE clip limit                               |
| `CLAHE_TILE`        | `8`             | CLAHE tile grid size                           |
| `JPEG_QUALITY`      | `85`            | JPEG quality for MQTT payload (1–100)          |

---

## File Structure

```
truck-detector/
├── detector.py        # Main RTSP detection loop
├── test_video.py      # Offline test against a local video
├── .env               # Your credentials (NOT committed)
├── .env.example       # Template (safe to commit)
├── .gitignore
├── requirements.txt
├── captures/          # Auto-created, saved best frames
└── README.md
```

---

## Roadmap / Next Steps

- [ ] Integrate ALPR / plate OCR (e.g. EasyOCR, PaddleOCR, OpenALPR)
- [ ] Fine-tune YOLOv8 on Indonesian truck/plate dataset for better recall
- [ ] Add plate-specific bounding box detection (YOLOv8 custom class)
- [ ] Publish OCR text result alongside image on MQTT
- [ ] Docker image for easy deployment on NVR / edge device
