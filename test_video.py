"""
test_video.py – Offline demo / unit test
=========================================
Runs the detector on a local MP4 file instead of the RTSP stream.
Useful for:
  - Verifying model detection accuracy before deploying
  - Tuning scoring parameters
  - Confirming MQTT publish works end-to-end

Usage:
    python test_video.py --video path/to/video.mp4 [--no-mqtt] [--show]
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Import shared logic from detector ──────────────────────────────────────────
from detector import (
    VEHICLE_CLASSES,
    CONFIDENCE_THRESHOLD,
    FRAME_BUFFER_SIZE,
    COOLDOWN_SECONDS,
    MIN_VEHICLE_AREA,
    enhance_for_ocr,
    score_detection,
    pick_best_frame,
    build_mqtt_client,
    publish_image,
    log,
)

try:
    from ultralytics import YOLO
    MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
except ImportError:
    sys.exit("ultralytics not installed. Run: pip install ultralytics")


def draw_boxes(frame: np.ndarray, boxes, scores: dict = None) -> np.ndarray:
    """Draw bounding boxes and class labels on a copy of the frame."""
    out = frame.copy()
    class_names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    for box in boxes:
        xyxy = [int(v) for v in box.xyxy[0].tolist()]
        x1, y1, x2, y2 = xyxy
        cls  = int(box.cls)
        conf = float(box.conf)
        label = f"{class_names.get(cls, cls)} {conf:.2f}"

        color = (0, 200, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def run_test(video_path: str, use_mqtt: bool, show: bool):
    log.info(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    mqtt_client = None
    if use_mqtt:
        mqtt_client = build_mqtt_client()
        time.sleep(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")

    total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps            = cap.get(cv2.CAP_PROP_FPS) or 30
    log.info(f"Video: {total_frames} frames @ {fps:.1f} fps")

    frame_buffer     = []
    last_publish     = 0.0
    consecutive_empty = 0
    frame_idx        = 0
    captures_saved   = 0

    os.makedirs("captures", exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)[0]
        vehicle_boxes = [b for b in results.boxes if int(b.cls) in VEHICLE_CLASSES]

        if show:
            vis = draw_boxes(frame, vehicle_boxes)
            label = f"Frame {frame_idx}/{total_frames}  Vehicles: {len(vehicle_boxes)}"
            cv2.putText(vis, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.imshow("Detection Preview", vis)
            key = cv2.waitKey(max(1, int(1000/fps)))
            if key == ord("q"):
                break

        if vehicle_boxes:
            consecutive_empty = 0
            frame_buffer.append((frame.copy(), vehicle_boxes))

            if len(frame_buffer) >= FRAME_BUFFER_SIZE:
                now = time.time()
                if now - last_publish >= COOLDOWN_SECONDS:
                    best_frame, best_box, score = pick_best_frame(frame_buffer)
                    if best_frame is not None and score > 0:
                        enhanced = enhance_for_ocr(best_frame)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
                        path = f"captures/capture_{ts}.jpg"
                        cv2.imwrite(path, enhanced)
                        log.info(f"[{frame_idx}] Capture saved → {path}  (score={score:.4f})")
                        captures_saved += 1

                        if mqtt_client:
                            publish_image(mqtt_client, enhanced)

                        last_publish = now
                frame_buffer.clear()

        else:
            consecutive_empty += 1
            if consecutive_empty >= 10 and frame_buffer:
                now = time.time()
                if now - last_publish >= COOLDOWN_SECONDS:
                    best_frame, best_box, score = pick_best_frame(frame_buffer)
                    if best_frame is not None and score > 0:
                        enhanced = enhance_for_ocr(best_frame)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
                        path = f"captures/capture_{ts}.jpg"
                        cv2.imwrite(path, enhanced)
                        log.info(f"[{frame_idx}] Capture saved (on-leave) → {path}  (score={score:.4f})")
                        captures_saved += 1

                        if mqtt_client:
                            publish_image(mqtt_client, enhanced)

                        last_publish = now
                frame_buffer.clear()
                consecutive_empty = 0

    # Flush remaining buffer at end of video
    if frame_buffer:
        best_frame, best_box, score = pick_best_frame(frame_buffer)
        if best_frame is not None and score > 0:
            enhanced = enhance_for_ocr(best_frame)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            path = f"captures/capture_{ts}.jpg"
            cv2.imwrite(path, enhanced)
            log.info(f"Final capture saved → {path}  (score={score:.4f})")
            captures_saved += 1

            if mqtt_client:
                publish_image(mqtt_client, enhanced)

    cap.release()
    if show:
        cv2.destroyAllWindows()
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()

    log.info(f"Done. Processed {frame_idx} frames, saved {captures_saved} capture(s).")
    log.info(f"Captures in: {os.path.abspath('captures')}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test detector on a local video file.")
    parser.add_argument("--video",   default="sample.mp4", help="Path to video file")
    parser.add_argument("--no-mqtt", action="store_true",   help="Skip MQTT publishing")
    parser.add_argument("--show",    action="store_true",   help="Show live preview window")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        sys.exit(f"Video not found: {args.video}")

    run_test(
        video_path=args.video,
        use_mqtt=not args.no_mqtt,
        show=args.show,
    )
