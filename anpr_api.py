"""
ANPR REST API - Aggressive OCR Pipeline
Dirancang untuk plat kecil, gelap, gambar CCTV grayscale
POST /detect → binary image → JSON { plate, ... } atau 204
"""

import os
import re
import io
import cv2
import numpy as np
import pytesseract
from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
from dotenv import load_dotenv

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv()

def env_bool(k, d="true"):  return os.getenv(k, d).lower() in ("true","1","yes")
def env_float(k, d):        return float(os.getenv(k, str(d)))
def env_int(k, d):          return int(os.getenv(k, str(d)))
def env_list(k, d):         return [x.strip() for x in os.getenv(k, str(d)).split(",")]

HOST             = os.getenv("HOST", "0.0.0.0")
PORT             = env_int("PORT", "5000")
YOLO_MODEL       = os.getenv("YOLO_MODEL", "yolo11n.pt")
YOLO_CONF_VEH    = env_float("YOLO_CONF_VEHICLE", "0.35")
VEHICLE_CLASSES  = [int(x) for x in env_list("VEHICLE_CLASSES", "2,3,5,7")]

ROI_Y_START      = env_float("ROI_Y_START", "0.20")
ROI_Y_END        = env_float("ROI_Y_END", "0.95")
ROI_X_START      = env_float("ROI_X_START", "0.05")
ROI_X_END        = env_float("ROI_X_END", "0.95")

PLATE_CROP_PAD   = env_int("PLATE_CROP_PAD", "12")
UPSCALE_FACTOR   = env_int("UPSCALE_FACTOR", "4")

USE_CLAHE        = env_bool("USE_CLAHE", "true")
CLAHE_CLIP       = env_float("CLAHE_CLIP_LIMIT", "3.0")
CLAHE_TILE       = env_int("CLAHE_TILE_SIZE", "8")
USE_DENOISE      = env_bool("USE_DENOISE", "true")
DENOISE_H        = env_int("DENOISE_H", "7")
BLUR_KERNEL      = env_int("BLUR_KERNEL", "3")
THRESHOLD_METHOD = os.getenv("THRESHOLD_METHOD", "both")
ADAPTIVE_BLOCK   = env_int("ADAPTIVE_BLOCK_SIZE", "15")
ADAPTIVE_C       = env_int("ADAPTIVE_C", "8")
USE_SHARPEN      = env_bool("USE_SHARPEN", "true")
SHARPEN_STRENGTH = env_float("SHARPEN_STRENGTH", "2.0")
MORPH_KERNEL     = env_int("MORPH_KERNEL", "1")
PSM_MODES        = [int(x) for x in env_list("TESSERACT_PSM_MODES", "7,8,6,13,11,3")]

# ── Tesseract ─────────────────────────────────────────────────────────────────
TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\padli\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
]

def setup_tesseract():
    cmd = os.getenv("TESSERACT_CMD", "")
    if cmd and os.path.exists(cmd):
        pytesseract.pytesseract.tesseract_cmd = cmd
        return cmd
    for p in TESSERACT_PATHS:
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            return p
    return "system PATH"

tess_path = setup_tesseract()

app = Flask(__name__)
print(f"[ANPR] Loading YOLO: {YOLO_MODEL}")
model = YOLO(YOLO_MODEL)
print(f"[ANPR] Ready | Tesseract: {tess_path} | Upscale: {UPSCALE_FACTOR}x")

last_debug = {"debug_img": None, "plate_crop": None, "ocr_log": []}


# ═══════════════════════════════════════════════════════════════════
# GRAYSCALE PIPELINE — semua parameter dari .env
# ═══════════════════════════════════════════════════════════════════

def to_gray(img: np.ndarray) -> np.ndarray:
    """Konversi BGR ke grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    """CLAHE — contrast limited adaptive histogram equalization."""
    if not USE_CLAHE:
        return gray
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE))
    return clahe.apply(gray)


def apply_denoise(gray: np.ndarray) -> np.ndarray:
    """Fast NL Means denoising untuk gambar CCTV."""
    if not USE_DENOISE:
        return gray
    return cv2.fastNlMeansDenoising(gray, h=DENOISE_H)


def apply_sharpen(gray: np.ndarray) -> np.ndarray:
    """Unsharp masking untuk pertajam tepi karakter."""
    if not USE_SHARPEN:
        return gray
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    return cv2.addWeighted(gray, 1 + SHARPEN_STRENGTH, blurred, -SHARPEN_STRENGTH, 0)


def apply_blur(gray: np.ndarray) -> np.ndarray:
    """Gaussian blur ringan sebelum threshold."""
    k = BLUR_KERNEL if BLUR_KERNEL % 2 == 1 else BLUR_KERNEL + 1
    return cv2.GaussianBlur(gray, (k, k), 0) if k > 1 else gray


def apply_morph(img: np.ndarray) -> np.ndarray:
    """Morphological closing — isi gap kecil antar piksel."""
    if MORPH_KERNEL <= 1:
        return img
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL, MORPH_KERNEL))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)


def grayscale_pipeline(img: np.ndarray) -> list:
    """
    Pipeline preprocessing lengkap.
    Semua parameter dikontrol via .env.
    Return: list of (label, processed_image)
    """
    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return []

    # 1. Upscale — makin besar makin akurat OCR
    scale = UPSCALE_FACTOR
    # Minimal lebar 600px setelah upscale
    if w * scale < 600:
        scale = max(scale, int(600 / w) + 1)
    img_up = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # 2. Grayscale
    gray = to_gray(img_up)

    # 3. CLAHE
    enhanced = apply_clahe(gray)

    # 4. Denoise
    denoised = apply_denoise(enhanced)

    # 5. Sharpen
    sharpened = apply_sharpen(denoised)

    # 6. Blur sebelum threshold
    blurred = apply_blur(sharpened)

    variants = []

    # 7a. Otsu threshold
    if THRESHOLD_METHOD in ("otsu", "both"):
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants += [
            ("otsu",         apply_morph(otsu)),
            ("otsu_inv",     apply_morph(cv2.bitwise_not(otsu))),
        ]
        # Otsu pada raw enhanced (tanpa denoise/sharpen)
        _, otsu_raw = cv2.threshold(
            apply_blur(apply_clahe(gray)), 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        variants += [
            ("otsu_raw",     otsu_raw),
            ("otsu_raw_inv", cv2.bitwise_not(otsu_raw)),
        ]

    # 7b. Adaptive threshold — cocok untuk gambar tidak merata
    if THRESHOLD_METHOD in ("adaptive", "both"):
        block = ADAPTIVE_BLOCK if ADAPTIVE_BLOCK % 2 == 1 else ADAPTIVE_BLOCK + 1
        adap = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            block, ADAPTIVE_C
        )
        variants += [
            ("adaptive",     apply_morph(adap)),
            ("adaptive_inv", apply_morph(cv2.bitwise_not(adap))),
        ]
        # Adaptive dengan C lebih besar
        adap2 = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            block, ADAPTIVE_C + 4
        )
        variants += [
            ("adaptive2",     adap2),
            ("adaptive2_inv", cv2.bitwise_not(adap2)),
        ]

    # 7c. Threshold manual beberapa nilai
    for thresh_val in [80, 100, 120, 140, 160]:
        _, manual = cv2.threshold(sharpened, thresh_val, 255, cv2.THRESH_BINARY)
        variants.append((f"manual_{thresh_val}", manual))
        variants.append((f"manual_{thresh_val}_inv", cv2.bitwise_not(manual)))

    # 8. Raw grayscale — tanpa threshold (kadang Tesseract lebih baik)
    variants += [
        ("gray_enhanced", enhanced),
        ("gray_sharpened", sharpened),
        ("gray_raw", gray),
    ]

    return variants


# ═══════════════════════════════════════════════════════════════════
# PLATE VALIDATOR & SCORER
# ═══════════════════════════════════════════════════════════════════

def clean_plate_text(raw: str) -> str:
    """Bersihkan teks OCR — ambil alphanumeric saja, uppercase."""
    return "".join(c for c in raw.strip() if c.isalnum()).upper()


def is_valid_plate(text: str) -> bool:
    """
    Validasi KETAT format plat Indonesia.
    Format: 1-2 huruf + 1-4 angka + 1-3 huruf
    Contoh: B1234XY, F8523WZ, D123AB
    """
    t = text.strip().upper().replace(" ", "")
    return bool(re.match(r'^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$', t)) and 4 <= len(t) <= 10


def plate_score(text: str) -> int:
    """
    Scoring untuk memilih hasil OCR terbaik meski tidak valid sempurna.
    Makin tinggi score = makin mirip format plat Indonesia.
    Skema: huruf-angka-huruf
    """
    t = text.strip().upper().replace(" ", "")
    if len(t) < 3 or len(t) > 12:
        return 0

    score = 0

    # Bonus: dimulai dengan 1-2 huruf
    if re.match(r'^[A-Z]{1,2}', t):
        score += 30

    # Bonus: ada angka di tengah
    if re.search(r'\d{1,4}', t):
        score += 25

    # Bonus: diakhiri huruf
    if re.search(r'[A-Z]{1,3}$', t):
        score += 20

    # Bonus: panjang ideal plat (5-9 karakter)
    if 5 <= len(t) <= 9:
        score += 15

    # Bonus: validasi penuh
    if is_valid_plate(t):
        score += 50

    # Penalti: terlalu panjang (noise)
    if len(t) > 10:
        score -= 30

    return score


def correct_ocr_errors(text: str) -> str:
    """
    Koreksi kesalahan OCR umum pada plat Indonesia.
    Angka yang sering salah dibaca sebagai huruf dan sebaliknya.
    """
    t = text.strip().upper().replace(" ", "")
    if len(t) < 3:
        return t

    # Deteksi posisi: prefix huruf, angka tengah, suffix huruf
    # Coba parse manual: ambil huruf depan, angka tengah, huruf belakang
    match = re.match(r'^([A-Z]{1,2})([A-Z0-9]{1,4})([A-Z]{1,3})$', t)
    if not match:
        # Coba koreksi: O→0, I→1, S→5, B→8 di bagian tengah
        # Temukan blok yang kemungkinan angka
        corrected = ""
        i = 0
        # Skip prefix huruf
        while i < len(t) and t[i].isalpha():
            corrected += t[i]
            i += 1
            if i >= 2:
                break
        # Koreksi bagian angka
        digit_part = ""
        while i < len(t) and len(digit_part) < 4:
            c = t[i]
            c = c.replace('O','0').replace('I','1').replace('S','5').replace('B','8').replace('G','6').replace('Z','2')
            if c.isdigit():
                digit_part += c
                i += 1
            else:
                break
        corrected += digit_part
        # Ambil sisa sebagai suffix huruf
        suffix = t[i:i+3]
        # Koreksi suffix: angka yang mirip huruf
        suffix = suffix.replace('0','O').replace('1','I').replace('5','S').replace('8','B').replace('6','G').replace('2','Z')
        corrected += suffix
        return corrected

    return t


# ═══════════════════════════════════════════════════════════════════
# OCR ENGINE
# ═══════════════════════════════════════════════════════════════════

def run_ocr_single(img: np.ndarray, psm: int) -> str:
    config = (
        f"--psm {psm} --oem 3 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
    )
    try:
        raw = pytesseract.image_to_string(img, config=config)
        return clean_plate_text(raw)
    except Exception:
        return ""


def ocr_with_pipeline(crop: np.ndarray, label: str = "") -> tuple:
    """
    Jalankan seluruh grayscale pipeline + semua PSM mode.
    Return (best_text, is_valid) — selalu return hasil terbaik meski tidak valid.
    """
    variants = grayscale_pipeline(crop)
    log = []
    best_text = ""
    best_score = 0
    found_valid = False

    for vname, processed in variants:
        for psm in PSM_MODES:
            result = run_ocr_single(processed, psm)
            if len(result) < 3:
                continue

            # Coba koreksi error OCR
            corrected = correct_ocr_errors(result)
            valid = is_valid_plate(corrected)
            score = plate_score(corrected)

            entry = (f"{label}{vname} psm={psm} "
                     f"raw='{result}' → '{corrected}' "
                     f"score={score} {'✓ VALID' if valid else '✗'}")
            print(f"[OCR] {entry}")
            log.append(entry)

            if score > best_score:
                best_score = score
                best_text = corrected
                if valid:
                    found_valid = True

            # Early exit jika sudah valid sempurna
            if valid and score >= 100:
                last_debug["ocr_log"] = log
                return corrected, True

    last_debug["ocr_log"] = log

    # Return hasil terbaik meski tidak valid sempurna
    # asal score cukup (>= 50 = minimal ada huruf + angka)
    if best_score >= 50:
        return best_text, found_valid

    return "", False


# ═══════════════════════════════════════════════════════════════════
# YOLO VEHICLE DETECTION
# ═══════════════════════════════════════════════════════════════════

def detect_vehicles(img: np.ndarray) -> list:
    """
    Deteksi kendaraan di ROI dengan YOLO11n.
    Return list of (plate_zone_crop, global_bbox).
    """
    h_img, w_img = img.shape[:2]
    y1r = int(h_img * ROI_Y_START);  y2r = int(h_img * ROI_Y_END)
    x1r = int(w_img * ROI_X_START);  x2r = int(w_img * ROI_X_END)
    roi = img[y1r:y2r, x1r:x2r]

    results = model(roi, conf=YOLO_CONF_VEH, verbose=False)
    vehicles = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            gx1 = max(0, x1 + x1r - PLATE_CROP_PAD)
            gy1 = max(0, y1 + y1r - PLATE_CROP_PAD)
            gx2 = min(w_img, x2 + x1r + PLATE_CROP_PAD)
            gy2 = min(h_img, y2 + y1r + PLATE_CROP_PAD)

            # Zona plat = 50%-100% tinggi kendaraan, tengah horizontal
            vh = gy2 - gy1
            vw = gx2 - gx1
            pz_y1 = gy1 + int(vh * 0.50)
            pz_y2 = gy2
            pz_x1 = gx1 + int(vw * 0.05)
            pz_x2 = gx2 - int(vw * 0.05)

            plate_zone = img[pz_y1:pz_y2, pz_x1:pz_x2]
            if plate_zone.size == 0:
                continue

            cls_name = {2:"car", 3:"moto", 5:"bus", 7:"truck"}.get(cls_id, str(cls_id))
            bbox = {
                "x1": gx1, "y1": gy1, "x2": gx2, "y2": gy2,
                "plate_zone": {"x1": pz_x1, "y1": pz_y1, "x2": pz_x2, "y2": pz_y2},
                "class_id": cls_id, "class_name": cls_name,
                "confidence": round(conf, 3)
            }
            vehicles.append((plate_zone, bbox))
            print(f"[YOLO] {cls_name} conf={conf:.2f}")

    return vehicles


# ═══════════════════════════════════════════════════════════════════
# PLATE ZONE → CANDIDATE CROPS via CONTOUR
# ═══════════════════════════════════════════════════════════════════

def find_plate_crops(zone: np.ndarray) -> list:
    """
    Cari area plat di dalam plate_zone via kontur.
    Return list crop, diurutkan dari yang paling mungkin plat.
    """
    if zone.size == 0:
        return []

    gray = to_gray(zone)
    enhanced = apply_clahe(gray)
    candidates = []

    for (lo, hi) in [(15, 100), (25, 150), (40, 200), (60, 250)]:
        edges = cv2.Canny(enhanced, lo, hi)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:25]

        for c in cnts:
            if cv2.contourArea(c) < 200:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / float(h) if h > 0 else 0
            if 1.5 <= aspect <= 8.0 and w > 25 and h > 8:
                pad = 5
                cx1 = max(0, x - pad);  cy1 = max(0, y - pad)
                cx2 = min(zone.shape[1], x + w + pad)
                cy2 = min(zone.shape[0], y + h + pad)
                crop = zone[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    candidates.append((crop, aspect, w * h))

    # Deduplikasi
    unique = []
    for item in candidates:
        crop, asp, area = item
        dup = any(abs(crop.shape[1] - u[0].shape[1]) < 15
                  and abs(crop.shape[0] - u[0].shape[0]) < 8
                  for u in unique)
        if not dup:
            unique.append(item)

    # Urutkan: aspek ratio mendekati 3.5 (ideal plat), area terbesar
    unique.sort(key=lambda x: (abs(x[1] - 3.5), -x[2]))

    # Selalu sertakan seluruh zone sebagai fallback
    result = [c for c, _, _ in unique[:5]]
    result.append(zone)  # fallback: seluruh zona plat
    return result


# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def detect_plate(image_bytes: bytes):
    """
    Return (result_dict, True) jika plat ditemukan.
    Return (None, False) jika tidak → 204 No Content.
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Gambar tidak dapat di-decode."}, False

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    last_debug["ocr_log"] = []
    last_debug["plate_crop"] = None

    # Debug image
    debug_img = img.copy()
    h_img, w_img = img.shape[:2]
    cv2.rectangle(debug_img,
                  (int(w_img * ROI_X_START), int(h_img * ROI_Y_START)),
                  (int(w_img * ROI_X_END),   int(h_img * ROI_Y_END)),
                  (255, 100, 0), 2)

    # Step 1: YOLO deteksi kendaraan
    vehicles = detect_vehicles(img)
    print(f"[ANPR] {len(vehicles)} kendaraan terdeteksi")

    if not vehicles:
        # Fallback: coba OCR seluruh gambar langsung
        print("[ANPR] Tidak ada kendaraan → coba OCR full image")
        crops = find_plate_crops(img)
        best_text, best_valid = "", False
        for i, crop in enumerate(crops):
            last_debug["plate_crop"] = crop.copy()
            text, valid = ocr_with_pipeline(crop, f"full#{i} ")
            if valid:
                last_debug["debug_img"] = debug_img
                return _make_result(text, None, "unknown", valid), True
            if text and plate_score(text) > plate_score(best_text):
                best_text, best_valid = text, valid
        last_debug["debug_img"] = debug_img
        if best_text:
            return _make_result(best_text, None, "unknown", best_valid), True
        return None, False

    # Step 2: Per kendaraan → cari crops plat → OCR
    best_text_global, best_score_global, best_bbox_global, best_cls_global, best_valid_global = "", 0, None, "unknown", False

    for i, (plate_zone, bbox) in enumerate(vehicles):
        cls = bbox["class_name"]

        # Gambar bbox kendaraan (hijau)
        cv2.rectangle(debug_img, (bbox["x1"], bbox["y1"]), (bbox["x2"], bbox["y2"]), (0,255,0), 2)
        cv2.putText(debug_img, f"{cls} {bbox['confidence']:.2f}",
                    (bbox["x1"], max(0, bbox["y1"]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Gambar plate zone (kuning)
        pz = bbox["plate_zone"]
        cv2.rectangle(debug_img, (pz["x1"], pz["y1"]), (pz["x2"], pz["y2"]), (0,255,255), 2)

        # Step 3: Cari kandidat crop di plate_zone
        crops = find_plate_crops(plate_zone)
        print(f"[ANPR] Kendaraan#{i} ({cls}): {len(crops)} crop kandidat")

        for j, crop in enumerate(crops):
            last_debug["plate_crop"] = crop.copy()
            print(f"[ANPR]   crop#{j} → {crop.shape[1]}x{crop.shape[0]}px")

            text, valid = ocr_with_pipeline(crop, f"v{i}c{j} ")
            sc = plate_score(text) if text else 0

            # Simpan yang terbaik
            if sc > best_score_global:
                best_text_global  = text
                best_score_global = sc
                best_bbox_global  = bbox
                best_cls_global   = cls
                best_valid_global = valid

            # Early exit jika plat valid sempurna
            if valid:
                cv2.rectangle(debug_img, (pz["x1"], pz["y1"]), (pz["x2"], pz["y2"]), (0,0,255), 3)
                cv2.putText(debug_img, text,
                            (pz["x1"], max(0, pz["y1"]-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                last_debug["debug_img"] = debug_img
                return _make_result(text, bbox, cls, True), True

    # Tidak ada yang valid sempurna → return hasil terbaik jika score cukup
    last_debug["debug_img"] = debug_img
    if best_text_global and best_score_global >= 50:
        pz = best_bbox_global["plate_zone"] if best_bbox_global else None
        if pz:
            cv2.rectangle(debug_img, (pz["x1"], pz["y1"]), (pz["x2"], pz["y2"]), (0,165,255), 3)
            cv2.putText(debug_img, best_text_global,
                        (pz["x1"], max(0, pz["y1"]-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,165,255), 2)
        print(f"[ANPR] Best result (score={best_score_global}): {best_text_global}")
        return _make_result(best_text_global, best_bbox_global, best_cls_global, best_valid_global), True

    print("[ANPR] Semua kandidat dicek, skor terlalu rendah → skip")
    return None, False


def _make_result(plate: str, bbox, cls_name: str, valid: bool) -> dict:
    return {
        "plate":          plate,
        "ocr_text":       plate,
        "confidence":     bbox["confidence"] if bbox else 0.5,
        "vehicle_class":  cls_name,
        "bbox":           bbox,
        "valid_format":   valid,
        "message":        "OK" if valid else "Plat terbaca tapi format tidak sempurna"
    }


# ═══════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "ANPR API running",
        "debug": {
            "bbox":  "GET /debug",
            "plate": "GET /debug/plate",
            "log":   "GET /debug/log"
        }
    }), 200


@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_data()
    if not data:
        return jsonify({"error": "Body kosong."}), 400
    try:
        result, found = detect_plate(data)
        if not found:
            return "", 204
        return jsonify(result), 200
    except pytesseract.pytesseract.TesseractNotFoundError:
        return jsonify({"error": "Tesseract tidak ditemukan."}), 500
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/debug", methods=["GET"])
def debug_view():
    img = last_debug.get("debug_img")
    if img is None:
        return "Belum ada gambar.", 404
    _, buf = cv2.imencode(".jpg", img)
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")


@app.route("/debug/plate", methods=["GET"])
def debug_plate():
    crop = last_debug.get("plate_crop")
    if crop is None:
        return "Belum ada crop.", 404
    _, buf = cv2.imencode(".jpg", crop)
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")


@app.route("/debug/log", methods=["GET"])
def debug_log():
    return jsonify({"total": len(last_debug["ocr_log"]),
                    "ocr_log": last_debug["ocr_log"]}), 200


@app.route("/health", methods=["GET"])
def health():
    try:
        ver = str(pytesseract.get_tesseract_version()); ok = True
    except Exception as e:
        ver = str(e); ok = False
    return jsonify({
        "status": "ok", "model": YOLO_MODEL,
        "tesseract_ok": ok, "tesseract_version": ver,
        "upscale_factor": UPSCALE_FACTOR,
        "threshold_method": THRESHOLD_METHOD,
        "psm_modes": PSM_MODES,
        "vehicle_classes": VEHICLE_CLASSES
    }), 200


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"[ANPR] http://{HOST}:{PORT}")
    print(f"[ANPR] Upscale={UPSCALE_FACTOR}x | CLAHE={USE_CLAHE} | Denoise={USE_DENOISE} | Sharpen={USE_SHARPEN}({SHARPEN_STRENGTH})")
    print(f"[ANPR] Threshold={THRESHOLD_METHOD} | PSM={PSM_MODES}")
    print(f"[ANPR] Debug: http://127.0.0.1:{PORT}/debug")
    print(f"[ANPR] Plate: http://127.0.0.1:{PORT}/debug/plate")
    print(f"[ANPR] Log:   http://127.0.0.1:{PORT}/debug/log")
    app.run(host=HOST, port=PORT, debug=False)