import cv2
import pytesseract
import numpy as np
import re
import shutil
from pytesseract import Output
from difflib import SequenceMatcher


path = shutil.which("tesseract")
if path:
    pytesseract.pytesseract.tesseract_cmd = path



# ==========================================
# CONSTANTS
# ==========================================

PAN_PATTERN = re.compile(r"[A-Z]{5}\d{4}[A-Z]")
AADHAAR_PATTERN = re.compile(r"\d{4}[\s\-]?\d{4}[\s\-]?\d{4}")


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def name_similarity(a: str, b: str) -> float:
    """Fuzzy similarity between two names (0.0 – 1.0)."""
    if not a or not b:
        return 0.0
    # Normalize: lowercase, remove extra spaces
    a_clean = " ".join(a.lower().split())
    b_clean = " ".join(b.lower().split())
    return SequenceMatcher(None, a_clean, b_clean).ratio()


def smart_resize(image: np.ndarray) -> np.ndarray:
    """
    Resize image for optimal OCR. Tesseract works best at ~300 DPI,
    which roughly translates to text being ~12px+ tall.
    Small images get upscaled, huge images get downscaled.
    """
    h, w = image.shape[:2]

    if w < 500:
        # Very small → upscale 3x
        image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    elif w < 800:
        # Small → upscale 2x
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    elif w > 3000:
        # Very large → downscale
        scale = 1500 / w
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    return image


def run_ocr_multi_strategy(image: np.ndarray) -> tuple:
    """
    Run Tesseract with MULTIPLE preprocessing strategies and pick the
    one that gives the highest average OCR confidence.

    Returns: (best_text, best_confidence, best_data_dict)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ---- Build candidate images ----
    candidates = {}

    # Strategy A: Plain grayscale (best for clean/printed documents)
    candidates["grayscale"] = gray

    # Strategy B: Otsu threshold (good for high-contrast docs)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates["otsu"] = otsu

    # Strategy C: Light sharpen + grayscale (good for slightly blurry photos)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    candidates["sharpened"] = sharpened

    # ---- Run OCR on each and pick the best ----
    best_text = ""
    best_conf = 0.0
    best_data = None
    best_strategy = ""

    for name, img in candidates.items():
        try:
            data = pytesseract.image_to_data(img, output_type=Output.DICT, config="--psm 6")

            words, confs = [], []
            for i in range(len(data["text"])):
                txt = data["text"][i].strip()
                try:
                    c = float(data["conf"][i])
                except (ValueError, TypeError):
                    continue
                if txt and c > 0:
                    words.append(txt)
                    confs.append(c)

            avg_conf = np.mean(confs) if confs else 0.0
            text = " ".join(words)

            if avg_conf > best_conf:
                best_conf = avg_conf
                best_text = text
                best_data = data
                best_strategy = name

        except Exception:
            continue

    # Also get line-structured text from the best strategy image
    best_img = candidates.get(best_strategy, gray)
    line_text = pytesseract.image_to_string(best_img, config="--psm 6")

    return best_text, best_conf, best_data, line_text, best_strategy


def detect_document_type(text: str) -> str:
    """Detect PAN vs Aadhaar vs Unknown."""
    upper = text.upper()

    pan_score = 0
    aadhaar_score = 0

    # Keyword signals
    for kw in ["INCOME TAX", "PERMANENT ACCOUNT", "PAN"]:
        if kw in upper:
            pan_score += 2

    for kw in ["AADHAAR", "AADHAR", "UIDAI", "UNIQUE IDENTIFICATION", "ENROL"]:
        if kw in upper:
            aadhaar_score += 2

    # Pattern signals
    if PAN_PATTERN.search(upper):
        pan_score += 4
    if AADHAAR_PATTERN.search(text):
        aadhaar_score += 4

    if pan_score > aadhaar_score:
        return "PAN"
    elif aadhaar_score > pan_score:
        return "AADHAAR"
    else:
        return "UNKNOWN"


# ==========================================
# NAME EXTRACTION
# ==========================================

def extract_name(full_text: str, line_text: str, doc_type: str) -> str:
    """
    Extract person's name using multiple strategies.
    """
    combined = full_text + "\n" + line_text

    # ---- Strategy 1: Look for "Name" keyword ----
    # Handles: "Name: Rahul Sharma", "Name PENAGANTI SAI", "नाम: ...", etc.
    name_patterns = [
        r"(?:Name|नाम)\s*[:\-]?\s*([A-Za-z][A-Za-z.\s]{2,40})",
        r"(?:Name|नाम)\s*[:\-]?\s*\n\s*([A-Za-z][A-Za-z.\s]{2,40})",
    ]

    for pattern in name_patterns:
        match = re.search(pattern, combined, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            name = match.group(1).strip()
            # Clean up: remove trailing non-name words
            name = re.sub(r"\b(?:Date|DOB|Father|Mother|Signature|Valid)\b.*", "", name, flags=re.IGNORECASE).strip()
            name = re.sub(r"[^A-Za-z\s.]", "", name).strip()
            if len(name) >= 3 and " " in name:
                return name

    # ---- Strategy 2: Line-by-line scan ----
    lines = [l.strip() for l in combined.split("\n") if l.strip()]

    for i, line in enumerate(lines):
        lower = line.lower()
        # If this line says "Name" and next line has the actual name
        if re.search(r"\bname\b", lower) and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            cleaned = re.sub(r"[^A-Za-z\s]", "", next_line).strip()
            words = cleaned.split()
            if 2 <= len(words) <= 4:
                return cleaned

    # ---- Strategy 3: For PAN cards, find ALL-CAPS name lines ----
    if doc_type == "PAN":
        # PAN card names are UPPERCASE, 2-3 words, and NOT keywords
        skip_words = {"INCOME", "TAX", "DEPARTMENT", "GOVT", "INDIA", "PERMANENT",
                      "ACCOUNT", "NUMBER", "CARD", "DATE", "BIRTH", "SIGNATURE",
                      "FATHER", "GOVERNMENT"}
        for line in lines:
            cleaned = re.sub(r"[^A-Z\s]", "", line).strip()
            words = cleaned.split()
            if 2 <= len(words) <= 4 and len(cleaned) >= 5:
                if not any(w in skip_words for w in words):
                    # Check that most words are actual name-like (3+ chars, no digits)
                    if all(len(w) >= 2 for w in words):
                        return cleaned

    # ---- Strategy 4: For Aadhaar, look for name near the top ----
    if doc_type == "AADHAAR":
        for line in lines[:8]:
            cleaned = re.sub(r"[^A-Za-z\s]", "", line).strip()
            words = cleaned.split()
            if 2 <= len(words) <= 4 and len(cleaned) >= 5:
                if all(len(w) >= 2 for w in words):
                    skip_check = {"government", "india", "unique", "identification",
                                  "authority", "aadhaar", "enrolment"}
                    if not any(w.lower() in skip_check for w in words):
                        return cleaned

    # ---- Strategy 5: Fallback — find any 2-3 capitalized-word sequence ----
    fallback = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b", combined)
    skip_names = {"Income Tax", "Permanent Account", "Date Birth", "Government India",
                  "Number Card", "Account Number"}
    for candidate in fallback:
        if candidate not in skip_names and len(candidate) >= 5:
            return candidate

    return ""


# ==========================================
# ID NUMBER EXTRACTION
# ==========================================

def extract_id_number(full_text: str, doc_type: str) -> tuple:
    """Extract PAN or Aadhaar number."""
    # Remove spaces for PAN matching (OCR sometimes adds spaces)
    no_space = re.sub(r"\s+", "", full_text.upper())

    if doc_type in ("PAN", "UNKNOWN"):
        match = PAN_PATTERN.search(no_space)
        if match:
            return match.group(), True

    if doc_type in ("AADHAAR", "UNKNOWN"):
        match = AADHAAR_PATTERN.search(full_text)
        if match:
            return match.group(), True

    # Try both anyway
    pan_match = PAN_PATTERN.search(no_space)
    if pan_match:
        return pan_match.group(), True

    aadhaar_match = AADHAAR_PATTERN.search(full_text)
    if aadhaar_match:
        return aadhaar_match.group(), True

    return "", False


# ==========================================
# FACE / PHOTO DETECTION
# ==========================================

def detect_face(image: np.ndarray) -> dict:
    """Detect face on the KYC document using Haar cascades."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) > 0:
        areas = [w * h for (x, y, w, h) in faces]
        idx = int(np.argmax(areas))
        x, y, w, h = faces[idx]
        return {
            "face_detected": True,
            "face_count": len(faces),
            "face_bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
        }

    return {"face_detected": False, "face_count": 0, "face_bbox": None}


# ==========================================
# MAIN EXTRACTION FUNCTION
# ==========================================

def extract_aligned_kyc_features(image_path: str, registered_name: str,
                                  debug: bool = True) -> dict:
    """
    Full KYC document analysis pipeline.
    Works with ANY image size and format (PAN, Aadhaar, etc.)

    Pipeline:
      1. Load & smart resize
      2. Multi-strategy OCR (grayscale / otsu / sharpened — picks best)
      3. Detect document type
      4. Extract name & compare
      5. Extract ID number
      6. Detect face/photo
    """

    # --- Load ---
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # --- Smart resize for OCR ---
    image = smart_resize(image)

    # --- Multi-strategy OCR ---
    full_text, avg_conf, ocr_data, line_text, strategy = run_ocr_multi_strategy(image)
    ocr_confidence = round(avg_conf / 100, 3)

    # --- Document type ---
    combined_text = full_text + " " + line_text
    doc_type = detect_document_type(combined_text)

    # --- Name ---
    extracted_name = extract_name(full_text, line_text, doc_type)
    similarity = name_similarity(extracted_name, registered_name)
    name_mismatch_flag = 1 if similarity < 0.7 else 0

    # --- ID Number ---
    extracted_id, pattern_valid = extract_id_number(combined_text, doc_type)
    invalid_id_format_flag = 0 if pattern_valid else 1

    # --- Face ---
    face_info = detect_face(image)

    # ===== OUTPUT =====
    output = {
        "name_mismatch_flag": name_mismatch_flag,
        "invalid_id_format_flag": invalid_id_format_flag,
        "ocr_confidence": ocr_confidence,
    }

    if debug:
        output.update({
            "document_type": doc_type,
            "ocr_strategy_used": strategy,
            "extracted_name": extracted_name,
            "registered_name": registered_name,
            "name_similarity": round(similarity, 3),
            "extracted_id": extracted_id,
            "id_pattern_valid": pattern_valid,
            "face_detected": face_info["face_detected"],
            "face_count": face_info["face_count"],
            "face_bbox": face_info["face_bbox"],
            "raw_text": full_text[:500],
        })

    return output


# ==========================================
# TEST RUN
# ==========================================

if __name__ == "__main__":

    test_cases = [
        
        (r"uploads/PAN CARD.jpg", "Penaganti Sai"),
    ]

    for image_path, name in test_cases:
        try:
            features = extract_aligned_kyc_features(image_path, name, debug=True)

            print("\n" + "=" * 55)
            print(f"  FILE: {image_path}")
            print("=" * 55)
            print(f"  Doc Type:      {features.get('document_type', 'N/A')}")
            print(f"  OCR Strategy:  {features.get('ocr_strategy_used', 'N/A')}")
            print(f"  OCR Confidence:{features['ocr_confidence']:.1%}")
            print(f"  Extracted Name:{features.get('extracted_name', '-')}")
            print(f"  Registered:    {features.get('registered_name', '-')}")
            print(f"  Similarity:    {features.get('name_similarity', 0):.1%}")
            print(f"  Name Mismatch: {'YES' if features['name_mismatch_flag'] else 'No'}")
            print(f"  Extracted ID:  {features.get('extracted_id', '-')}")
            print(f"  ID Valid:      {'Yes' if features.get('id_pattern_valid') else 'NO'}")
            print(f"  Face Detected: {'Yes' if features.get('face_detected') else 'NO'}")
            print(f"  Face Count:    {features.get('face_count', 0)}")
            print(f"  Raw Text:      {features.get('raw_text', '')[:150]}")
        except Exception as e:
            print(f"\nERROR on {image_path}: {e}")

    print("\n" + "=" * 55)



