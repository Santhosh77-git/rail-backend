from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import tensorflow as tf
from datetime import datetime
import os


# ============================================================
# LOAD TFLITE MODEL
# ============================================================

MODEL_PATH = "qr_quality_model.tflite"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. "
        "Make sure qr_quality_model.tflite is in the backend folder."
    )

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# ============================================================
# CREATE STORAGE FOLDER
# ============================================================

os.makedirs("scanned_qrs", exist_ok=True)


# ============================================================
# FASTAPI SETUP
# ============================================================

app = FastAPI(
    title="RailTrack QR Quality API",
    description="QR decoding and AI-based QR quality checking API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# AI PROCESSING FUNCTION
# ============================================================

def run_ai_check(file_path: str):

    frame = cv2.imread(file_path)

    if frame is None:
        print(f"[ERROR] Could not read image: {file_path}")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img_resized = cv2.resize(
        gray,
        (128, 128)
    )

    img_resized = img_resized.astype(np.float32) / 255.0

    img_resized = np.expand_dims(
        img_resized,
        axis=(0, -1)
    )

    interpreter.set_tensor(
        input_details[0]["index"],
        img_resized
    )

    interpreter.invoke()

    prediction = interpreter.get_tensor(
        output_details[0]["index"]
    )

    ai_result = (
        "clear"
        if np.argmax(prediction) == 1
        else "blurry"
    )

    print(
        f"[AI RESULT] {file_path} -> {ai_result}"
    )


# ============================================================
# QR SCAN API
# ============================================================

@app.post("/scan_qr")
async def scan_qr(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):

    # --------------------------------------------------------
    # Read uploaded image
    # --------------------------------------------------------

    contents = await file.read()

    npimg = np.frombuffer(
        contents,
        np.uint8
    )

    frame = cv2.imdecode(
        npimg,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        return {
            "status": "error",
            "message": "Invalid image file"
        }

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Upscale image
    # --------------------------------------------------------

    scale = 4

    big = cv2.resize(
        gray,
        (
            gray.shape[1] * scale,
            gray.shape[0] * scale
        )
    )

    # --------------------------------------------------------
    # Sharpen image
    # --------------------------------------------------------

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharp = cv2.filter2D(
        big,
        -1,
        kernel
    )

    # --------------------------------------------------------
    # QR Detection
    # --------------------------------------------------------

    detector = cv2.QRCodeDetector()

    data, bbox, _ = detector.detectAndDecode(
        sharp
    )

    # --------------------------------------------------------
    # Save uploaded image
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S%f"
    )

    file_path = os.path.join(
        "scanned_qrs",
        f"qr_{timestamp}.png"
    )

    cv2.imwrite(
        file_path,
        frame
    )

    # --------------------------------------------------------
    # QR successfully decoded
    # --------------------------------------------------------

    if data:

        return {
            "status": "decoded",
            "qr_text": data
        }

    # --------------------------------------------------------
    # QR could not be decoded
    # Run AI quality check
    # --------------------------------------------------------

    if background_tasks:

        background_tasks.add_task(
            run_ai_check,
            file_path
        )

    # --------------------------------------------------------
    # Immediate AI inference
    # --------------------------------------------------------

    img_resized = cv2.resize(
        gray,
        (128, 128)
    )

    img_resized = (
        img_resized.astype(np.float32)
        / 255.0
    )

    img_resized = np.expand_dims(
        img_resized,
        axis=(0, -1)
    )

    interpreter.set_tensor(
        input_details[0]["index"],
        img_resized
    )

    interpreter.invoke()

    prediction = interpreter.get_tensor(
        output_details[0]["index"]
    )

    ai_result = (
        "clear"
        if np.argmax(prediction) == 1
        else "blurry"
    )

    confidence = float(
        np.max(prediction)
    )

    return {
        "status": "ai_checked",
        "prediction": ai_result,
        "confidence": confidence
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "RailTrack QR Quality API",
        "version": "1.0.0"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# LOCAL SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
