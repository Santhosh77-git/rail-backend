from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import tensorflow as tf
from datetime import datetime
import os

# -------------------------- Load TFLite model --------------------------
interpreter = tf.lite.Interpreter(model_path="qr_quality_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------------- Create folder --------------------------
os.makedirs("scanned_qrs", exist_ok=True)

# -------------------------- FastAPI Setup --------------------------
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change if needed
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------- AI Processing Function --------------------------
def run_ai_check(file_path: str):
    frame = cv2.imread(file_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(gray, (128, 128))
    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = np.expand_dims(img_resized, axis=(0, -1))

    interpreter.set_tensor(input_details[0]['index'], img_resized)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    ai_result = "clear" if np.argmax(prediction) == 1 else "blurry"
    print(f"[AI RESULT] {file_path} -> {ai_result}")  # optional log

# -------------------------- API Endpoint --------------------------
@app.post("/scan_qr")
async def scan_qr(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Upscale & sharpen as in your original code
    scale = 4
    big = cv2.resize(gray, (gray.shape[1]*scale, gray.shape[0]*scale))
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharp = cv2.filter2D(big, -1, kernel)

    # QR detection
    detector = cv2.QRCodeDetector()
    data, bbox, _ = detector.detectAndDecode(sharp)

    # Save frame
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    file_path = f"scanned_qrs/qr_{timestamp}.png"
    cv2.imwrite(file_path, frame)

    if data:
        # QR decoded successfully
        return {
            "status": "decoded",
            "qr_text": data
        }
    else:
        # Schedule AI clarity check in background (non-blocking)
        if background_tasks:
            background_tasks.add_task(run_ai_check, file_path)

        # Also do immediate AI check for response (same as your original output)
        img_resized = cv2.resize(gray, (128, 128))
        img_resized = img_resized.astype(np.float32) / 255.0
        img_resized = np.expand_dims(img_resized, axis=(0, -1))

        interpreter.set_tensor(input_details[0]['index'], img_resized)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        ai_result = "clear" if np.argmax(prediction) == 1 else "blurry"

        return {
            "status": "ai_checked",
            "prediction": ai_result
        }

# -------------------------- Run Server --------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
