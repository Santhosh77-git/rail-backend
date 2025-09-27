# generate_secure_dense_dataset.py
import os
import qrcode
from qrcode.constants import ERROR_CORRECT_H
import json, base64
import random
from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256
import cv2
import numpy as np

# --------------------------
# Create dataset folders
# --------------------------
os.makedirs("dataset/PASS", exist_ok=True)
os.makedirs("dataset/FAIL", exist_ok=True)

# --------------------------
# Generate RSA Keys (2048-bit)
# --------------------------
key = RSA.generate(2048)
private_key = key
public_key = key.publickey()

# --------------------------
# Function to generate dense secure QR
# --------------------------
def make_secure_qr(data_dict, box_size=10, border=4):
    # Convert dict to JSON string
    message = json.dumps(data_dict)

    # Sign with RSA
    h = SHA256.new(message.encode())
    signature = pkcs1_15.new(private_key).sign(h)
    signature_b64 = base64.b64encode(signature).decode()

    # Prepare payload
    payload = {
        "data": data_dict,
        "signature": signature_b64
    }

    # Generate dense QR
    qr = qrcode.QRCode(
        version=None,
        error_correction=ERROR_CORRECT_H,
        box_size=box_size,
        border=border
    )
    qr.add_data(json.dumps(payload))
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("L")
    return np.array(img).astype(np.uint8)

# --------------------------
# Distortion functions for FAIL QR
# --------------------------
def add_distortions(img):
    h, w = img.shape
    mode = random.randint(0,5)
    if mode == 0:
        img = cv2.GaussianBlur(img, (11,11),0)
    elif mode == 1:
        angle = random.choice([15,25,35,-20])
        M = cv2.getRotationMatrix2D((w//2,h//2),angle,1.0)
        img = cv2.warpAffine(img, M, (w,h), borderValue=255)
    elif mode == 2:
        gauss = np.random.normal(0,30,img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16)+gauss,0,255).astype(np.uint8)
    elif mode == 3:
        sp_img = img.copy()
        num = int(0.03*h*w)
        coords = (np.random.randint(0,h,num), np.random.randint(0,w,num))
        sp_img[coords] = 255
        coords = (np.random.randint(0,h,num), np.random.randint(0,w,num))
        sp_img[coords] = 0
        img = cv2.GaussianBlur(sp_img,(5,5),0)
    elif mode == 4:
        img = (img.astype(np.float32)*0.5).astype(np.uint8)
        gauss = np.random.normal(0,20,img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16)+gauss,0,255).astype(np.uint8)
    else:
        for _ in range(random.randint(1,4)):
            x1,y1=random.randint(0,w-1),random.randint(0,h-1)
            x2,y2=random.randint(0,w-1),random.randint(0,h-1)
            cv2.line(img,(x1,y1),(x2,y2),random.choice([0,255]),random.randint(1,3))
        rw,rh=random.randint(w//6,w//3),random.randint(h//6,h//3)
        x,y=random.randint(0,w-rw),random.randint(0,h-rh)
        cv2.rectangle(img,(x,y),(x+rw,y+rh),random.choice([0,255]),-1)
    return img

# --------------------------
# ECC-H simulation: enlarge QR to strengthen modules
# --------------------------
def ecc_h(img):
    h,w = img.shape
    img = cv2.resize(img,(w*2,h*2), interpolation=cv2.INTER_NEAREST)
    return img

# --------------------------
# Generate 20 PASS QR images
# --------------------------
for i in range(50):
    data_dict = {
        "id": f"RF-CLIP-20250914-{1000+i}",
        "batch": f"BATCH-{random.randint(1000,9999)}",
        "qc": "PASS"
    }
    img = make_secure_qr(data_dict)
    img = ecc_h(img)
    img = cv2.resize(img,(256,256), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"dataset/PASS/qr_pass_{i}.png",img)

print("✅ 20 PASS QR images generated (dense + RSA + ECC-H)")

# --------------------------
# Generate 20 FAIL QR images
# --------------------------
for i in range(50):
    data_dict = {
        "id": f"RF-CLIP-20250914-{2000+i}",
        "batch": f"BATCH-{random.randint(1000,9999)}",
        "qc": "FAIL"
    }
    img = make_secure_qr(data_dict)
    img = add_distortions(img)
    img = ecc_h(img)
    img = cv2.resize(img,(256,256), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"dataset/FAIL/qr_fail_{i}.png",img)

print("✅ 20 FAIL QR images generated (dense + RSA + ECC-H + distortions)")
