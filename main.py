from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Depends,
    Form
)

from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy.orm import Session

from datetime import datetime

import os
import json
import cv2
import numpy as np

from database import (
    engine,
    Base,
    get_db
)

from models import (
    Asset,
    Inspection
)

from ai_quality import (
    check_qr_quality
)


# =========================================================
# DATABASE
# =========================================================

Base.metadata.create_all(
    bind=engine
)


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="RailTrack AI QR Traceability API",
    description=(
        "AI-powered railway asset QR "
        "verification and inspection system"
    ),
    version="2.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =========================================================
# STORAGE
# =========================================================

os.makedirs(
    "scanned_qrs",
    exist_ok=True
)


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "status": "online",
        "service": "RailTrack AI QR API",
        "version": "2.0.0"
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# =========================================================
# REGISTER ASSET
# =========================================================

@app.post("/assets")
def register_asset(
    asset_code: str,
    vendor: str,
    supply: str,
    warranty: str,
    support: str,
    location: str,
    db: Session = Depends(get_db)
):

    existing = (
        db.query(Asset)
        .filter(
            Asset.asset_code == asset_code
        )
        .first()
    )

    if existing:

        return {
            "status": "error",
            "message": "Asset already exists"
        }


    asset = Asset(

        asset_code=asset_code,

        vendor=vendor,

        supply=supply,

        warranty=warranty,

        support=support,

        location=location,

        status="active"
    )

    db.add(asset)

    db.commit()

    db.refresh(asset)


    return {

        "status": "created",

        "asset": {
            "id": asset.id,
            "asset_code": asset.asset_code,
            "vendor": asset.vendor,
            "supply": asset.supply,
            "warranty": asset.warranty,
            "support": asset.support,
            "location": asset.location,
            "status": asset.status
        }
    }


# =========================================================
# SCAN QR
# =========================================================

@app.post("/scan_qr")
async def scan_qr(

    file: UploadFile = File(...),

    inspector: str = Form(
        default="unknown"
    ),

    notes: str = Form(
        default=""
    ),

    db: Session = Depends(get_db)
):

    # -----------------------------------------------------
    # READ IMAGE
    # -----------------------------------------------------

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
            "message": "Invalid image"
        }


    # -----------------------------------------------------
    # SAVE IMAGE
    # -----------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    filename = (
        f"qr_{timestamp}.png"
    )

    file_path = os.path.join(
        "scanned_qrs",
        filename
    )

    cv2.imwrite(
        file_path,
        frame
    )


    # -----------------------------------------------------
    # PREPROCESS
    # -----------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    scale = 4

    big = cv2.resize(
        gray,
        (
            gray.shape[1] * scale,
            gray.shape[0] * scale
        )
    )


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


    # -----------------------------------------------------
    # QR DECODER
    # -----------------------------------------------------

    detector = cv2.QRCodeDetector()

    data, bbox, _ = (
        detector.detectAndDecode(sharp)
    )


    # -----------------------------------------------------
    # AI QUALITY
    # -----------------------------------------------------

    ai_prediction, confidence = (
        check_qr_quality(frame)
    )


    # -----------------------------------------------------
    # QR SUCCESS
    # -----------------------------------------------------

    if data:

        asset_code = None

        try:

            payload = json.loads(
                data
            )

            qr_data = payload.get(
                "data",
                {}
            )

            asset_code = (
                qr_data.get("id")
            )

        except Exception:

            asset_code = data


        # -------------------------------------------------
        # DATABASE LOOKUP
        # -------------------------------------------------

        asset = None

        if asset_code:

            asset = (
                db.query(Asset)
                .filter(
                    Asset.asset_code
                    == asset_code
                )
                .first()
            )


        # -------------------------------------------------
        # INSPECTION RECORD
        # -------------------------------------------------

        inspection = Inspection(

            asset_code=(
                asset_code
                or "unknown"
            ),

            qr_status="decoded",

            ai_prediction=ai_prediction,

            ai_confidence=confidence,

            inspector=inspector,

            image_path=file_path,

            notes=notes
        )

        db.add(
            inspection
        )

        db.commit()


        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------

        return {

            "status": "decoded",

            "qr_text": data,

            "asset_code": asset_code,

            "ai": {

                "prediction":
                    ai_prediction,

                "confidence":
                    confidence
            },

            "asset_found":
                asset is not None,

            "asset": (
                {
                    "asset_code":
                        asset.asset_code,

                    "vendor":
                        asset.vendor,

                    "supply":
                        asset.supply,

                    "warranty":
                        asset.warranty,

                    "support":
                        asset.support,

                    "location":
                        asset.location,

                    "status":
                        asset.status
                }
                if asset
                else None
            )
        }


    # -----------------------------------------------------
    # QR NOT DECODED
    # -----------------------------------------------------

    inspection = Inspection(

        asset_code="unknown",

        qr_status="not_decoded",

        ai_prediction=ai_prediction,

        ai_confidence=confidence,

        inspector=inspector,

        image_path=file_path,

        notes=notes
    )

    db.add(
        inspection
    )

    db.commit()


    return {

        "status": "ai_checked",

        "qr_decoded": False,

        "prediction":
            ai_prediction,

        "confidence":
            confidence,

        "message":
            "QR could not be decoded. "
            "AI quality analysis completed."
    }


# =========================================================
# GET ASSET
# =========================================================

@app.get("/assets/{asset_code}")
def get_asset(
    asset_code: str,
    db: Session = Depends(get_db)
):

    asset = (
        db.query(Asset)
        .filter(
            Asset.asset_code
            == asset_code
        )
        .first()
    )

    if not asset:

        return {
            "status": "not_found",
            "message": "Asset not found"
        }


    inspections = (
        db.query(Inspection)
        .filter(
            Inspection.asset_code
            == asset_code
        )
        .order_by(
            Inspection.timestamp.desc()
        )
        .all()
    )


    return {

        "status": "success",

        "asset": {

            "asset_code":
                asset.asset_code,

            "vendor":
                asset.vendor,

            "supply":
                asset.supply,

            "warranty":
                asset.warranty,

            "support":
                asset.support,

            "location":
                asset.location,

            "status":
                asset.status
        },

        "inspection_history": [

            {

                "id":
                    inspection.id,

                "qr_status":
                    inspection.qr_status,

                "ai_prediction":
                    inspection.ai_prediction,

                "ai_confidence":
                    inspection.ai_confidence,

                "inspector":
                    inspection.inspector,

                "timestamp":
                    inspection.timestamp
            }

            for inspection
            in inspections
        ]
    }
