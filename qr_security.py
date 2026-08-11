import json
import base64

from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256


def verify_qr_signature(
    qr_text: str,
    public_key_pem: str
):

    try:

        payload = json.loads(qr_text)

        data = payload.get("data")
        signature_b64 = payload.get("signature")

        if not data or not signature_b64:
            return False, None

        message = json.dumps(
            data,
            separators=(",", ":")
        )

        signature = base64.b64decode(
            signature_b64
        )

        public_key = RSA.import_key(
            public_key_pem
        )

        digest = SHA256.new(
            message.encode()
        )

        pkcs1_15.new(
            public_key
        ).verify(
            digest,
            signature
        )

        return True, data

    except Exception as e:

        print(
            f"[SIGNATURE ERROR] {e}"
        )

        return False, None
