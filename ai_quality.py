import os
import cv2
import numpy as np
import tensorflow as tf


MODEL_PATH = "qr_quality_model.tflite"


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        "qr_quality_model.tflite not found"
    )


interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = (
    interpreter.get_input_details()
)

output_details = (
    interpreter.get_output_details()
)


def check_qr_quality(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    resized = cv2.resize(
        gray,
        (128, 128)
    )

    resized = (
        resized.astype(np.float32)
        / 255.0
    )

    resized = np.expand_dims(
        resized,
        axis=(0, -1)
    )

    interpreter.set_tensor(
        input_details[0]["index"],
        resized
    )

    interpreter.invoke()

    prediction = interpreter.get_tensor(
        output_details[0]["index"]
    )

    class_index = int(
        np.argmax(prediction)
    )

    confidence = float(
        np.max(prediction)
    )

    # IMPORTANT:
    # Verify your train_data.class_indices.
    #
    # Example:
    # {'FAIL': 0, 'PASS': 1}

    if class_index == 1:
        result = "clear"
    else:
        result = "blurry"

    return result, confidence
