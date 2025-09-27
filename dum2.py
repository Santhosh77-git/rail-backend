import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ------------------------------
# Step 1: Dataset Loading
# ------------------------------
data_dir = "dataset"  # Folder with PASS and FAIL subfolders

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    color_mode="grayscale",
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation"
)

# ------------------------------
# Step 2: Build Model (CNN)
# ------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(128,128,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(2, activation="softmax")  # PASS vs FAIL
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ------------------------------
# Step 3: Train Model
# ------------------------------
history = model.fit(train_data, validation_data=val_data, epochs=5)

# ------------------------------
# Step 4: Save Model
# ------------------------------
model.save("qr_quality_model.h5")
print("✅ Model trained and saved as qr_quality_model.h5")

# ------------------------------
# Step 5: Convert to TFLite
# ------------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("qr_quality_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model converted and saved as qr_quality_model.tflite")
