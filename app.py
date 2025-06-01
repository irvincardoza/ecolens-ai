import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load models
cnn_model = load_model("waste_cnn_classifierVERSION.h5")
mobilenet_model = load_model("waste_classifier_mobilenetv2VERSION.h5")

IMG_SIZE = 224
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Function to classify
def classify_roi(roi):
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    pred1 = cnn_model.predict(img, verbose=0)
    pred2 = mobilenet_model.predict(img, verbose=0)
    avg_pred = (pred1 + pred2) / 2
    return class_names[np.argmax(avg_pred)], round(np.max(avg_pred) * 100, 2)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ No webcam found.")
    exit()

print("🟢 Press 'q' to quit.")
frame_counter = 0
label, conf = "", 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    box_size = int(min(h, w) * 0.4)
    x1, y1 = w // 2 - box_size // 2, h // 2 - box_size // 2
    x2, y2 = x1 + box_size, y1 + box_size

    # Draw center box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    roi = frame[y1:y2, x1:x2]

    # Classify every 10 frames
    frame_counter += 1
    if frame_counter % 10 == 0:
        label, conf = classify_roi(roi)

    # Show label
    cv2.putText(frame, f"{label.upper()} ({conf}%)", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Waste Detector (Center Box)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
