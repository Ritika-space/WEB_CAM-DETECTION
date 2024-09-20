# WEB_CAM-DETECTION

import cv2
import numpy as np
import os
import urllib.request
import tkinter as tk

# Model files
model_file = "mobilenet_iter_73000.caffemodel"
prototxt_file = "deploy.prototxt"

# URLs for downloading model files
model_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"
prototxt_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/deploy.prototxt"

def download_file(url, filename):
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print(f"{filename} downloaded successfully.")

# Check if model files exist, download if not
if not os.path.exists(model_file):
    download_file(model_url, model_file)
if not os.path.exists(prototxt_file):
    download_file(prototxt_url, prototxt_file)

# Load the model
net = cv2.dnn.readNetFromCaffe(prototxt_file, model_file)

# Define a list of class labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

def detect_objects(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def start_webcam_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        result_frame = detect_objects(frame)
        cv2.imshow("Webcam Object Detection", result_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Create the main window
root = tk.Tk()
root.title("Webcam Object Detection")

# Create and pack the start button
start_button = tk.Button(root, text="Start Webcam Detection", command=start_webcam_detection)
start_button.pack(pady=10)

root.mainloop()
