from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="Object Detection", page_icon="", layout="wide")

image = st.file_uploader(label="Upload Your Image", type=["jpg", "png", "bmp", "jpeg", "webp"])

@st.cache_resource
def predict(_img):
    model = YOLO("yolov8n-oiv7.pt")
    results = model(source=_img, conf=0.4)
    return results


def draw_boxes(image, results):
    img_array = np.array(image)
    for result in results:
        boxes = result.boxes.xyxy.numpy()  # Bounding boxes
        confidences = result.boxes.conf.numpy()  # Confidences
        class_ids = result.boxes.cls.numpy()  # Class IDs

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            name = result.names[int(class_id)]
            x1, y1, x2, y2 = map(int, box)
            label = f"{name}: {confidence:.2f}"

            # Draw the bounding box
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw the label
            cv2.putText(img_array, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return Image.fromarray(img_array)




if image is None:
    st.text("Please upload your image")
else:
    img = Image.open(image)

    st.image(img, use_column_width=True)

    if st.button("Detect"):

        objects = predict(img)
        for obj in objects:
            boxes = obj.boxes.xyxy.numpy()  # Bounding boxes
            confidences = obj.boxes.conf.numpy()  # Confidences
            class_ids = obj.boxes.cls.numpy()  # Class IDs

            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                name = obj.names[int(class_id)]
                st.write(f"Object: {name}, Confidence: {confidence:.2f}")
                st.write(f"Bounding box: (x1: {box[0]:.2f}, y1: {box[1]:.2f}, x2: {box[2]:.2f}, y2: {box[3]:.2f})")
        
        
        
        img_with_boxes = draw_boxes(img, objects)
        st.image(img_with_boxes, caption="Image with detected objects", use_column_width=True)




