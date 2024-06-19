from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Set page configuration
st.set_page_config(page_title="Object Detection", page_icon="", layout="wide")

# Custom CSS for styling the button
st.markdown("""
    <style>
    .centered-button {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px; /* Adjust height as needed */
    }
    .animated-button {
        background-color: #4CAF50; /* Green */
        border: none;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
        border-radius: 12px;
    }
    .animated-button:hover {
        background-color: white;
        color: black;
        border: 2px solid #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Page title
st.markdown("<h1 style='text-align: center; color: navy;'> Object Detection </h1>", unsafe_allow_html=True)

# Image upload widget
image = st.file_uploader(label="Upload Your Image", type=["jpg", "png", "bmp", "jpeg", "webp"])


# Function to perform detection of objects
def predict(_img):
    model = YOLO("yolov8l-oiv7.pt")  # Load the pre-trained YOLO model
    results = model(source=_img, conf=0.4) # Perform inference
    return results

# Function to draw bounding boxes on the image
def draw_boxes(image, results):
    img_array = np.array(image)
    for result in results:
        boxes = result.boxes.xyxy.numpy()  
        confidences = result.boxes.conf.numpy()  
        class_ids = result.boxes.cls.numpy()  

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            name = result.names[int(class_id)]
            x1, y1, x2, y2 = map(int, box)
            label = f"{name}: {confidence:.2f}"

            cv2.rectangle(img_array, (x1, y1), (x2, y2), (255, 0, 0), 2)  
            cv2.putText(img_array, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)  

    return Image.fromarray(img_array)



# Main logic
if image is None:
    st.text("Please upload your image")
else:
    img = Image.open(image)

    st.image(img, use_column_width=True)

    
    if st.markdown('<div class="centered-button"><button class="animated-button">Analyse Image</button></div>', unsafe_allow_html=True):

        objects = predict(img)
        
        for obj in objects:
            boxes = obj.boxes.xyxy.numpy()  
            confidences = obj.boxes.conf.numpy()  
            class_ids = obj.boxes.cls.numpy()  

            for idx, (box, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                name = obj.names[int(class_id)]
                st.success(f"{idx + 1}. Object: {name}, Confidence: {confidence:.2f}")
                st.write(f"Bounding box: (x1: {box[0]:.2f}, y1: {box[1]:.2f}, x2: {box[2]:.2f}, y2: {box[3]:.2f})")
        
        
        
        img_with_boxes = draw_boxes(img, objects)
        st.image(img_with_boxes, caption="Image with detected objects", use_column_width=True)




