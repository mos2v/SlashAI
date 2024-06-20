from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from torchvision.models.detection import ssd300_vgg16
from torchvision import transforms
import json
import torch
with open('coco_labels.json', 'r') as f:
    COCO_LABELS = json.load(f)


# Set page configuration
st.set_page_config(page_title="Object Detection", page_icon="💸", layout="wide")

# Page title
st.markdown("<h1 style='text-align: center; color: navy;'> Object Detection </h1>", unsafe_allow_html=True)

# Image upload widget
image = st.file_uploader(label="Upload Your Image", type=["jpg", "png", "bmp", "jpeg", "webp"])

def detect_SSD(img):
    model = ssd300_vgg16(pretrained=True)
    model.eval()

    transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    image_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        detections = model(image_tensor)
    
    return detections
    
# Function to perform detection of objects
def detect_YOLO(img):
    model = YOLO("yolov8l-oiv7.pt")  # Load the pre-trained YOLO model
    results = model(source=img, conf=0.4) # Perform inference
    return results

# Function to draw bounding boxes on the image
def draw_boxes_YOLO(image, results):
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

def draw_boxes_SSD(image, results, COCO_LABELS):
    img_array = np.array(image)
    original_h, original_w = img_array.shape[:2]

    if len(results) > 0:
        boxes = results[0]['boxes'].detach().cpu().numpy()
        scores = results[0]['scores'].detach().cpu().numpy()
        labels = results[0]['labels'].detach().cpu().numpy()

        detection_info = []

        for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if score > 0.5:  # Display detections with confidence score > 0.5
                x_min = int(box[0] * original_w / 300)
                y_min = int(box[1] * original_h / 300)
                x_max = int(box[2] * original_w / 300)
                y_max = int(box[3] * original_h / 300)

                cv2.rectangle(img_array, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                label_text = f'{COCO_LABELS[label]}: {score:.2f}'
                cv2.putText(img_array, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                detection_info.append({
                    "label": COCO_LABELS[label],
                    "score": score,
                    "box": (x_min, y_min, x_max, y_max)
                })

        return Image.fromarray(img_array), detection_info
    else:
        return Image.fromarray(img_array), []



# Main logic
if image is None:
    st.text("Please upload your image")
else:
    img = Image.open(image)

    st.image(img)

    
    if st.button("Analyze Image with YOLO"):

        objects = detect_YOLO(img)
        
        for obj in objects:
            boxes = obj.boxes.xyxy.numpy()  
            confidences = obj.boxes.conf.numpy()  
            class_ids = obj.boxes.cls.numpy()  

            for idx, (box, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                name = obj.names[int(class_id)]
                st.success(f"{idx + 1}. Object: {name}, Confidence: {confidence:.2f}")
                st.write(f"Bounding box: (x1: {box[0]:.2f}, y1: {box[1]:.2f}, x2: {box[2]:.2f}, y2: {box[3]:.2f})")
        
        
        
        img_with_boxes = draw_boxes_YOLO(img, objects)
        st.image(img_with_boxes, caption="Image with detected objects", use_column_width=True)
    if st.button("Analyze Image with SSD"):
        objects = detect_SSD(img)
        img_with_boxes, detection_info = draw_boxes_SSD(img, objects, COCO_LABELS)

        for idx, info in enumerate(detection_info):
            st.success(f"{idx + 1}. Object: {info['label']}, Confidence: {info['score']:.2f}")
            st.write(f"Bounding box: (x1: {info['box'][0]}, y1: {info['box'][1]}, x2: {info['box'][2]}, y2: {info['box'][3]})")
        
        st.image(img_with_boxes, caption="Image with detected objects", use_column_width=True)

        




