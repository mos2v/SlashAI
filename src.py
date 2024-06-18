from ultralytics import YOLO

img = "fronx-exterior-right-front-three-quarter-109.webp"
model = YOLO("yolov8n.pt")
model(img)
