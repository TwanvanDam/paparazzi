from ultralytics import YOLO
import os
import time  # Added to enable delay

# Load a smaller model
# model = YOLO("yolo11s.pt")  # or "yolov11-n.pt"
# 
# # Train on your dataset
# model.train(data="./data.yaml", epochs=100, batch=4, imgsz=520)

# Evaluate the model
model = YOLO("yolov11s_finetuned.pt")
model = model.eval()

# Loop through and show all images in the validation dataset
val_dir = "./dataset/images/val"
for image_name in os.listdir(val_dir):
    if image_name.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(val_dir, image_name)
        results = model(image_path)
        results[0].show()
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
        classes = results[0].boxes.cls.cpu().numpy().tolist()
        print(boxes, classes)  # Print the bounding box coordinates
        time.sleep(2)  # Process a new image every 2 seconds

# Save the model
model.save("./yolov11s_finetuned.pt")

