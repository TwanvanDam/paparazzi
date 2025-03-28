import matplotlib.pyplot as plt
from ultralytics import YOLO
import torchvision

# Load a smaller model
# model = YOLO("yolo11s.pt")  # or "yolov11-n.pt"
#
# # Train on your dataset
# model.train(data="./data.yaml", epochs=100, batch=4, imgsz=520)

# Evaluate the model
model = YOLO("yolov11s_finetuned.pt")
model = model.eval()

# Loop through and show all images in the validation dataset
test_image =  "./dataset/images/train/122215661.jpg"
results = model(test_image)
#results[0].show()

label_to_name = {0.0: "Panel", 1.0: "Plant", 2.0: "Pole", 3.0: "Blocks"}

plt.imshow(torchvision.io.read_image(test_image).permute(1, 2, 0) / 255)
boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
classes = results[0].boxes.cls.cpu().numpy().tolist()
for (box, pred_class) in zip(boxes, classes):
    x1, y1, x2, y2 = box
    print(x1, x2, y1, y2)
    plt.plot([x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1], "r-", linewidth=2)  # Plot the bounding box
    if y1 < 20:
        plot_text_y = 20
    else:
        plot_text_y = y1 - 20
    plt.text((x1+x2)/2, plot_text_y, label_to_name[pred_class], color="r", fontsize=12, ha='center', va='center')
plt.savefig("YOLO_prediction.pdf", dpi=300, bbox_inches='tight')
plt.show()

# Save the model
# model.save("./yolov11s_finetuned.pt")
