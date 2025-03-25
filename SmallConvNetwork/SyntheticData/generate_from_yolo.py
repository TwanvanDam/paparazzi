# TODO implement generation of synthetic data to aid SmallCNN model training
import os
import glob
import shutil
import random
import tqdm
import cv2
from ultralytics import YOLO


def convert_to_yolo_format(x1, y1, x2, y2, img_width, img_height):
    """
    Convert bounding box from [x1, y1, x2, y2] to YOLO format:
    x_center, y_center, width, height normalized by image dimensions.
    """
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return x_center, y_center, width, height


def main(input_dir, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    os.mkdir(os.path.join(output_dir, "images"))
    shutil.copytree("/home/twan/Labels_annotated/labels", output_dir+"/labels")

    for file in tqdm.tqdm(glob.glob("/home/twan/Labels_annotated/labels/*.txt"), desc="Copying hand labeled images", unit="img"):
        base = file.split("/")[-1].replace(".txt", "")
        image = base + ".jpg"
        label = base + ".txt"
        shutil.copy(file, output_dir+"/labels/"+label)
        shutil.copy(input_dir+"/" +image, output_dir+"/images/"+image)

    # Load the finetuned YOLO model
    model = YOLO("yolov11s_finetuned.pt")

    # List all jpg files in the image directory
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    skipped = 0
    new = 0
    subset_images = random.sample(image_paths, int(0.5 * len(image_paths)))

    for image_path in tqdm.tqdm(list(subset_images), desc="Processing images", unit="img"):
        if os.path.exists("/home/twan/Labels_annotated/labels/" + image_path.split("/")[-1].replace(".jpg", ".txt")):
            skipped += 1
            continue
        else:
            new += 1
        # Load image dimensions using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            print(f"Unable to read {image_path}, skipping.")
            continue
        img_height, img_width, _ = image.shape

        # Run prediction on the single image
        results = model(image_path, verbose=False)

        # Convert predictions to list
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
        classes = results[0].boxes.cls.cpu().numpy().tolist()
        lines = []

        # Convert each bounding box to YOLO format and compose output lines.
        for bbox, cls in zip(boxes, classes):
            x1, y1, x2, y2 = bbox
            x_center, y_center, w, h = convert_to_yolo_format(x1, y1, x2, y2, img_width, img_height)
            line = f"{int(cls)} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            lines.append(line)

        # Save the labels to a txt file with the same base name as the image.
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        output_file = output_dir + "/labels/" + base_name + ".txt"
        with open(output_file, "w") as f:
            f.write("\n".join(lines))
        shutil.copy(image_path, output_dir + "/images/" + base_name + ".jpg")

    print(f"Skipped {skipped} images, created labels for {new} images.")


if __name__ == "__main__":
    # Path to input images and output directory for YOLO format text files
    image_dir = "/home/twan/YOLO_dataset"
    output_dir = "/home/twan/YOLO_dataset_generated"

    main(image_dir, output_dir)