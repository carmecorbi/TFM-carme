import os
from PIL import Image
from datasets import Dataset, DatasetDict
from tqdm import tqdm

def yolo_to_xywh(yolo_box, img_width, img_height):
    """Converteix anotació YOLO a bounding box en píxels [x, y, w, h]."""
    class_id, x_center, y_center, w, h = map(float, yolo_box)
    x = (x_center - w / 2) * img_width
    y = (y_center - h / 2) * img_height
    return [int(x), int(y), int(w * img_width), int(h * img_height)]

def load_yolo_dataset(images_dir, labels_dir):
    data = []
    image_id_counter = 0

    for seq_folder in tqdm(sorted(os.listdir(images_dir)), desc=f"Loading {images_dir}"):
        seq_img_path = os.path.join(images_dir, seq_folder)
        seq_lbl_path = os.path.join(labels_dir, seq_folder)
        if not os.path.isdir(seq_img_path):
            continue

        for img_name in sorted(os.listdir(seq_img_path)):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(seq_img_path, img_name)
            lbl_path = os.path.join(seq_lbl_path, os.path.splitext(img_name)[0] + ".txt")

            with Image.open(img_path) as img:
                width, height = img.size

                objects = {
                    "id": [],
                    "area": [],
                    "bbox": [],
                    "category": []
                }

                if os.path.exists(lbl_path):
                    with open(lbl_path, "r") as f:
                        for i, line in enumerate(f):
                            parts = line.strip().split()
                            if len(parts) != 5:
                                continue
                            bbox = yolo_to_xywh(parts, width, height)
                            area = bbox[2] * bbox[3]
                            objects["id"].append(i)
                            objects["area"].append(area)
                            objects["bbox"].append(bbox)
                            objects["category"].append(0)  # només una classe

                data.append({
                    "image_id": image_id_counter,
                    "image": img.copy(),
                    "width": width,
                    "height": height,
                    "objects": objects
                })
                image_id_counter += 1
    return data

base_path = "/data-fast/data-server/ccorbi/ball_singleclass"

train_data = load_yolo_dataset(
    images_dir=os.path.join(base_path, "images/train"),
    labels_dir=os.path.join(base_path, "labels/train")
)
val_data = load_yolo_dataset(
    images_dir=os.path.join(base_path, "images/val"),
    labels_dir=os.path.join(base_path, "labels/val")
)
test_data = load_yolo_dataset(
    images_dir=os.path.join(base_path, "images/test"),
    labels_dir=os.path.join(base_path, "labels/test")
)

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)
test_dataset = Dataset.from_list(test_data)

dataset_dict = DatasetDict({
    "train": train_dataset,
    "val": val_dataset,
    "test": test_dataset
})

dataset_dict.push_to_hub("ccorbi/ball-detection-singleclass")
