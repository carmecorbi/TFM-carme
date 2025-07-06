import os
import json
import shutil
from tqdm import tqdm

# Configuració
IMG_DIR = '/data-fast/data-server/ccorbi/ball_singleclass/images'
LBL_DIR = '/data-fast/data-server/ccorbi/ball_singleclass/labels'
OUTPUT_DIR = '/data-fast/data-server/ccorbi/ball_detr'
IMG_WIDTH = 1920
IMG_HEIGHT = 1080

# Classes (ara amb id=1 en lloc de 0)
CATEGORIES = [{"id": 0, "name": "ball"}]

def yolo_to_coco_bbox(x_center, y_center, width, height):
    x = (x_center - width / 2) * IMG_WIDTH
    y = (y_center - height / 2) * IMG_HEIGHT
    w = width * IMG_WIDTH
    h = height * IMG_HEIGHT
    return [x, y, w, h]

def process_split(split):
    img_id = 0
    ann_id = 0
    images = []
    annotations = []

    img_out_dir = os.path.join(OUTPUT_DIR, f'{split}2017')
    os.makedirs(img_out_dir, exist_ok=True)

    split_img_root = os.path.join(IMG_DIR, split)
    split_lbl_root = os.path.join(LBL_DIR, split)

    for seq in tqdm(sorted(os.listdir(split_img_root)), desc=f"Processing {split}"):
        seq_img_dir = os.path.join(split_img_root, seq)
        seq_lbl_dir = os.path.join(split_lbl_root, seq)
        if not os.path.isdir(seq_img_dir):
            continue

        for img_file in sorted(os.listdir(seq_img_dir)):
            if not img_file.endswith('.jpg'):
                continue

            # Image info
            image_path = os.path.join(seq_img_dir, img_file)
            out_image_path = os.path.join(img_out_dir, f"{split}_{seq}_{img_file}")
            shutil.copy(image_path, out_image_path)

            images.append({
                "id": img_id,
                "file_name": os.path.basename(out_image_path),
                "width": IMG_WIDTH,
                "height": IMG_HEIGHT
            })

            # Annotation info
            label_file = img_file.replace('.jpg', '.txt')
            label_path = os.path.join(seq_lbl_dir, label_file)

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        if line.strip() == '':
                            continue
                        parts = list(map(float, line.strip().split()))
                        cls, x, y, w, h = parts
                        bbox = yolo_to_coco_bbox(x, y, w, h)
                        annotations.append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": 0,  # Sempre 1
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0
                        })
                        ann_id += 1

            img_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES
    }

def main():
    os.makedirs(os.path.join(OUTPUT_DIR, 'annotations'), exist_ok=True)

    for split in ['train', 'val', 'test']:
        coco_dict = process_split(split)
        with open(os.path.join(OUTPUT_DIR, 'annotations', f'instances_{split}2017.json'), 'w') as f:
            json.dump(coco_dict, f)

if __name__ == "__main__":
    main()
