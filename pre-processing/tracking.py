import os
import json

# Ruta base on es troben les seqüències
base_path = "/data-fast/data-server/ccorbi/SoccerNetGS/gamestate-2024/valid"
output_path = "/data-fast/data-server/ccorbi/SN-Tracking/tracking/val"

IMG_WIDTH = 1920
IMG_HEIGHT = 1080

for seq_name in os.listdir(base_path):
    seq_path = os.path.join(base_path, seq_name)
    json_path = os.path.join(seq_path, "Labels-GameState.json")

    if not os.path.isfile(json_path):
        continue

    # Crear carpeta GT
    seq_output_path = os.path.join(output_path, seq_name)
    gt_dir = os.path.join(seq_output_path, "gt")
    os.makedirs(gt_dir, exist_ok=True)

    gt_file_path = os.path.join(gt_dir, "gt.txt")

    # Carregar JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    gt_entries = []

    for annotation in data.get("annotations", []):
        image_id = annotation["image_id"]
        frame_id = int(image_id[-3:])  # Últims 3 dígits

        category_id = annotation["category_id"]
        if category_id not in [1, 2, 3, 4]:
            continue  # Només jugadors i porters

        if "track_id" not in annotation:
            continue  # Si no té tracklet_id, saltar

        bbox = annotation["bbox_image"]
        x = int(bbox["x_center"] - bbox["w"] / 2)
        y = int(bbox["y_center"] - bbox["h"] / 2)
        w = int(bbox["w"])
        h = int(bbox["h"])
        track_id = annotation["track_id"]

        gt_entries.append((track_id, frame_id, x, y, w, h))

    # Ordenar per track_id i frame_id
    gt_entries.sort(key=lambda tup: (tup[0], tup[1]))

    # Escriure fitxer gt.txt
    with open(gt_file_path, 'w') as gt_file:
        for track_id, frame_id, x, y, w, h in gt_entries:
            gt_file.write(f"{frame_id},{track_id},{x},{y},{w},{h},1,-1,-1,-1\n")
