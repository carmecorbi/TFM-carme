import os
import sys
from ultralytics import YOLO

# Redirigeix stdout i stderr a un fitxer
log_file = open('entrenament_log_yolo12s_frozen.txt', 'w')
sys.stdout = log_file
sys.stderr = log_file


# Carrega el model
model = YOLO('yolo12s.pt')

# Entrenament
results = model.train(
    data="/data-fast/data-server/ccorbi/dataset_tracking/full/data.yaml",
    epochs=50,
    batch=-1,
    imgsz=1024,
    patience=10,
    project='train_full_backbonefrozen',
    device=1,
    freeze=10,
    classes=[0, 1],
    plots=True
)

# Tanquem el fitxer al final
log_file.close()
