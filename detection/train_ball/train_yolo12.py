import os
import sys
from ultralytics import YOLO

# Redirigeix stdout i stderr a un fitxer
log_file = open('yolo12m_ball_class0.txt', 'w')
sys.stdout = log_file
sys.stderr = log_file



# Carrega el model
model = YOLO('/home-net/ccorbi/detection/train_ball/yolo12m.pt')

# Entrenament
results = model.train(
    data="/data-fast/data-server/ccorbi/dataset_tracking/ball/data.yaml",
    epochs=50,
    batch=0.7,
    imgsz=1024,
    device=4,
    patience=15,
    single_cls=True,
    cls=0,
    project='train_ball_backbone_yolo12m_class0',
    freeze=10,
    classes=[0],
    plots=True
)

# Tanquem el fitxer al final
log_file.close()
