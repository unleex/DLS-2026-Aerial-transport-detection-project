from pathlib import Path
from ultralytics import YOLO

MODELS = [
    "runs/detect/yolo26x_high_res/weights/best.pt",
    "runs/detect/yolo26s_high_res/weights/best.pt",
]
MODEL_NAMES = ["Pro", "Fast"]
WEIGHT_DIR = Path("weights")
WEIGHT_DIR.mkdir(exist_ok=True)
DEVICE = "cuda"
BATCH_SIZE = 64
YOLO_YAML_PATH = Path("dataset/VisDrone.yaml")
for pretrained_model, model_name in zip(MODELS, MODEL_NAMES):
    model = YOLO(pretrained_model, verbose=True)
    model.train(
        data=YOLO_YAML_PATH,
        epochs=100,
        device=DEVICE,
        name=pretrained_model.rstrip(".pt") + "_high_res",
        exist_ok=True,
        workers=32,
        cos_lr=True,
        imgsz=1024,
        batch=BATCH_SIZE,
        resume=True,
    )
    model.save(str(WEIGHT_DIR / model_name) + ".pt")
