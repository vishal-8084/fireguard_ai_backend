import torch
import torch.nn as nn
import timm
from ultralytics import YOLO

# =========================================
# CONFIG
# =========================================
YOLO_MODEL_PATH = "models/best_100epoch.pt"
EFFNET_PATH = "models/efficientnet_fire_smoke_v2.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ['fire', 'normal', 'smoke']
YOLO_CLASS_MAP = {0: "smoke", 1: "fire"}

# =========================================
# LOAD MODELS (RUNS ONLY ONCE)
# =========================================
print("🔥 Loading YOLO model...")
yolo_model = YOLO(YOLO_MODEL_PATH)

print("🧠 Loading EfficientNet...")
model = timm.create_model('efficientnet_b4', pretrained=False)

in_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, 3)
)

model.load_state_dict(torch.load(EFFNET_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("✅ Models loaded successfully!")