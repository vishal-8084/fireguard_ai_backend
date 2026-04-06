import cv2
import numpy as np
import time
from collections import deque, Counter
from PIL import Image
import torch
from torchvision import transforms

# IMPORT MODELS
from app.model import yolo_model, model, DEVICE, CLASS_NAMES, YOLO_CLASS_MAP

# =========================================
# STATE (same as your previous)
# =========================================
label_queue = deque(maxlen=7)

# =========================================
# TRANSFORM
# =========================================
transform = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor()
])

# =========================================
# EFFICIENTNET PREDICTION
# =========================================
def predict_effnet(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_t = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs[0], dim=0)
        conf, pred = torch.max(probs, 0)

    return CLASS_NAMES[pred.item()], conf.item()

# =========================================
# MAIN PROCESS FUNCTION 
# =========================================
def process_frame(frame):
    h, w, _ = frame.shape

    fire_score = 0
    smoke_score = 0

    # YOLO DETECTION
    results = yolo_model(frame, conf=0.25, verbose=False)

    for b in results[0].boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        cls_id = int(b.cls[0].item())
        cls_name = YOLO_CLASS_MAP.get(cls_id, "unknown")

        # expand crop
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        eff_label, eff_conf = predict_effnet(crop)

        # SCORING LOGIC
        if cls_name == "fire":
            fire_score += 0.5

        if eff_label == "fire":
            fire_score += eff_conf * 0.6

        elif eff_label == "smoke":
            smoke_score += eff_conf * 0.6

    # FULL FRAME CHECK
    full_label, full_conf = predict_effnet(frame)

    if full_label == "fire":
        fire_score += full_conf * 0.3
    elif full_label == "smoke":
        smoke_score += full_conf * 0.3

    # FINAL DECISION
    if fire_score > 0.7:
        temp_label = "fire"
    elif smoke_score > 0.6:
        temp_label = "smoke"
    elif fire_score > 0.4:
        temp_label = "warning"
    else:
        temp_label = "normal"

    # TEMPORAL SMOOTHING
    label_queue.append(temp_label)
    final_label = Counter(label_queue).most_common(1)[0][0]

    return {
        "label": final_label,
        "fire_score": float(fire_score),
        "smoke_score": float(smoke_score)
    }