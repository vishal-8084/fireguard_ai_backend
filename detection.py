import cv2
import torch
import torch.nn as nn
import timm
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from collections import deque, Counter
import threading
import time
import os
import sqlite3
import winsound
import subprocess

# =========================================
# CONFIGURATION
# =========================================
YOLO_MODEL_PATH = "models/yolo11-d-fire-dataset.pt"
EFFNET_PATH = "models/efficientnet_fire_smoke_v2.pth"

ALARM_SOUND = "static/alarm-301729 (1).wav.crdownload"
SNAPSHOT_DIR = "static/snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ['fire', 'normal', 'smoke']
YOLO_CLASS_MAP = {0: "smoke", 1: "fire"}

EMAIL_COOLDOWN = 600
MAX_SNAPS = 5
MAX_EMAILS_PER_INCIDENT = 3
INCIDENT_TIMEOUT = 10

# =========================================
# STATE TRACKING (IMPORTANT)
# =========================================
alarm_playing = False
manual_alarm_override = False

in_incident = False
incident_snap_count = 0
incident_last_seen = 0
email_count_this_incident = 0
last_email_time = 0
last_snap_time = 0

label_queue = deque(maxlen=7)

# =========================================
# MODEL LOADING
# =========================================
yolo_model = YOLO(YOLO_MODEL_PATH)

model = timm.create_model('efficientnet_b4', pretrained=False)
in_features = model.classifier.in_features

model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, 3)
)

model.load_state_dict(torch.load(EFFNET_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor()
])

# =========================================
# CORE FUNCTIONS
# =========================================
def predict_effnet(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_t = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs[0], dim=0)
        conf, pred = torch.max(probs, 0)

    return CLASS_NAMES[pred.item()], conf.item()

def start_alarm():
    global alarm_playing, manual_alarm_override
    if alarm_playing or manual_alarm_override:
        return
    alarm_playing = True
    winsound.PlaySound(ALARM_SOUND, winsound.SND_ASYNC | winsound.SND_LOOP)

def stop_alarm():
    global alarm_playing
    winsound.PlaySound(None, winsound.SND_PURGE)
    alarm_playing = False

def send_email(status, severity, image_path=None):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    args = ["node", "alert_email/sendMail.js", status, str(severity), timestamp]
    if image_path:
        args.append(image_path)

    threading.Thread(target=lambda: subprocess.Popen(args), daemon=True).start()

def save_alert_to_db(timestamp, label, severity, snapshot_path):
    try:
        conn = sqlite3.connect("database/alerts.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO alerts (timestamp, label, severity, snapshot_path)
            VALUES (?, ?, ?, ?)
        """, (timestamp, label, severity, snapshot_path))
        conn.commit()
        conn.close()
    except:
        pass

# =========================================
# MAIN PROCESS
# =========================================
def process_frame(frame):
    global in_incident, incident_snap_count, incident_last_seen
    global email_count_this_incident, last_email_time, last_snap_time

    h, w, _ = frame.shape
    fire_score = 0
    smoke_score = 0

    # YOLO
    results = yolo_model(frame, conf=0.25, verbose=False)

    for b in results[0].boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        cls_id = int(b.cls[0].item())
        cls_name = YOLO_CLASS_MAP.get(cls_id, "unknown")

        pad = 10
        crop = frame[max(0,y1-pad):min(h,y2+pad),
                     max(0,x1-pad):min(w,x2+pad)]

        if crop.size == 0:
            continue

        eff_label, eff_conf = predict_effnet(crop)

        if cls_name == "fire":
            fire_score += 0.5

        if eff_label == "fire":
            fire_score += eff_conf * 0.6
        elif eff_label == "smoke":
            smoke_score += eff_conf * 0.6

    # fallback
    full_label, full_conf = predict_effnet(frame)

    if full_label == "fire":
        fire_score += full_conf * 0.3
    elif full_label == "smoke":
        smoke_score += full_conf * 0.3

    # decision
    if fire_score > 0.7:
        temp_label = "fire"
    elif smoke_score > 0.6:
        temp_label = "smoke"
    elif fire_score > 0.4:
        temp_label = "warning"
    else:
        temp_label = "normal"

    label_queue.append(temp_label)
    final_label = Counter(label_queue).most_common(1)[0][0]

    # =====================================
    # INCIDENT + ALERT SYSTEM
    # =====================================
    severity = 3 if final_label=="fire" else 1 if final_label=="smoke" else 0
    now = time.time()

    if final_label == "fire":

        if not in_incident:
            in_incident = True
            incident_snap_count = 0
            email_count_this_incident = 0

        incident_last_seen = now
        start_alarm()

        # SNAPSHOT CONTROL
        if incident_snap_count < MAX_SNAPS and (now-last_snap_time)>10:
            path = os.path.join(SNAPSHOT_DIR, f"{int(now)}.jpg")
            cv2.imwrite(path, frame)

            save_alert_to_db(time.strftime("%Y-%m-%d %H:%M:%S"),
                             "fire", severity, path)

            # EMAIL CONTROL
            if (email_count_this_incident < MAX_EMAILS_PER_INCIDENT and
                (now - last_email_time > EMAIL_COOLDOWN)):

                send_email("FIRE", severity, path)
                email_count_this_incident += 1
                last_email_time = now

            incident_snap_count += 1
            last_snap_time = now

    else:
        stop_alarm()

        if in_incident and (now - incident_last_seen > INCIDENT_TIMEOUT):
            in_incident = False

    # =====================================
    # DISPLAY
    # =====================================
    color = (0,0,255) if final_label=="fire" else \
            (150,150,150) if final_label=="smoke" else \
            (0,255,0)

    cv2.putText(frame, f"STATUS: {final_label.upper()}",
                (10,35), 1, 1.5, color, 2)

    cv2.putText(frame, f"ALARM: {'ON' if alarm_playing else 'OFF'}",
                (10,70), 1, 1.2,
                (0,0,255) if alarm_playing else (0,255,0), 2)

    return frame

# =========================================
# MAIN LOOP
# =========================================
if __name__ == "__main__":
    cap = cv2.VideoCapture("videos/posVideo9.878.avi")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)
        cv2.imshow("Fire Guard System", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('m'):   # 🔥 ADD THIS
            manual_alarm_override = True
            stop_alarm()
            print("🛑 Alarm manually stopped")
    stop_alarm()
    cap.release()
    cv2.destroyAllWindows()