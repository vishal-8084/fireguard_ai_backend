import time
import os
import cv2
import sqlite3
import threading
import subprocess
from collections import deque, Counter

from app.inference import process_frame

# =========================================
# CONFIG
# =========================================
SNAPSHOT_DIR = "static/snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

MAX_SNAPS = 5
MAX_EMAILS_PER_INCIDENT = 3
INCIDENT_TIMEOUT = 10

# =========================================
# ENGINE CLASS
# =========================================
class FireDetectionEngine:

    def __init__(self):
        self.in_incident = False
        self.incident_last_seen = 0
        self.incident_snap_count = 0
        self.email_count = 0
        self.last_snap_time = 0

        self.alarm_playing = False
        self.label_queue = deque(maxlen=7)
        self.manual_stop = False
        self.init_db()

    # -------------------------------------
    # DATABASE
    # -------------------------------------
    def init_db(self):
        conn = sqlite3.connect("database/alerts.db")
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                timestamp TEXT,
                label TEXT,
                severity INTEGER,
                snapshot_path TEXT
            )
        """)

        conn.commit()
        conn.close()

    

    def save_alert(self, timestamp, label, severity, path):
        try:
            conn = sqlite3.connect("database/alerts.db")
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO alerts (timestamp, label, severity, snapshot_path)
                VALUES (?, ?, ?, ?)
            """, (timestamp, label, severity, path))
            conn.commit()
            conn.close()
        except:
            pass

    # -------------------------------------
    # ALARM
    # -------------------------------------
    def start_alarm(self):
        if self.alarm_playing:
            return
        self.alarm_playing = True
        import winsound
        winsound.PlaySound("static/alarm-301729 (1).wav.crdownload",
                           winsound.SND_ASYNC | winsound.SND_LOOP)

    def stop_alarm(self):
        import winsound
        winsound.PlaySound(None, winsound.SND_PURGE)
        self.alarm_playing = False

    def manual_stop_alarm(self):
        self.stop_alarm()
        self.manual_stop = True

    # -------------------------------------
    # EMAIL
    # -------------------------------------
    def send_email(self, status, severity, image_path):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        args = ["node", "alert_email/sendMail.js",
                status, str(severity), timestamp, image_path]

        threading.Thread(
            target=lambda: subprocess.Popen(args),
            daemon=True
        ).start()

    # -------------------------------------
    # MAIN PROCESS
    # -------------------------------------
    def process(self, frame):

        result = process_frame(frame)

        label = result["label"]
        fire_score = result["fire_score"]
        smoke_score = result["smoke_score"]

        # TEMPORAL SMOOTHING
        self.label_queue.append(label)
        final_label = Counter(self.label_queue).most_common(1)[0][0]

        severity = 3 if final_label == "fire" else 1 if final_label == "smoke" else 0
        now = time.time()

        # -------------------------------------
        # INCIDENT LOGIC
        # -------------------------------------
        if final_label == "fire":

            if not self.in_incident:
                self.in_incident = True
                self.incident_snap_count = 0
                self.email_count = 0
                self.manual_stop = False 

            self.incident_last_seen = now
            if not self.manual_stop:
                self.start_alarm()

            # SNAPSHOT
            if self.incident_snap_count < MAX_SNAPS and (now - self.last_snap_time) > 10:

                filename = f"{int(now)}.jpg"
                path = os.path.join("static", "snapshots", filename).replace("\\", "/")
                cv2.imwrite(path, frame)

                self.save_alert(
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    "fire",
                    severity,
                    path
                )

                # EMAIL
                if self.email_count < MAX_EMAILS_PER_INCIDENT:
                    self.send_email("FIRE", severity, path)
                    self.email_count += 1

                self.incident_snap_count += 1
                self.last_snap_time = now

        elif final_label == "smoke":
            pass

            if self.in_incident and (now - self.incident_last_seen) > INCIDENT_TIMEOUT:
                self.in_incident = False

        else:
            pass

            if self.in_incident and (now - self.incident_last_seen) > INCIDENT_TIMEOUT:
                self.in_incident = False

        self.last_fire_score = fire_score
        self.last_smoke_score = smoke_score

        return {
            "label": final_label,
            "fire_score": fire_score,
            "smoke_score": smoke_score,
            "alarm": self.alarm_playing
        }

    

# =========================================
# GLOBAL ENGINE INSTANCE
# =========================================
engine = FireDetectionEngine()
