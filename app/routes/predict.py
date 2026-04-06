from fastapi import APIRouter, UploadFile, File
import numpy as np
import cv2
import sqlite3
import os
import time
import threading
import subprocess

from fastapi.responses import StreamingResponse
from app.services.detection_engine import engine
from app.inference import process_frame

router = APIRouter()

# ================= GLOBAL STATES =================
camera_running = False
camera_source = 0
camera_cap = None   

video_path = None
video_running = False

latest_frame = None
latest_result = {
    "label": "normal",
    "fire_score": 0,
    "smoke_score": 0
}

camera_frame = None
camera_result = {
    "label": "normal",
    "fire_score": 0,
    "smoke_score": 0
}

# ================= CAMERA AI THREAD =================
def process_camera_frames():
    global camera_frame, camera_result, camera_running

    while camera_running:
        if camera_frame is None:
            time.sleep(0.01)
            continue

        frame = cv2.resize(camera_frame, (640, 480))
        camera_result = engine.process(frame)


# ================= CAMERA STREAM =================
def generate_frames():
    global camera_running, camera_source, camera_frame, camera_result, camera_cap

    # OPEN CAMERA ONCE
    camera_cap = cv2.VideoCapture(camera_source, cv2.CAP_DSHOW)

    if not camera_cap.isOpened():
        print(f"❌ Cannot open camera {camera_source}")
        return

    try:
        while camera_running:
            success, frame = camera_cap.read()

            if not success:
                print("Camera read failed")
                break

            camera_frame = frame.copy()
            label = camera_result["label"]

            #  COLOR LOGIC
            if label == "fire":
                color = (0, 0, 255)      
            elif label == "smoke":
                color = (255, 0, 0)      
            else:
                color = (0, 255, 0)      

            cv2.putText(frame, label.upper(), (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color, 2)

            _, buffer = cv2.imencode('.jpg', frame)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')

            time.sleep(0.03)

    except GeneratorExit:
        print(" Client disconnected stream")

    finally:
        if camera_cap is not None:
            camera_cap.release()
            print("✅ Camera released")


# ================= CAMERA CONTROL =================
@router.post("/start-camera")
def start_camera(source: int = 0):
    global camera_running, camera_source, camera_cap, camera_frame

    # FULL RESET
    camera_running = False

    if camera_cap is not None:
        camera_cap.release()
        camera_cap = None

    time.sleep(1)  # VERY IMPORTANT

    camera_frame = None
    camera_source = source
    camera_running = True

    threading.Thread(target=process_camera_frames, daemon=True).start()

    return {"status": f"camera {source} started"}


@router.post("/stop-camera")
def stop_camera():
    global camera_running, camera_cap

    camera_running = False

    if camera_cap is not None:
        camera_cap.release()
        camera_cap = None

    time.sleep(0.5)

    engine.manual_stop_alarm()

    return {"status": "camera stopped"}


# ================= CAMERA LIST =================
@router.get("/list-cameras")
def list_cameras():
    cameras = []

    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            cameras.append({"id": i, "name": f"Camera {i}"})
            cap.release()

    return cameras


@router.get("/live")
def live_stream():
    return StreamingResponse(generate_frames(),
                             media_type='multipart/x-mixed-replace; boundary=frame')


# ================= STATUS =================
@router.get("/status")
def get_status():
    return {
        "label": engine.label_queue[-1] if engine.label_queue else "normal",
        "alarm": engine.alarm_playing,
        "fire_score": getattr(engine, "last_fire_score", 0),
        "smoke_score": getattr(engine, "last_smoke_score", 0),
        "email_sent": getattr(engine, "email_count", 0) > 0
    }


# ================= IMAGE =================
@router.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image"}

    result = process_frame(frame)

    return {
        "label": result["label"],
        "fire_score": result["fire_score"],
        "smoke_score": result["smoke_score"]
    }


# ================= VIDEO THREAD =================
def process_frames():
    global latest_frame, latest_result, video_running

    while video_running:
        if latest_frame is None:
            time.sleep(0.01)
            continue

        frame = cv2.resize(latest_frame, (640, 480))
        latest_result = engine.process(frame)


# ================= VIDEO START =================
@router.post("/start-video")
async def start_video(file: UploadFile = File(...)):
    global video_path, video_running, latest_frame

    if video_path and os.path.exists(video_path):
        os.remove(video_path)

    video_path = f"temp_{file.filename}"

    with open(video_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        fallback_path = "temp_video.mp4"

        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vcodec", "libx264",
            "-acodec", "aac",
            fallback_path
        ])

        video_path = fallback_path

    cap.release()

    video_running = True
    latest_frame = None

    threading.Thread(target=process_frames, daemon=True).start()

    return {"status": "video started"}


# ================= VIDEO STOP =================
@router.post("/stop-video")
def stop_video():
    global video_running

    video_running = False
    engine.manual_stop_alarm()

    return {"status": "video stopped"}


# ================= VIDEO STREAM =================
def generate_video_frames():
    global video_path, video_running, latest_frame, latest_result

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25

    delay = 1.0 / fps

    while video_running:
        start = time.time()

        success, frame = cap.read()
        if not success:
            break

        latest_frame = frame.copy()
        label = latest_result["label"]

        if label == "fire":
            color = (0, 0, 255)
        elif label == "smoke":
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)

        cv2.putText(frame, label.upper(), (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color, 2)

        _, buffer = cv2.imencode('.jpg', frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

        elapsed = time.time() - start
        if delay - elapsed > 0:
            time.sleep(delay - elapsed)

    cap.release()


@router.get("/video-stream")
def video_stream():
    return StreamingResponse(generate_video_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


# ================= ALERTS =================
@router.get("/alerts")
def get_alerts():
    conn = sqlite3.connect("database/alerts.db")
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT rowid, * FROM alerts ORDER BY timestamp DESC LIMIT 50")
        rows = cursor.fetchall()
    except:
        rows = []

    conn.close()

    alerts = []

    for r in rows:
        # FIX BACKSLASH ISSUE
        clean_path = r[4].replace("\\", "/")

        alerts.append({
            "id": str(r[0]),  # unique ID
            "timestamp": r[1],
            "label": r[2],   # fire / smoke
            "severity": r[3],
            "image_url": f"http://localhost:8000/{clean_path}"
        })

    return alerts

@router.post("/stop-alarm")
def stop_alarm():
    engine.manual_stop_alarm()
    return {"status": "alarm stopped"}


@router.post("/delete-alert")
def delete_alert(id: str):
    try:
        conn = sqlite3.connect("database/alerts.db")
        cursor = conn.cursor()

        # Get file path using ID
        cursor.execute("SELECT snapshot_path FROM alerts WHERE rowid=?", (id,))
        row = cursor.fetchone()

        if row:
            file_path = row[0]

            #  Delete image file
            if os.path.exists(file_path):
                os.remove(file_path)

        # Delete DB record
        cursor.execute("DELETE FROM alerts WHERE rowid=?", (id,))
        conn.commit()
        conn.close()

        return {"status": "deleted"}

    except Exception as e:
        print("Delete error:", e)
        return {"status": "error"}