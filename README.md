# 🚒 Fire & Smoke Detection API

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-brightgreen)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Made with YOLOv11](https://img.shields.io/badge/YOLOv11-Fire%20Detection-red)](https://ultralytics.com/yolo)

Real-time **Fire & Smoke Detection** system using **YOLOv11** + **EfficientNet-B4**. Features live camera streaming, video processing, intelligent alerts (email, sound alarm, database), and temporal smoothing for reliable detection.

## ✨ Features

- 🔴 **Dual-Model Detection**: YOLOv11 (object detection) + EfficientNet-B4 (classification)
- 📹 **Live Camera Streaming** with real-time overlays (fire/smoke labels + confidence)
- 🎥 **Video Upload & Processing** with frame-by-frame analysis
- 🖼️ **Single Image Prediction**
- 🚨 **Smart Alert System**:
  - SQLite database storage
  - Automatic email notifications (Node.js)
  - Windows sound alarm
  - Snapshot capture (max 5 per incident)
- ⏱️ **Temporal Smoothing** (7-frame queue) for stable predictions
- 📊 **Alert Dashboard** with image gallery
- 🌐 **CORS-enabled API** for web/mobile integration
- 💾 **Production-ready** with static file serving

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone <your-repo>
cd Backend
pip install -r requirements.txt
```

### 2. Download Models
Models are loaded from `models/`:
```
models/yolo11-d-fire-dataset.pt      # Custom YOLOv11 fire/smoke
models/efficientnet_fire_smoke_v2.pth # EfficientNet-B4 classifier
```

### 3. Run Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Visit [http://localhost:8000](http://localhost:8000) 🚀

### 4. Access Docs
- [Interactive API Docs](http://localhost:8000/docs)
- [Static Files](http://localhost:8000/static/)

## 📖 API Endpoints

| Endpoint | Method | Description | Parameters |
|----------|--------|-------------|------------|
| `/` | GET | Health check | - |
| `/live` | GET | Live camera stream | - |
| `/video-stream` | GET | Video processing stream | - |
| `/predict-image` | POST | Analyze single image | `file` (image) |
| `/start-camera` | POST | Start camera feed | `source` (int, default=0) |
| `/stop-camera` | POST | Stop camera | - |
| `/start-video` | POST | Upload & process video | `file` (video) |
| `/stop-video` | POST | Stop video | - |
| `/status` | GET | Current detection status | - |
| `/alerts` | GET | Recent alerts (last 50) | - |
| `/stop-alarm` | POST | Stop sound alarm | - |
| `/delete-alert?id={id}` | POST | Delete alert & image | `id` (str) |
| `/list-cameras` | GET | Available cameras | - |

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐
│   FastAPI App   │───▶│   /docs (Swagger) │
└─────────────────┘    └──────────────────┘
         │
         ▼
┌─────────────────┐    ┌──────────────────┐
│  Camera/Video   │───▶│ YOLOv11 Detector  │──┐
│     Input       │    │   + Crop ROIs     │   │
└─────────────────┘    └──────────────────┘   ▼
                                               │
                                    ┌──────────────────┐
                                    │ EfficientNet-B4   │
                                    │   Classification  │
                                    └──────────────────┘
                                               │
                                               ▼
                                    ┌──────────────────┐
                                    │ Temporal Smoothing│ (7-frame queue)
                                    └──────────────────┘
                                               │
              ┌──────────────────────┼──────────────────────┐
              │                      ▼                      │
    ┌─────────▼─────────┐    ┌─────────▼─────────┐    ┌─────▼──────┐
    │   SQLite Alerts   │    │   Email Alert      │    │ Sound Alarm│
    │ (snapshots, meta) │    │ (Node.js script)   │    │ (Windows)  │
    └──────────────────┘    └────────────────────┘    └────────────┘
```

## 🤖 AI Models

### 1. **YOLOv11** (`yolo11-d-fire-dataset.pt`)
- **Role**: Bounding box detection for fire/smoke regions
- **Classes**: `smoke` (0), `fire` (1)
- **Confidence**: ≥0.25

### 2. **EfficientNet-B4** (`efficientnet_fire_smoke_v2.pth`)
- **Role**: Binary classification on YOLO crops + full frame
- **Classes**: `fire`, `normal`, `smoke`
- **Scoring**: Combined confidence → final label

### Detection Logic
```
fire_score = (YOLO_fire_detections × 0.5) + (EffNet_fire × 0.6) + (Full_frame_fire × 0.3)
smoke_score = similar logic...

Thresholds:
- fire_score > 0.7 → "fire"
- smoke_score > 0.6 → "smoke" 
- fire_score > 0.4 → "warning"
- else → "normal"
```

## 📊 Database Schema

**`database/alerts.db`**
```sql
CREATE TABLE alerts (
    timestamp TEXT,
    label TEXT,        -- "fire" | "smoke"
    severity INTEGER,  -- 3 (fire) | 1 (smoke)
    snapshot_path TEXT -- /static/snapshots/12345.jpg
);
```

## 🚨 Alert System

1. **Incident Detection** → Start alarm + capture snapshots (max 5)
2. **Email Alerts** → Node.js `sendmail.js` (max 3 per incident)
3. **Timeout** → 10s after last detection
4. **Manual Stop** → Clears incident state

**Email Trigger**: `node alert_email/sendmail.js "FIRE" 3 "2024-..." "/static/snapshots/123.jpg"`


```
[Live Stream Overlay]  [Alert Dashboard]  [API Docs]
```

## 🛠️ Development

### Run with Reload
```bash
uvicorn app.main:app --reload
```

### Available Cameras
```
GET /list-cameras
```

### Custom Models
Replace files in `models/` and restart.

### Email Setup
Configure `alert_email/sendmail.js` with your SMTP credentials.

## 🤝 Contributing

1. Fork the repo
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [FastAPI](https://fastapi.tiangolo.com/)
- [PyTorch](https://pytorch.org/)
- [timm](https://github.com/huggingface/pytorch-image-models)

---

⭐ **Star this repo if you found it useful!** ⭐
