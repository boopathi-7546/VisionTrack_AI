# 🎯 VisionTrack AI — Smart Object Detection & Tracking System

Real-time multi-object detection and tracking powered by **YOLOv8 + DeepSORT**, wrapped in a modern cyberpunk-inspired glassmorphism dashboard built with Flask.

---

## 🚀 Features

| Category        | Details                                                                |
| --------------- | ---------------------------------------------------------------------- |
| Detection       | YOLOv8n (80 COCO classes), confidence filtering, neon bounding boxes   |
| Tracking        | DeepSORT unique IDs, motion trails, multi-object simultaneous tracking |
| Analytics       | FPS counter, total detections, label frequency, average confidence     |
| Heatmap         | Gaussian accumulation overlay for crowd density visualization          |
| Zone Monitoring | Polygon-based restricted zones with live alerts                        |
| Counting Line   | Inbound / Outbound object counting                                     |
| Snapshots       | Save annotated frames                                                  |
| Reports         | Export detection logs as CSV or JSON                                   |
| Input Sources   | Webcam & video upload                                                  |
| UI              | Cyberpunk glassmorphism dashboard with animated effects                |

---

## 🛠️ Tech Stack

### Backend

* Python
* Flask

### Computer Vision & AI

* YOLOv8
* DeepSORT
* OpenCV
* PyTorch
* NumPy

### Frontend

* HTML5
* CSS3
* JavaScript

---

## 📂 Project Structure

```text
VisionTrack_AI/
├── app.py
├── detector.py
├── tracker.py
├── analytics.py
├── requirements.txt
├── README.md
│
├── models/
│
├── static/
│   ├── css/style.css
│   ├── js/script.js
│   ├── uploads/
│   ├── reports/
│   └── snapshots/
│
├── templates/
│   └── index.html
│
└── logs/
    └── visiontrack.log
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/boopathi-7546/VisionTrack_AI.git
cd VisionTrack_AI
```

### Create Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Open:

```text
http://localhost:5000
```

---

## 📸 Screenshots

### Dashboard

*Add dashboard screenshot here*

### Live Detection

*Add webcam detection screenshot here*

### Analytics

*Add analytics screenshot here*

---

## 🎮 Usage

### Webcam Mode

* Click **Start Webcam**
* Allow camera access
* Real-time detection begins instantly

### Video Upload Mode

* Click **Upload Video**
* Select supported video file
* Analyze uploaded footage

Supported formats:

```text
.mp4 .avi .mov .mkv .webm
```

---

## 📊 API Endpoints

| Method | Endpoint          | Description     |
| ------ | ----------------- | --------------- |
| GET    | /video_feed       | MJPEG stream    |
| POST   | /api/start_webcam | Start webcam    |
| POST   | /api/upload_video | Upload video    |
| POST   | /api/stop         | Stop stream     |
| GET    | /api/stats        | Live statistics |
| POST   | /api/snapshot     | Save snapshot   |
| POST   | /api/export_csv   | Export report   |
| POST   | /api/export_json  | Export report   |

---

## 🔮 Future Enhancements

* Face Recognition
* Cloud Deployment
* Database Integration
* Mobile Dashboard
* Email / SMS Alerts
* Advanced Heatmap Analytics
* AI-Based Behavioral Analysis

---

## 👨‍💻 Author

### Boopathi P

Aspiring AI & Data Science Engineer passionate about:

* Artificial Intelligence
* Computer Vision
* Machine Learning
* Data Analytics
* Software Development

GitHub:
https://github.com/boopathi-7546

---

## ⭐ Support

If you found this project useful:

* ⭐ Star the repository
* 🍴 Fork the project
* 🤝 Contribute improvements

---

## 📜 License

This project is licensed under the MIT License.
