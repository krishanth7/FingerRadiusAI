# 🖐️ FingerRadiusAI

**Real-time Hand Finger Radius Graph Visualization System using AI Hand Tracking**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10%2B-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Project Overview

**FingerRadiusAI** is a professional Python computer vision application that detects hand landmarks in real-time using Google's MediaPipe Tasks API, computes finger radius values (Euclidean distances between fingertips), and renders a live scrolling graph alongside a corporate-styled dashboard overlay.

The system tracks all **21 hand landmarks**, calculates distances between adjacent fingertip pairs and wrist-to-tip pairs, classifies hand gestures (Open / Closed / Pinch / Partial), and displays everything in a sleek, professional analytics dashboard.

---

## ✨ Features

### Core
- ✅ Real-time hand landmark detection via MediaPipe Tasks API
- ✅ Track all 21 hand landmarks accurately
- ✅ Calculate finger radius (Euclidean distance) between:
  - Thumb tip ↔ Index tip
  - Index tip ↔ Middle tip
  - Middle tip ↔ Ring tip
  - Ring tip ↔ Pinky tip
  - Wrist ↔ each fingertip
- ✅ Dynamic radius circles on video feed
- ✅ Connecting lines between landmarks
- ✅ Live numerical radius values near fingers
- ✅ Real-time scrolling graph (Radius vs Time)
- ✅ Separate color per finger pair on graph
- ✅ FPS counter with status indicator
- ✅ EMA smoothing for stable tracking

### Advanced
- ✅ Hand gesture classification (Open / Closed / Pinch / Partial)
- ✅ Radius data recording over time
- ✅ One-key CSV export
- ✅ Motion history trails on fingertips
- ✅ Hand status badge overlay
- ✅ Professional corporate dashboard UI
- ✅ Side analytics panel with live stats, radius bars, and controls

---

## 📂 Project Structure

```
FingerRadiusAI/
│
├── src/
│   ├── __init__.py            # Package init
│   ├── hand_tracker.py        # MediaPipe hand detection & skeleton drawing
│   ├── radius_calculator.py   # Distance computation & gesture detection
│   ├── graph_visualizer.py    # Real-time OpenCV graph renderer
│   └── utils.py               # Smoothing, FPS, CSV export, UI helpers
│
├── models/
│   └── hand_landmarker.task   # MediaPipe hand landmark model
│
├── main.py                    # Application entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- A webcam / USB camera
- pip (Python package manager)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/krishanth7/FingerRadiusAI.git
   cd FingerRadiusAI
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the hand landmark model** (if not already present)
   ```bash
   # Windows PowerShell
   New-Item -ItemType Directory -Force -Path models
   Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile "models/hand_landmarker.task"

   # macOS / Linux
   mkdir -p models
   curl -o models/hand_landmarker.task -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
   ```

---

## 🚀 Usage

### Run the application
```bash
python main.py
```

### Keyboard Controls

| Key         | Action                     |
|-------------|----------------------------|
| `Q` / `ESC` | Quit the application      |
| `E`         | Export recorded data to CSV |
| `R`         | Reset data buffers         |
| `T`         | Toggle motion trails       |
| `G`         | Toggle graph panel         |
| `S`         | Take a screenshot          |

### Output
- **Live Window** — Side analytics panel + video feed with overlays + scrolling radius graph
- **CSV Export** — Press `E` to save `radius_data.csv` with timestamps, radius values, and hand status
- **Screenshots** — Press `S` to save a timestamped PNG of the current composite view

---

## 🏗️ Architecture

```
Camera Frame
     │
     ▼
 HandTracker           ←  MediaPipe Tasks API  (detection + EMA smoothing)
     │
     ├──▶ landmarks (21 points)
     │
     ▼
 RadiusCalculator      ←  Euclidean distances + hand classification
     │
     ├──▶ pair_radii, wrist_radii, hand_status
     │
     ▼
 GraphVisualizer       ←  OpenCV-rendered scrolling chart
     │
     ▼
 Composite Display     ←  Analytics Panel + Video Feed + Graph
```

---

## 🔮 Future Improvements

- [ ] **Multi-hand support** — Track and display radii for both hands simultaneously
- [ ] **3D radius** — Use MediaPipe z-coordinates for depth-aware distance
- [ ] **Gesture library** — Recognize more gestures (peace, thumbs-up, OK, pointing)
- [ ] **PyQt / Tkinter GUI** — Windowed UI with settings panel and playback controls
- [ ] **Video file input** — Process pre-recorded video files instead of live camera
- [ ] **Data visualization dashboard** — Export data and render interactive plots with Plotly
- [ ] **Real-time audio feedback** — Map radius values to sound parameters for accessibility
- [ ] **GPU acceleration** — ONNX Runtime for higher FPS on supported hardware
- [ ] **Kalman filter** — Replace EMA with Kalman filter for more accurate smoothing
- [ ] **Custom themes** — Switchable color themes (cyberpunk, minimal, retro)

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

<p align="center">
  Built with ❤️ using Python, OpenCV & MediaPipe
</p>