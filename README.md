<div align="center">

# ⚽ Football Analytics CV

### CV Football Analytics from Broadcast Videos

Transform football broadcast footage into real-time player tracking, tactical insights, and match analytics using Computer Vision and Deep Learning.

<p align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![YOLO11](https://img.shields.io/badge/YOLO11-Ultralytics-red?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

</p>

---


# 📌 Overview

Football Analytics CV is an end-to-end Computer Vision pipeline that analyzes football broadcast videos and automatically generates player tracking, tactical visualizations, heatmaps, match statistics, and interactive reports.

Unlike simple object detection projects, this system combines multiple Computer Vision modules into one integrated analytics pipeline.

---

# ✨ Features

## Detection

- Player Detection
- Referee Detection
- Ball Detection
- Instance Segmentation

## Tracking

- ByteTrack Multi-Object Tracking
- Kalman Filtering
- Stable Player IDs
- Track Recovery
- Camera Cut Handling

## Pose & Actions

- YOLO11 Pose Estimation
- Skeleton Visualization
- Running Detection
- Ball Interaction Detection

## Tactical Analytics

- Team Classification
- World Coordinate Mapping
- Camera Motion Compensation
- Tactical Minimap
- Heatmaps
- Distance Covered
- Speed Estimation
- Ball Possession
- Match Statistics

## Reports

Automatically generates:

- JSON Report
- CSV Player Statistics
- Heatmap Images
- Analytics Dashboard
- HTML Match Report

---

# 🏗 Architecture

```
football_analytics/

├── config.py
├── detector.py
├── tracker.py
├── pose.py
├── ball_tracker.py
├── homography.py
├── team_classifier.py
├── analytics.py
├── visualization.py
├── minimap.py
├── dashboard.py
├── reports.py
├── utils.py
└── main.py
```

---

# 🧠 Technologies Used

| Category | Technology |
|-----------|------------|
| Language | Python 3.12 |
| Detection | YOLO11 Segmentation |
| Pose Estimation | YOLO11 Pose |
| Tracking | ByteTrack |
| Motion Model | Kalman Filter |
| Computer Vision | OpenCV |
| Deep Learning | PyTorch |
| Numerical Computing | NumPy |
| Scientific Computing | SciPy |
| Team Classification | K-Means Clustering |
| Visualization | Matplotlib |

---

# ⚙️ Pipeline

```
Broadcast Video
        │
        ▼
YOLO11 Segmentation
        │
        ▼
ByteTrack Tracking
        │
        ▼
Kalman Filter
        │
        ▼
Pose Estimation
        │
        ▼
Team Classification
        │
        ▼
Homography
        │
        ▼
Analytics Engine
        │
        ▼
Visualization
        │
        ▼
Reports
```

---

# 📊 Output

The pipeline automatically generates:

- Annotated Match Video
- Tactical Minimap
- Heatmaps
- Distance Covered
- Player Speed
- Ball Possession
- Match Dashboard
- HTML Report
- JSON Export
- CSV Statistics

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/Smithaker10/football-analytics-cv.git

cd football-analytics-cv
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

```bash
python main.py input.mp4
```

---

# 📁 Output Structure

```
outputs/

football_output.mp4

football_dashboard.png

football_heatmap.png

football_report.html

player_stats.csv

pose_analysis.json
```

---

# 📈 Roadmap

### Version 1

- [x] Player Detection
- [x] Ball Detection
- [x] ByteTrack
- [x] Pose Estimation
- [x] Team Classification
- [x] Tactical Minimap
- [x] Match Reports

---

### Version 2

- [ ] Stable Re-Identification (ReID)
- [ ] Pass Detection
- [ ] Shot Detection
- [ ] Formation Recognition
- [ ] Event Detection
- [ ] xG Analytics
- [ ] xT Analytics
- [ ] Pressing Metrics
- [ ] Player Statistics Dashboard
- [ ] Real-Time Webcam Support



# 💡 Why This Project?

Football Analytics CV was built to explore how multiple Computer Vision techniques can work together to transform ordinary football broadcasts into meaningful tactical insights.

The project focuses not only on object detection, but also on tracking, localization, pose estimation, visualization, and analytics in a single unified pipeline.

---

# 🤝 Contributing

Contributions, ideas, and feature requests are welcome.

If you'd like to improve tracking, analytics, or visualization, feel free to open an issue or submit a pull request.

---

# ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub.

It helps others discover the project and motivates future development.

---

<div align="center">

Made with ❤️ by **Smit Thaker**

Building AI-powered sports analytics one frame at a time.

</div>
