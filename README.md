# Real-Time Student Entry and Exit Tracking via Computer Vision

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange.svg)](https://opencv.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v8-red.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This project implements a real-time computer vision system to automatically detect, track, and count students as they enter and exit university lecture halls.

Leveraging YOLO object detection, trajectory tracking, and zone-based logic, the system determines movement direction (IN/OUT) using virtual lines or polygons. It supports both CSV logging and optional API integration for live data reporting.

---

## Key Features

- YOLOv8-based object detection  
- Real-time direction-aware tracking (IN / OUT)  
- Smart zone crossing detection using Shapely  
- CSV or database event logging  
- Optional API reporting for integration with university systems  
- Per-class and total counts with on-screen overlays  
- Supports any video input including CCTV and drones

---

## Sample Use Cases

- Smart attendance automation  
- Lecture hall entry/exit logging  
- Student flow analytics  
- Integration with college or university management platforms  

---

## Tech Stack

| Component     | Description                              |
|---------------|------------------------------------------|
| Python        | Main programming language                |
| OpenCV        | Image/video processing                   |
| YOLOv8        | Real-time object detection               |
| Shapely       | Zone/geometry logic for crossing detection |
| NumPy         | Numerical operations                     |
| CSV / API     | Logging and external system integration  |

---

## Project Structure

```

Real_Time_Student_Entry_and_Exit_Tracking_via_Computer_Viion/
│
├── data/
│   ├── inputs/
│   │   └── people_in_marathon.mp4
│   └── outputs/
│       └── 03-07-2025_record.csv
│
├── models/
│   ├── VisDrone_YOLOv8s.pt
│   └── VisDrone_YOLOv8s_x2.pt
│
├── notebooks/
│   ├── training-yolo-on-visdrone-dataset.ipynb
│   └── runs.zip
│
├── src/
│   ├── api/
│   │   └── hall_status_api.py
│   │
│   ├── counting/
│   │   ├── object_counter.py
│   │   └── __init__.py
│   │
│   └── utils/
│       ├── csv_data_recorder.py
│       ├── draw.py
│       └── __init__.py
│
├── main.py
├── main.ipynb
└── requirements.txt

```

---

## Installation

### Clone the repository
```bash
git clone https://github.com/Raafat-Nagy/Real_Time_Student_Entry_and_Exit_Tracking_via_Computer_Vision.git
cd Real_Time_Student_Entry_and_Exit_Tracking_via_Computer_Vision
````

### Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows
```

### Install requirements

```bash
pip install -r requirements.txt
```

---

## Model Training (Optional)

You can train your own custom YOLO model using the VisDrone Dataset provided in the `/notebooks` folder.

See: `notebooks/training-yolo-on-visdrone-dataset.ipynb`

---

## Running the System

Edit `main.py` to change the input video and region if needed:

```bash
python main.py
```

Or run the Jupyter notebook:

```bash
jupyter notebook main.ipynb
```

---

## Output Example

* Real-time video with bounding boxes
* Visual direction count on screen
* CSV log file stored at: `data/outputs/DATE_record.csv`

---

## API Integration (Optional)

If `send_api_events=True`, entry/exit events are sent to a university backend like:

```
GET https://nextgenedu-database.azurewebsites.net/api/hall/enter/{hall_id}
GET https://nextgenedu-database.azurewebsites.net/api/hall/exit/{hall_id}
```

You can control this in `main.py`.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

* [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
* [OpenCV](https://opencv.org/)
* [NumPy](https://numpy.org/)
* [Shapely Geometry Library](https://shapely.readthedocs.io/)
* [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)

---
