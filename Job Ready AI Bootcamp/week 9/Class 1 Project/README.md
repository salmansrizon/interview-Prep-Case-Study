# 📷 Face Effects Cam

## সহজ ভাষায় Project Overview

**📷 Face Effects Cam** project-এ lecture-এর theory-কে working software বা executable notebook-এ convert করা হয়েছে। লক্ষ্য শুধু final output দেখা নয়; input থেকে preprocessing, core logic/model, evaluation এবং output—পুরো pipeline বোঝা।

### কোন Problem Solve করে?

Manual বা disconnected workflow-কে repeatable code pipeline-এ আনে। এর ফলে একই process নতুন data-তে আবার চালানো, result compare করা, error trace করা এবং future feature add করা সহজ হয়।

### কীভাবে কাজ করে?

Input/Data → Validation ও Preprocessing → Core Algorithm/Model → Evaluation → UI, Report বা Saved Output। নিচের detailed section-গুলোতে project-specific command, feature এবং architecture দেওয়া আছে।

### কেন এই Approach ভালো?

- **Repeatable:** একই input দিলে একই workflow follow করে।
- **Testable:** প্রতিটি stage আলাদাভাবে verify করা যায়।
- **Explainable:** কোন step কী কাজ করছে তা code এবং output দিয়ে দেখা যায়।
- **Portfolio-ready:** শুধু notebook result নয়, setup, structure এবং usage-সহ complete project হিসেবে দেখানো যায়।

> **Run করার নিয়ম:** আগে virtual environment তৈরি করে dependency install করুন। তারপর README-এর Quick Start follow করুন, sample input দিয়ে smoke test করুন এবং expected metric/output-এর সাথে result compare করুন।

---

A real-time face effects application using Python, OpenCV, and Streamlit. It detects faces via webcam and applies selectable effects only to each detected face.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.8+-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)

## ✨ Features

- ✨ **Four Face Effects** — Blur, Pixelate, Grayscale, and Negative
- 🧠 **Real-time Face Detection** — Haar Cascade classifier optimized for speed
- 🎨 **Face-only Processing** — The background remains unchanged
- ⚡ **Threaded Camera Capture** — Non-blocking frame acquisition prevents UI lag
- 📊 **Performance HUD** — Live FPS, frame count, and processing stats
- 🐛 **Debug Mode** — Visualize face detection bounding boxes
- 📐 **Adjustable Resolution** — 480p, 720p, 1080p support
- 🔄 **Auto Frame Dropping** — Prevents memory buildup and lag

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## 📁 Project Structure

```
face_effects_app/
├── app.py                 # Streamlit UI and main loop
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── core/
    ├── __init__.py
    ├── config.py         # App configuration (immutable dataclass)
    ├── camera.py         # Threaded camera capture with queue management
    ├── face_detector.py  # Haar Cascade face detection
    └── processor.py      # Face effects and pipeline orchestration
```

## ⚙️ Configuration

Environment variables (all optional):

| Variable | Default | Description |
|----------|---------|-------------|
| `CAMERA_INDEX` | `0` | Webcam device index |
| `FRAME_WIDTH` | `1280` | Capture width |
| `FRAME_HEIGHT` | `720` | Capture height |
| `FPS_TARGET` | `30` | Target frame rate |

## 🔧 Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Webcam    │────▶│   Camera     │────▶│   Frame     │
│  (OpenCV)   │     │   Thread     │     │   Queue     │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                       ┌────────────────────────┘
                       ▼
              ┌─────────────────┐
              │   Frame         │
              │   Processor     │
              └────────┬────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
    ┌─────────┐  ┌──────────┐  ┌──────────┐
    │  Face   │  │  Effect  │  │   HUD    │
    │Detector │  │  Filter  │  │  Overlay │
    └─────────┘  └──────────┘  └──────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Streamlit     │
              │   Display       │
              └─────────────────┘
```

## 🛠️ Effects

Choose **Blur**, **Pixelate**, **Grayscale**, or **Negative** from the live selector. Effects are implemented with OpenCV in `core/processor.py`.

## 📜 License

MIT License — feel free to use in your own projects!
