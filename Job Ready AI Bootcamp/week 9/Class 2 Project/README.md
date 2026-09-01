# 🛡️ Privacy Shield

## সহজ ভাষায় Project Overview

**🛡️ Privacy Shield** project-এ lecture-এর theory-কে working software বা executable notebook-এ convert করা হয়েছে। লক্ষ্য শুধু final output দেখা নয়; input থেকে preprocessing, core logic/model, evaluation এবং output—পুরো pipeline বোঝা।

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

A production-grade real-time face anonymization application using Python, OpenCV, and Streamlit. Automatically detects and obscures human faces in live video streams to protect privacy.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.8+-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🌫️ **Gaussian Blur** | Adjustable strength with feathered edges for smooth transitions |
| 🟦 **Pixelate** | Classic pixel block effect with configurable block size |
| ⬛ **Solid Mask** | Semi-transparent color overlay with opacity control |
| 📺 **Black Bar** | TV-style censor bar with "CENSORED" text |
| 🛡️ **Emoji Mask** | Shield emoji overlay with blurred background |
| 👁️ **Eyes-Only Mode** | Protect only the eye region while keeping rest of face visible |
| ✨ **Feathered Edges** | Smooth blending between protected and clear regions |
| 📦 **Debug Boxes** | Optional face detection boundary visualization |
| 📊 **Live HUD** | Real-time FPS, face count, and processing stats |

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
privacy_shield/
├── app.py                 # Streamlit UI and main loop
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── core/
    ├── __init__.py
    ├── config.py         # App configuration (immutable dataclass)
    ├── camera.py         # Threaded camera capture with queue management
    ├── face_detector.py  # Haar Cascade face detection
    ├── privacy_engine.py # Anonymization modes (blur, pixelate, mask, etc.)
    └── processor.py      # Frame processing pipeline
```

## 🔒 Privacy Modes

### Blur
Applies Gaussian blur to detected faces. Adjustable strength (15-101). Feathered edges create smooth transitions.

### Pixelate
Downsamples face region then upsamples with nearest-neighbor interpolation for a retro pixelated look.

### Solid Mask
Applies a semi-transparent colored overlay. Configurable opacity and feathered edges.

### Black Bar
Classic TV censor style — black rectangle with "CENSORED" text for faces wider than 100px.

### Emoji Mask
Blurs the face then overlays a shield emoji (🛡️) centered on the face region.

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
    │  Face   │  │  Privacy │  │   HUD    │
    │Detector │  │  Engine  │  │  Overlay │
    └─────────┘  └──────────┘  └──────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Streamlit     │
              │   Display       │
              └─────────────────┘
```

## 🛠️ Development

### Adding New Privacy Modes

1. Add mode to `PrivacyMode` enum in `core/privacy_engine.py`
2. Implement `_apply_your_mode()` method in `PrivacyEngine`
3. Add UI button in `app.py` sidebar

## 📜 License

MIT License
