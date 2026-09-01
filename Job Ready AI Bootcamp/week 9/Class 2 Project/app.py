"""
Privacy Shield - Production Grade Streamlit Application
=======================================================

Real-time face anonymization using OpenCV face detection
and Streamlit for the web interface.

Usage:
    streamlit run app.py
"""
import streamlit as st
import cv2
import numpy as np
import time

from core.config import CONFIG
from core.camera import ThreadedCamera
from core.processor import FrameProcessor
from core.privacy_engine import PrivacyMode, PrivacySettings


# Page configuration
st.set_page_config(
    page_title=CONFIG.STREAMLIT_PAGE_TITLE,
    page_icon="🛡️",
    layout=CONFIG.STREAMLIT_LAYOUT,
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        color: #888;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .privacy-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #333;
        margin-bottom: 0.5rem;
    }
    .privacy-active {
        border-left: 4px solid #00d26a;
    }
    .privacy-inactive {
        border-left: 4px solid #ff4757;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        "camera": None,
        "processor": None,
        "is_running": False,
        "privacy_mode": PrivacyMode.BLUR,
        "blur_strength": 51,
        "pixelate_size": 15,
        "show_boxes": False,
        "show_stats": True,
        "eyes_only": False,
        "feather_edges": True,
        "frame_count": 0,
        "start_time": None,
        "camera_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_settings() -> PrivacySettings:
    """Build PrivacySettings from session state."""
    return PrivacySettings(
        mode=st.session_state.privacy_mode,
        blur_strength=st.session_state.blur_strength,
        pixelate_size=st.session_state.pixelate_size,
        show_detection_boxes=st.session_state.show_boxes,
        show_stats=st.session_state.show_stats,
        protect_eyes_only=st.session_state.eyes_only,
        feather_edges=st.session_state.feather_edges,
    )


def start_camera():
    """Initialize camera and processor."""
    camera = ThreadedCamera(
        camera_index=CONFIG.CAMERA_INDEX,
        width=CONFIG.FRAME_WIDTH,
        height=CONFIG.FRAME_HEIGHT,
        max_queue_size=CONFIG.MAX_QUEUE_SIZE,
    )

    if not camera.start():
        st.session_state.camera_error = "Failed to open camera. Check permissions."
        return False

    processor = FrameProcessor()
    processor.privacy_settings = build_settings()

    st.session_state.camera = camera
    st.session_state.processor = processor
    st.session_state.is_running = True
    st.session_state.start_time = time.time()
    st.session_state.camera_error = None

    return True


def stop_camera():
    """Release camera resources."""
    if st.session_state.camera:
        st.session_state.camera.stop()
        st.session_state.camera = None
    st.session_state.processor = None
    st.session_state.is_running = False


def main():
    """Main application entry point."""
    init_session_state()

    # Header
    st.markdown('<div class="main-header">🛡️ Privacy Shield</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time face anonymization for video streams</div>', unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")

        # Start/Stop
        if not st.session_state.is_running:
            if st.button("▶️ Start Camera", type="primary", use_container_width=True):
                with st.spinner("Initializing camera..."):
                    if start_camera():
                        st.success("Camera active!")
                        time.sleep(0.3)
                        st.rerun()
                    else:
                        st.error(st.session_state.camera_error or "Unknown error")
        else:
            if st.button("⏹️ Stop Camera", type="secondary", use_container_width=True):
                stop_camera()
                st.rerun()

        st.divider()

        # Privacy Mode Selector
        st.subheader("🔒 Privacy Mode")

        mode_cols = st.columns(2)
        modes = [
            ("🌫️ Blur", PrivacyMode.BLUR),
            ("🟦 Pixelate", PrivacyMode.PIXELATE),
            ("⬛ Solid Mask", PrivacyMode.SOLID_MASK),
            ("📺 Black Bar", PrivacyMode.BLACK_BAR),
            ("🛡️ Emoji", PrivacyMode.EMOJI_MASK),
        ]

        for idx, (label, mode) in enumerate(modes):
            with mode_cols[idx % 2]:
                is_selected = st.session_state.privacy_mode == mode
                btn_type = "primary" if is_selected else "secondary"
                if st.button(label, key=f"mode_{mode.value}", type=btn_type, use_container_width=True):
                    st.session_state.privacy_mode = mode
                    if st.session_state.processor:
                        st.session_state.processor.privacy_settings = build_settings()
                    st.rerun()

        st.divider()

        # Mode-specific settings
        current_mode = st.session_state.privacy_mode

        if current_mode == PrivacyMode.BLUR:
            blur_val = st.slider(
                "Blur Strength",
                min_value=15, max_value=101, value=st.session_state.blur_strength, step=2,
                help="Higher = more blur (must be odd)"
            )
            if blur_val != st.session_state.blur_strength:
                st.session_state.blur_strength = blur_val
                if st.session_state.processor:
                    st.session_state.processor.privacy_settings = build_settings()

        elif current_mode == PrivacyMode.PIXELATE:
            pix_val = st.slider(
                "Pixel Block Size",
                min_value=5, max_value=40, value=st.session_state.pixelate_size,
                help="Larger blocks = more pixelation"
            )
            if pix_val != st.session_state.pixelate_size:
                st.session_state.pixelate_size = pix_val
                if st.session_state.processor:
                    st.session_state.processor.privacy_settings = build_settings()

        st.divider()

        # Advanced options
        st.subheader("🔧 Advanced")

        eyes_only = st.toggle(
            "👁️ Eyes Only",
            value=st.session_state.eyes_only,
            help="Only blur the eye region of detected faces"
        )
        if eyes_only != st.session_state.eyes_only:
            st.session_state.eyes_only = eyes_only
            if st.session_state.processor:
                st.session_state.processor.privacy_settings = build_settings()

        feather = st.toggle(
            "✨ Feather Edges",
            value=st.session_state.feather_edges,
            help="Smooth the transition between blurred and clear regions"
        )
        if feather != st.session_state.feather_edges:
            st.session_state.feather_edges = feather
            if st.session_state.processor:
                st.session_state.processor.privacy_settings = build_settings()

        show_boxes = st.toggle(
            "📦 Show Detection Boxes",
            value=st.session_state.show_boxes,
            help="Visualize face detection boundaries"
        )
        if show_boxes != st.session_state.show_boxes:
            st.session_state.show_boxes = show_boxes
            if st.session_state.processor:
                st.session_state.processor.privacy_settings = build_settings()

        show_stats = st.toggle(
            "📊 Show Stats HUD",
            value=st.session_state.show_stats,
            help="Display FPS and face count overlay"
        )
        if show_stats != st.session_state.show_stats:
            st.session_state.show_stats = show_stats
            if st.session_state.processor:
                st.session_state.processor.privacy_settings = build_settings()

        # Stats
        if st.session_state.is_running and st.session_state.camera:
            st.divider()
            st.subheader("📈 Camera Stats")
            stats = st.session_state.camera.get_stats()
            elapsed = time.time() - st.session_state.start_time if st.session_state.start_time else 0

            st.markdown(f"""
            <div class="privacy-card privacy-active">
                <b>Camera FPS:</b> {stats.fps:.1f}<br>
                <b>Frames:</b> {stats.frame_count:,}<br>
                <b>Dropped:</b> {stats.dropped_frames}<br>
                <b>Uptime:</b> {elapsed:.1f}s
            </div>
            """, unsafe_allow_html=True)

    # Main content
    main_col = st.container()

    with main_col:
        frame_placeholder = st.empty()

        status_cols = st.columns([1, 1, 2])
        with status_cols[0]:
            if st.session_state.is_running:
                st.success("🟢 LIVE — Privacy Active")
            else:
                st.info("⏸️ Camera Off")
        with status_cols[1]:
            if st.session_state.is_running:
                mode_label = st.session_state.privacy_mode.value.upper()
                st.markdown(f"**Mode:** `{mode_label}`")

    # Processing loop
    if st.session_state.is_running and st.session_state.camera and st.session_state.processor:
        camera = st.session_state.camera
        processor = st.session_state.processor

        max_frames = 60

        for _ in range(max_frames):
            frame = camera.read()

            if frame is None:
                time.sleep(0.01)
                continue

            result = processor.process(frame)
            st.session_state.frame_count += 1

            rgb_frame = cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(
                rgb_frame,
                channels="RGB",
                width="stretch",
            )

        time.sleep(0.01)
        st.rerun()

    else:
        # Placeholder when off
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Camera Off", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 2)
        cv2.putText(placeholder, "Click Start Camera to begin", (140, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 1)
        frame_placeholder.image(placeholder, channels="RGB", width="stretch")


if __name__ == "__main__":
    main()
