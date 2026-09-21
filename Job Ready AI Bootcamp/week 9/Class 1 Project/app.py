"""
Face Effects Cam - Streamlit Application
============================================================

A real-time face effects application using OpenCV face detection
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


# Page configuration
st.set_page_config(
    page_title=CONFIG.STREAMLIT_PAGE_TITLE,
    page_icon="📷",
    layout=CONFIG.STREAMLIT_LAYOUT,
    initial_sidebar_state="expanded",
)

# Custom CSS for polished UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .stats-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #4ECDC4;
    }
    .camera-feed {
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        "camera": None,
        "processor": None,
        "is_running": False,
        "camera_enabled": False,
        "selected_effect": FrameProcessor.EFFECTS[0],
        "show_debug": False,
        "frame_count": 0,
        "start_time": None,
        "camera_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def start_camera():
    """Initialize camera and processor."""
    camera = ThreadedCamera(
        camera_index=CONFIG.CAMERA_INDEX,
        width=CONFIG.FRAME_WIDTH,
        height=CONFIG.FRAME_HEIGHT,
        max_queue_size=CONFIG.MAX_QUEUE_SIZE,
    )

    if not camera.start():
        st.session_state.camera_error = "Failed to open camera. Check permissions and connections."
        return False

    processor = FrameProcessor()
    processor.current_effect = st.session_state.selected_effect
    processor.show_debug = st.session_state.show_debug

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


def toggle_camera():
    """Start or stop the camera when the sidebar toggle changes."""
    if st.session_state.camera_enabled:
        if not start_camera():
            st.session_state.camera_enabled = False
    else:
        stop_camera()


@st.fragment(run_every=0.1)
def render_camera_feed():
    """Render one camera frame without blocking the rest of the app."""
    if st.session_state.is_running and st.session_state.camera and st.session_state.processor:
        frame = st.session_state.camera.read()
        if frame is None:
            st.caption("Waiting for camera frame...")
            return

        result = st.session_state.processor.process(frame)
        st.session_state.frame_count += 1
        st.image(cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
        return

    placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        placeholder_img,
        "Camera Off",
        (200, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (100, 100, 100),
        2,
    )
    st.image(placeholder_img, channels="RGB", width="stretch")


def main():
    """Main application entry point."""
    init_session_state()

    # Header
    st.markdown('<div class="main-header">📷 Face Effects Cam</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time face detection with live effects</div>', unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")

        st.toggle("📷 Camera", key="camera_enabled", on_change=toggle_camera)
        if st.session_state.camera_error:
            st.error(st.session_state.camera_error)

        st.divider()

        # Debug mode
        debug_toggle = st.toggle(
            "🐛 Debug Mode",
            value=st.session_state.show_debug,
            help="Show face detection boxes and performance stats",
        )
        if debug_toggle != st.session_state.show_debug:
            st.session_state.show_debug = debug_toggle
            if st.session_state.processor:
                st.session_state.processor.show_debug = debug_toggle

        # Camera settings
        st.subheader("📐 Camera Settings")

        resolution_options = {
            "720p (1280x720)": (1280, 720),
            "1080p (1920x1080)": (1920, 1080),
            "480p (640x480)": (640, 480),
        }

        selected_res = st.selectbox(
            "Resolution",
            options=list(resolution_options.keys()),
            index=0,
        )

        # Stats display
        if st.session_state.is_running and st.session_state.camera:
            st.divider()
            st.subheader("📊 Stats")

            stats = st.session_state.camera.get_stats()
            elapsed = time.time() - st.session_state.start_time if st.session_state.start_time else 0

            st.markdown(f"""
            <div class="stats-card">
                <b>Camera FPS:</b> {stats.fps:.1f}<br>
                <b>Frames Captured:</b> {stats.frame_count}<br>
                <b>Dropped Frames:</b> {stats.dropped_frames}<br>
                <b>Uptime:</b> {elapsed:.1f}s
            </div>
            """, unsafe_allow_html=True)

    # Main content area
    main_col, side_col = st.columns([3, 1])

    with main_col:
        render_camera_feed()

        # Status indicator
        status_col, _ = st.columns([1, 3])
        with status_col:
            if st.session_state.is_running:
                st.success("🟢 Live")
            else:
                st.info("⏸️ Camera Off")

    with side_col:
        st.subheader("✨ Face Effect")
        selected_effect = st.selectbox(
            "Effect",
            options=FrameProcessor.EFFECTS,
            key="selected_effect",
        )
        if st.session_state.processor:
            st.session_state.processor.current_effect = selected_effect

if __name__ == "__main__":
    main()
