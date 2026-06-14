"""
Emoji Webcam App - Production Grade Streamlit Application
============================================================

A real-time emoji overlay application using OpenCV face detection
and Streamlit for the web interface.

Usage:
    streamlit run app.py
"""
import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image
import base64
import io

from core.config import CONFIG
from core.camera import ThreadedCamera
from core.processor import FrameProcessor, ProcessResult


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
    .emoji-btn {
        font-size: 2rem;
        padding: 0.5rem;
        border-radius: 12px;
        border: 2px solid transparent;
        background: #f0f2f6;
        cursor: pointer;
        transition: all 0.2s;
    }
    .emoji-btn:hover {
        background: #e0e2e6;
        transform: scale(1.1);
    }
    .emoji-btn.selected {
        border-color: #FF6B6B;
        background: #fff0f0;
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
        "selected_emoji": "😀",
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
    processor.current_emoji = st.session_state.selected_emoji
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


def render_emoji_grid(available_emojis: list, cols: int = 6):
    """Render emoji selection grid."""
    for i in range(0, len(available_emojis), cols):
        row_emojis = available_emojis[i:i + cols]
        cols_ui = st.columns(cols)

        for idx, emoji_char in enumerate(row_emojis):
            with cols_ui[idx]:
                is_selected = emoji_char == st.session_state.selected_emoji
                btn_class = "emoji-btn selected" if is_selected else "emoji-btn"

                if st.button(
                    emoji_char,
                    key=f"emoji_{emoji_char}",
                    use_container_width=True,
                ):
                    st.session_state.selected_emoji = emoji_char
                    if st.session_state.processor:
                        st.session_state.processor.current_emoji = emoji_char
                    st.rerun()


def main():
    """Main application entry point."""
    init_session_state()

    # Header
    st.markdown('<div class="main-header">📷 Emoji Cam</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time face detection with emoji overlays</div>', unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")

        # Start/Stop
        if not st.session_state.is_running:
            if st.button("▶️ Start Camera", type="primary", use_container_width=True):
                with st.spinner("Initializing camera..."):
                    if start_camera():
                        st.success("Camera started!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(st.session_state.camera_error or "Unknown error")
        else:
            if st.button("⏹️ Stop Camera", type="secondary", use_container_width=True):
                stop_camera()
                st.rerun()

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
        # Camera feed placeholder
        frame_placeholder = st.empty()

        # Status indicator
        status_col, _ = st.columns([1, 3])
        with status_col:
            if st.session_state.is_running:
                st.success("🟢 Live")
            else:
                st.info("⏸️ Camera Off")

    with side_col:
        st.subheader("🎭 Emoji Selector")

        if st.session_state.processor:
            available = st.session_state.processor.get_available_emojis()
        else:
            # Temporary processor just to get emoji list
            temp = FrameProcessor()
            available = temp.get_available_emojis()

        render_emoji_grid(available)

        st.divider()
        st.caption(f"Selected: **{st.session_state.selected_emoji}**")

    # Main processing loop
    if st.session_state.is_running and st.session_state.camera and st.session_state.processor:
        camera = st.session_state.camera
        processor = st.session_state.processor

        # Run for a limited time per rerun to prevent browser timeout
        # Streamlit will auto-rerun, creating a continuous loop effect
        max_frames_per_run = 60  # ~2 seconds at 30fps

        for _ in range(max_frames_per_run):
            frame = camera.read()

            if frame is None:
                time.sleep(0.01)
                continue

            # Process frame
            result = processor.process(frame)
            st.session_state.frame_count += 1

            # Convert BGR to RGB for Streamlit
            rgb_frame = cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB)

            # Display
            frame_placeholder.image(
                rgb_frame,
                channels="RGB",
                use_container_width=True,
            )

        # Auto-rerun to continue stream
        time.sleep(0.01)
        st.rerun()
    else:
        # Show placeholder when camera is off
        placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add "Camera Off" text
        cv2.putText(
            placeholder_img,
            "Camera Off",
            (200, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (100, 100, 100),
            2,
        )
        frame_placeholder.image(
            placeholder_img,
            channels="RGB",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
