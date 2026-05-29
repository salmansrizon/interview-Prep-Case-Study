"""
Shared sidebar navigation and project info.
"""

import streamlit as st
from src.config import get_config


def render_sidebar() -> str:
    """Render the navigation sidebar. Returns selected page name."""
    config = get_config()

    with st.sidebar:
        st.title("🔧 Equipment Predictor")
        st.caption(f"v{config.project.version}")

        st.markdown("---")

        page = st.radio(
            "Navigation",
            options=[
                "🏠 Home",
                "📊 Data Explorer",
                "🔮 Predict",
                "📈 Analytics",
                "🧠 Training",
            ],
            index=0,
        )

        st.markdown("---")

        # Project info
        with st.expander("ℹ️ Project Info"):
            st.markdown(f"""
            **Model:** Deep Neural Network  
            **Hidden Layers:** {config.model.hidden_units}  
            **Dropout:** {config.model.dropout_rate}  
            **Learning Rate:** {config.model.learning_rate}  
            **Batch Size:** {config.model.batch_size}  
            **Loss:** {config.training.loss_function.upper()}
            """)

        # Quick stats
        with st.expander("📁 Data Paths"):
            st.code(f"""
Raw:      {config.paths.data_raw}
Processed: {config.paths.data_processed}
Models:   {config.paths.models_dir}
Logs:     {config.paths.logs_dir}
            """)

        st.markdown("---")
        st.caption("Built with TensorFlow + Streamlit")

    return page
