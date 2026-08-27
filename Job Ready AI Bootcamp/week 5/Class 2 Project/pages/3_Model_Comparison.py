import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from utils.model_utils import MODELS_DIR
from utils.model_utils import load_model_results, load_model, load_scaler
from utils.visualizations import plot_confusion_matrix, plot_roc_curve, plot_model_comparison
import numpy as np

st.set_page_config(page_title="Model Comparison", page_icon="⚖️", layout="wide")

st.title("⚖️ Model Comparison")
st.markdown("Side-by-side evaluation of Logistic Regression, Decision Tree, and Random Forest.")

results = load_model_results()

# ============================================================
# METRICS TABLE
# ============================================================
st.subheader("📊 Performance Metrics")

metrics_df = pd.DataFrame(results).T
metrics_df.index = [idx.replace('_', ' ').title() for idx in metrics_df.index]
metrics_df = metrics_df.round(4)

st.dataframe(metrics_df, use_container_width=True)

# Highlight best
st.markdown("**🏆 Best Performer by Metric:**")
best_metrics = {}
for col in metrics_df.columns:
    best_model = metrics_df[col].idxmax()
    best_val = metrics_df[col].max()
    best_metrics[col] = (best_model, best_val)
    st.markdown(f"• **{col.title()}:** {best_model} ({best_val:.4f})")

# ============================================================
# RADAR CHART
# ============================================================
st.markdown("---")
st.subheader("📈 Radar Chart Comparison")

fig_radar = plot_model_comparison(results)
st.plotly_chart(fig_radar, use_container_width=True)

# ============================================================
# CONFUSION MATRICES
# ============================================================
st.markdown("---")
st.subheader("📉 Confusion Matrices")

cm_lr = np.load(MODELS_DIR / "cm_logistic.npy")
cm_dt = np.load(MODELS_DIR / "cm_decision_tree.npy")
cm_rf = np.load(MODELS_DIR / "cm_random_forest.npy")

col1, col2, col3 = st.columns(3)
with col1:
    fig = plot_confusion_matrix(cm_lr, title="Logistic Regression")
    st.pyplot(fig)
with col2:
    fig = plot_confusion_matrix(cm_dt, title="Decision Tree")
    st.pyplot(fig)
with col3:
    fig = plot_confusion_matrix(cm_rf, title="Random Forest")
    st.pyplot(fig)

st.markdown("""
**How to read:**
- **Top-Left (TN):** Correctly denied loans
- **Top-Right (FP):** Incorrectly approved (Type I error — risky!)
- **Bottom-Left (FN):** Incorrectly denied (Type II error — lost business)
- **Bottom-Right (TP):** Correctly approved loans
""")

# ============================================================
# ROC CURVES
# ============================================================
st.markdown("---")
st.subheader("📈 ROC Curves")

# Real test-set ROC points, saved by train_model.py — no approximations.
curves = np.load(MODELS_DIR / "roc_curves.npz")
fig, ax = plt.subplots(figsize=(10, 8))

for model_name, res in results.items():
    ax.plot(curves[f"{model_name}_fpr"], curves[f"{model_name}_tpr"], lw=2,
            label=f"{model_name.replace('_', ' ').title()} (AUC = {res['roc_auc']:.3f})")

ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
ax.fill_between([0, 1], [0, 1], alpha=0.1, color='gray')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve Comparison', fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
st.pyplot(fig)

st.markdown("""
**AUC Interpretation:**
- **0.5** = Random guessing (no better than coin flip)
- **0.6-0.7** = Poor discrimination
- **0.7-0.8** = Acceptable discrimination
- **0.8-0.9** = Excellent discrimination
- **>0.9** = Outstanding discrimination
""")

# ============================================================
# BUSINESS IMPACT
# ============================================================
st.markdown("---")
st.subheader("💼 Business Impact Analysis")

# Calculate business metrics from confusion matrices
def business_metrics(cm, total_loans=1000, avg_loan=150000, default_rate=0.15, profit_margin=0.03):
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()

    # Scale to business scenario
    scale = total_loans / total
    tn_n, fp_n, fn_n, tp_n = tn*scale, fp*scale, fn*scale, tp*scale

    # Revenue from approved loans
    approved_revenue = (tp_n + fp_n) * avg_loan * profit_margin

    # Loss from defaults (FP)
    default_loss = fp_n * avg_loan * default_rate

    # Opportunity cost (FN — lost good loans)
    opportunity_cost = fn_n * avg_loan * profit_margin

    net_profit = approved_revenue - default_loss - opportunity_cost

    return {
        'net_profit': net_profit,
        'approved_correctly': tp_n,
        'denied_correctly': tn_n,
        'false_approvals': fp_n,
        'false_denials': fn_n
    }

biz_lr = business_metrics(cm_lr)
biz_dt = business_metrics(cm_dt)
biz_rf = business_metrics(cm_rf)

biz_df = pd.DataFrame({
    'Logistic Regression': biz_lr,
    'Decision Tree': biz_dt,
    'Random Forest': biz_rf
}).T

st.dataframe(biz_df, use_container_width=True)

# Whichever model actually earns the most — not whichever one we expected to.
best_model = biz_df['net_profit'].idxmax()

st.markdown(f"""
<div style="background-color: #E8F5E9; padding: 15px; border-radius: 8px;">
    <strong>Scenario:</strong> 1,000 loan applications, avg loan $150K, 3% profit margin, 15% default rate on bad loans.<br>
    <strong>Best Model:</strong> {best_model} with estimated net profit of <strong>${biz_df.loc[best_model, 'net_profit']:,.0f}</strong>
</div>
""", unsafe_allow_html=True)
