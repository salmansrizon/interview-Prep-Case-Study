import streamlit as st
import pandas as pd
import numpy as np
from utils.model_utils import load_model, load_feature_names, predict_price, explain_prediction
from utils.data_loader import load_housing_data, get_feature_ranges
from utils.visualizations import plot_prediction_breakdown

st.set_page_config(page_title="Predict Price", page_icon="🔮", layout="wide")

st.title("🔮 House Price Predictor")
st.markdown("Enter house details below to get an instant price estimate with full explanation.")

st.markdown("""
<div style="background-color: #E8F5E9; padding: 20px; border-radius: 10px; border-left: 5px solid #2E7D32;">
<h3 style="margin-top: 0; color: #2E7D32;">How It Works</h3>
<p>
Our Linear Regression model learned patterns from 5,000 real estate transactions. 
It combines <strong>16 features</strong> to estimate market value. Each feature contributes 
positively or negatively to the final price based on its learned coefficient.
</p>
</div>
""", unsafe_allow_html=True)

# Load data for ranges
df = load_housing_data()
ranges = get_feature_ranges(df)

# ============================================================
# INPUT FORM
# ============================================================
st.markdown("---")
st.subheader("🏠 Enter House Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Location & Demographics**")
    median_income = st.slider("Median Income (block group, $10K)", 
                               float(ranges['median_income']['min']), 
                               float(ranges['median_income']['max']),
                               float(ranges['median_income']['median']), 0.1)
    latitude = st.slider("Latitude", 32.5, 42.0, 36.0, 0.01)
    longitude = st.slider("Longitude", -124.5, -114.0, -119.0, 0.01)
    distance_to_city = st.slider("Distance to City (miles)", 
                                  float(ranges['distance_to_city']['min']),
                                  float(ranges['distance_to_city']['max']),
                                  float(ranges['distance_to_city']['median']), 0.5)
    school_rating = st.slider("School Rating (1-10)", 1, 10, 7, 1)

with col2:
    st.markdown("**House Characteristics**")
    house_age = st.slider("House Age (years)", 
                           int(ranges['house_age']['min']), 
                           int(ranges['house_age']['max']),
                           int(ranges['house_age']['median']), 1)
    avg_rooms = st.slider("Avg Rooms per Household", 
                           float(ranges['avg_rooms']['min']),
                           float(ranges['avg_rooms']['max']),
                           float(ranges['avg_rooms']['median']), 0.1)
    avg_bedrooms = st.slider("Avg Bedrooms per Household", 
                              float(ranges['avg_bedrooms']['min']),
                              float(ranges['avg_bedrooms']['max']),
                              float(ranges['avg_bedrooms']['median']), 0.1)
    lot_size_sqft = st.slider("Lot Size (sq ft)", 
                               float(ranges['lot_size_sqft']['min']),
                               float(ranges['lot_size_sqft']['max']),
                               float(ranges['lot_size_sqft']['median']), 100.0)
    year_built = 2024 - house_age

with col3:
    st.markdown("**Amenities & Community**")
    garage_spaces = st.selectbox("Garage Spaces", [0, 1, 2, 3, 4], index=2)
    has_pool = st.checkbox("Has Swimming Pool")
    has_basement = st.checkbox("Has Basement")
    population = st.slider("Block Population", 
                            int(ranges['population']['min']),
                            int(ranges['population']['max']),
                            int(ranges['population']['median']), 10)
    avg_occupancy = st.slider("Avg Occupancy", 
                               float(ranges['avg_occupancy']['min']),
                               float(ranges['avg_occupancy']['max']),
                               float(ranges['avg_occupancy']['median']), 0.1)
    property_tax_rate = st.slider("Property Tax Rate (%)", 
                                   float(ranges['property_tax_rate']['min']),
                                   float(ranges['property_tax_rate']['max']),
                                   float(ranges['property_tax_rate']['median']), 0.01)

# Build feature dictionary
features = {
    'median_income': median_income,
    'house_age': house_age,
    'avg_rooms': avg_rooms,
    'avg_bedrooms': avg_bedrooms,
    'population': population,
    'avg_occupancy': avg_occupancy,
    'latitude': latitude,
    'longitude': longitude,
    'lot_size_sqft': lot_size_sqft,
    'garage_spaces': garage_spaces,
    'has_pool': int(has_pool),
    'has_basement': int(has_basement),
    'distance_to_city': distance_to_city,
    'school_rating': school_rating,
    'property_tax_rate': property_tax_rate,
    'year_built': year_built
}

# ============================================================
# PREDICTION
# ============================================================
st.markdown("---")

if st.button("🔮 Predict Price", type="primary", use_container_width=True):
    predicted_price = predict_price(features)
    contributions = explain_prediction(features)

    # Display prediction
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 30px; border-radius: 15px; text-align: center; margin: 20px 0;">
        <h2 style="margin: 0; font-size: 3rem; color: white;">💰 ${:,.0f}</h2>
        <p style="margin: 10px 0 0 0; font-size: 1.3rem; opacity: 0.9;">Estimated Market Value</p>
    </div>
    """.format(predicted_price), unsafe_allow_html=True)

    # Price range (±1 RMSE)
    rmse = 22433  # From model evaluation
    st.markdown(f"""
    <div style="text-align: center; color: #666; margin-bottom: 20px;">
        <strong>Confidence Range:</strong> ${predicted_price - rmse:,.0f} — ${predicted_price + rmse:,.0f}
        <br><small>(Based on model RMSE of ${rmse:,.0f})</small>
    </div>
    """, unsafe_allow_html=True)

    # Prediction breakdown
    st.subheader("📊 How Each Feature Contributes")

    # Sort contributions by absolute value
    sorted_contrib = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True))

    contrib_df = pd.DataFrame({
        'Feature': list(sorted_contrib.keys()),
        'Contribution ($)': list(sorted_contrib.values()),
        'Direction': ['⬆️ Increases' if v > 0 else '⬇️ Decreases' for v in sorted_contrib.values()]
    })

    # Format contributions
    contrib_df['Contribution ($)'] = contrib_df['Contribution ($)'].apply(lambda x: f"${x:+,.0f}")
    st.dataframe(contrib_df, use_container_width=True)

    # Top positive and negative drivers
    positive = {k: v for k, v in contributions.items() if v > 0 and k != 'Intercept'}
    negative = {k: v for k, v in contributions.items() if v < 0}

    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown("**🟢 Top Price Boosters:**")
        for feat, val in sorted(positive.items(), key=lambda x: x[1], reverse=True)[:5]:
            st.markdown(f"• `{feat}`: +${val:,.0f}")
    with col_neg:
        st.markdown("**🔴 Top Price Reducers:**")
        for feat, val in sorted(negative.items(), key=lambda x: x[1])[:5]:
            st.markdown(f"• `{feat}`: ${val:,.0f}")

    st.markdown("---")

    # Feature comparison to dataset averages
    st.subheader("📈 How Your House Compares to the Market")

    compare_features = ['median_income', 'avg_rooms', 'lot_size_sqft', 'school_rating', 'house_age']
    compare_data = []
    for feat in compare_features:
        compare_data.append({
            'Feature': feat.replace('_', ' ').title(),
            'Your Value': features[feat],
            'Market Avg': df[feat].mean(),
            'Market Median': df[feat].median(),
            'Percentile': (df[feat] <= features[feat]).mean() * 100
        })

    compare_df = pd.DataFrame(compare_data)
    compare_df['Your Value'] = compare_df['Your Value'].round(2)
    compare_df['Market Avg'] = compare_df['Market Avg'].round(2)
    compare_df['Market Median'] = compare_df['Market Median'].round(2)
    compare_df['Percentile'] = compare_df['Percentile'].round(1).astype(str) + 'th'
    st.dataframe(compare_df, use_container_width=True)

st.markdown("---")
st.caption("💡 This prediction is an estimate based on historical data patterns. Actual market prices may vary due to factors not captured in the model (e.g., recent renovations, neighborhood trends, market conditions).")
