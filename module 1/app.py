import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import traceback

# --- IMPORTANT: Redefine the FeatureEngineer class for joblib to work ---
# IT MUST BE IDENTICAL TO THE ONE USED DURING TRAINING.
class FeatureEngineer:
    def __init__(self):
        self.fitted_params = {}
        self.is_fitted = False

    def fit(self, sample_df):
        pass

    def transform(self, df):
        """Apply feature engineering transformations."""
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform. Check if the saved object is valid.")
        df = df.copy()

        # --- Feature 1: Amount Spent Binning ---
        bins = [0] + list(self.fitted_params['amount_spent_quantiles']) + [float('inf')]
        labels = ['Low_Spender', 'Mid_Spender', 'Upper-Mid_Spender', 'High_Spender']
        df['amount_spent_segment'] = pd.cut(df['Amount_spent'], bins=bins, labels=labels, include_lowest=True).astype('category')

        # --- Feature 2: Spending per Age Ratio ---
        df['spending_per_age'] = df['Amount_spent'] / (df['Age'] + 1e-6)
        max_ratio = self.fitted_params['amount_spent_max'] / self.fitted_params['age_min'] if self.fitted_params['age_min'] > 0 else float('inf')
        df['spending_per_age'] = df['spending_per_age'].clip(upper=max_ratio)

        # --- Feature 3: Target Encoding for Segment ---
        df['segment_target_encoded'] = df['Segment'].map(self.fitted_params['segment_means']).astype(float)
        df['segment_target_encoded'] = df['segment_target_encoded'].fillna(0.0)

        # --- Feature 4: Time-based Features ---
        df['days_since_start'] = (df['Transaction_date'] - self.fitted_params['min_date']).dt.days
        df['transaction_year'] = df['Transaction_date'].dt.year
        df['transaction_month'] = df['Transaction_date'].dt.month
        df['transaction_dayofweek'] = df['Transaction_date'].dt.dayofweek
        df['is_weekend'] = (df['transaction_dayofweek'] >= 5).astype(int)

        # Cyclical encoding for month
        df['month_sin'] = np.sin(2 * np.pi * df['transaction_month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['transaction_month'] / 12)

        return df
# --- END CLASS DEFINITION ---

@st.cache_resource
def load_assets():
    """Load the trained model, fitted feature engineer, and fitted imputer."""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.normpath(os.path.join(app_dir, 'outputs', 'final_best_model.joblib'))
    engineer_path = os.path.normpath(os.path.join(app_dir, 'outputs', 'feature_engineer.joblib'))
    imputer_path = os.path.normpath(os.path.join(app_dir, 'outputs', 'imputer.joblib'))

    missing_files = []
    if not os.path.exists(model_path):
        missing_files.append(model_path)
    if not os.path.exists(engineer_path):
        missing_files.append(engineer_path)
    if not os.path.exists(imputer_path):
        missing_files.append(imputer_path)

    if missing_files:
        st.error("Required model/engineer/imputer files not found:")
        for f in missing_files:
            st.error(f"- {f}")
        st.error("Please run the previous notebook cells to generate them.")
        st.stop()

    try:
        model = joblib.load(model_path)
        engineer = joblib.load(engineer_path)
        imputer = joblib.load(imputer_path)
        return model, engineer, imputer
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        st.stop()

st.title('Online Store Customer Spending Predictor')
st.write('Enter customer details to predict their spending category (Low, Medium, High).')

model, engineer, imputer = load_assets()

if model is not None and engineer is not None and imputer is not None:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input('Age', min_value=18, max_value=100, value=30)
        gender = st.selectbox('Gender', ['Male', 'Female'])
        marital = st.selectbox('Marital Status', ['Single', 'Married'])
    with col2:
        state = st.selectbox('State Names', options=[
            'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
            'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD',
            'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH',
            'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY'
        ])
        segment = st.selectbox('Segment', ['Basic', 'Silver', 'Gold', 'Platinum'])
        employees_status = st.selectbox('Employee Status', ['Employees', 'self-employed'])
        payment_method = st.selectbox('Payment Method', ['Card', 'Cash', 'PayPal'])
        referral = st.checkbox('Referred by Friend?', value=False)
        amount = st.number_input('Amount Spent (Current Transaction)', value=1000.0)
        transaction_date = st.date_input('Transaction Date', value=pd.Timestamp.now().date())

    if st.button('Predict Spending Category'):

        # KEY FIX: Include Transaction_ID as a dummy value (0).
        # The imputer AND model were both fitted with this column present,
        # so it must flow through the entire pipeline unchanged.
        input_data = pd.DataFrame({
            'Transaction_ID': [0],  # Dummy — required by pipeline, not meaningful for prediction
            'Transaction_date': [pd.to_datetime(transaction_date)],
            'Gender': [gender],
            'Age': [float(age)],
            'Marital_status': [marital],
            'State_names': [state],
            'Segment': [segment],
            'Employees_status': [employees_status],
            'Payment_method': [payment_method],
            'Referral': [1.0 if referral else 0.0],
            'Amount_spent': [float(amount)]
        })

        try:
            # Step 1: Feature engineering
            transformed = engineer.transform(input_data)

            # Step 2: Select numeric features, drop target + raw date only
            # Do NOT drop Transaction_ID — both imputer and model expect it
            features_df = transformed.select_dtypes(include=[np.number]).drop(
                ['Amount_spent', 'Transaction_date'],
                axis=1,
                errors='ignore'
            )

            # Step 3: Align to exact column order imputer was fitted on
            feature_names_for_training = list(imputer.feature_names_in_)
            features_df_aligned = features_df.reindex(
                columns=feature_names_for_training, fill_value=np.nan
            )

            # Step 4: Impute missing values
            features_imputed_array = imputer.transform(features_df_aligned)
            features_for_prediction = pd.DataFrame(
                features_imputed_array,
                columns=feature_names_for_training,
                index=features_df_aligned.index
            )

            # Step 5: Predict
            pred_array = model.predict(features_for_prediction)
            prediction = pred_array[0]

            if hasattr(prediction, 'item'):
                prediction = prediction.item()
            prediction_str = str(prediction).strip()

            prediction_proba = model.predict_proba(features_for_prediction)[0]
            class_names = model.classes_
            probas_dict = {str(cls): float(prob) for cls, prob in zip(class_names, prediction_proba)}

            st.success(f"Predicted Spending Category: **{prediction_str}**")
            st.write("**Prediction Probabilities:**")
            st.json(probas_dict)

        except Exception as e:
            st.error(f"❌ Prediction failed: {type(e).__name__}: {e}")
            st.code(traceback.format_exc())

else:
    st.warning("Waiting for model, engineer, and imputer assets to load...")