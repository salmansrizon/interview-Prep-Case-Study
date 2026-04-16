import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import traceback # Import traceback for detailed errors

# --- IMPORTANT: Redefine the FeatureEngineer class for joblib to work ---
# This is necessary because Streamlit runs in a different context and needs to know how to recreate the object.
class FeatureEngineer:
    def __init__(self):
        self.fitted_params = {}
        self.is_fitted = False

    def fit(self, sample_df):
        # This method is typically called during training, not in the deployed app.
        # The app uses a *pre-fitted* engineer loaded from a .joblib file.
        # Therefore, fit() might not be strictly needed in the app, but defining it keeps the class structure consistent.
        # If fit() is called in the app, it will overwrite the pre-fitted parameters!
        # Usually, the app only calls transform().
        pass

    def transform(self, df):
        """Apply feature engineering transformations."""
        if not self.is_fitted:
            # In a deployed app, this error indicates the saved engineer object
            # wasn't loaded correctly or is corrupted.
            raise ValueError("FeatureEngineer must be fitted before transform. Check if the saved object is valid.")
        df = df.copy()

        # --- Feature 1: Amount Spent Binning ---
        # Quantiles from fit: [0.25, 0.5, 0.75] -> 3 values
        # Bins: [0, q1, q2, q3, inf] -> 4 intervals
        # Labels: Must be 4 to match 4 intervals
        bins = [0] + list(self.fitted_params['amount_spent_quantiles']) + [float('inf')]
        labels = ['Low_Spender', 'Mid_Spender', 'Upper-Mid_Spender', 'High_Spender'] # 4 labels for 4 bins
        df['amount_spent_segment'] = pd.cut(df['Amount_spent'], bins=bins, labels=labels, include_lowest=True).astype('category')

        # --- Feature 2: Spending per Age Ratio (with clipping to prevent extreme outliers) ---
        df['spending_per_age'] = df['Amount_spent'] / (df['Age'] + 1e-6) # Add small value to prevent division by zero
        # Clip to prevent extreme ratios based on fitted parameters
        max_ratio = self.fitted_params['amount_spent_max'] / self.fitted_params['age_min'] if self.fitted_params['age_min'] > 0 else float('inf')
        df['spending_per_age'] = df['spending_per_age'].clip(upper=max_ratio)

        # --- Feature 3: Target Encoding for Segment (with fallback) ---
        df['segment_target_encoded'] = df['Segment'].map(self.fitted_params['segment_means']).astype(float)
        df['segment_target_encoded'] = df['segment_target_encoded'].fillna(0.0) # Using 0 as fallback for unseen segments during fit

        # --- Feature 4: Time-based Features ---
        df['days_since_start'] = (df['Transaction_date'] - self.fitted_params['min_date']).dt.days
        df['transaction_year'] = df['Transaction_date'].dt.year
        df['transaction_month'] = df['Transaction_date'].dt.month
        df['transaction_dayofweek'] = df['Transaction_date'].dt.dayofweek
        df['is_weekend'] = (df['transaction_dayofweek'] >= 5).astype(int)

        # Cyclical encoding for month (captures seasonal patterns)
        df['month_sin'] = np.sin(2 * np.pi * df['transaction_month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['transaction_month'] / 12)

        return df
# --- END CLASS DEFINITION ---

@st.cache_resource
def load_assets():
    """Load the trained model, fitted feature engineer, and fitted imputer."""
    # Construct paths relative to where the app script will be run
    app_dir = os.path.dirname(os.path.abspath(__file__)) # Directory of app.py
    model_path = os.path.normpath(os.path.join(app_dir, 'outputs', 'final_best_model.joblib')) # Adjust path as needed
    engineer_path = os.path.normpath(os.path.join(app_dir, 'outputs', 'feature_engineer.joblib')) # Adjust path as needed
    imputer_path = os.path.normpath(os.path.join(app_dir, 'outputs', 'imputer.joblib')) # Path to the saved imputer

    # Check if all required files exist
    missing_files = []
    if not os.path.exists(model_path):
        missing_files.append(model_path)
    if not os.path.exists(engineer_path):
        missing_files.append(engineer_path)
    if not os.path.exists(imputer_path):
        missing_files.append(imputer_path)

    if missing_files:
        st.error(f"Required model/engineer/imputer files not found:")
        for f in missing_files:
            st.error(f"- {f}")
        st.error("Please run the previous notebook cells to generate them.")
        st.stop() # Stop execution if critical files are missing

    try:
        model = joblib.load(model_path)
        engineer = joblib.load(engineer_path)
        imputer = joblib.load(imputer_path) # Load the imputer
        return model, engineer, imputer
    except Exception as e:
        st.error(f"Error loading assets: {{e}}")
        st.error(f"Traceback: {{traceback.format_exc()}}")
        st.stop()

st.title('Online Store Customer Spending Predictor')
st.write('Enter customer details to predict their spending category (Low, Medium, High).')

# Load all necessary assets
model, engineer, imputer = load_assets()

if model is not None and engineer is not None and imputer is not None:
    # Create input fields in two columns
    col1, col2 = st.columns(2)
    with col1:
        transaction_id = st.text_input('Transaction ID (optional)', value='TEMP_ID')
        age = st.number_input('Age', min_value=18, max_value=100, value=30)
        gender = st.selectbox('Gender', ['Male', 'Female']) # Assuming these were categories in the original data
        marital = st.selectbox('Marital Status', ['Single', 'Married'])
    with col2:
        state = st.selectbox('State Names', options=['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']) # Provide a list or load dynamically if possible
        segment = st.selectbox('Segment', ['Basic', 'Silver', 'Gold', 'Platinum']) # Assuming these were categories
        employees_status = st.selectbox('Employee Status', ['Employees', 'self-employed']) # Assuming these were categories
        payment_method = st.selectbox('Payment Method', ['Card', 'Cash', 'PayPal']) # Assuming these were categories
        referral = st.checkbox('Referred by Friend?', value=False)
        amount = st.number_input('Amount Spent (Current Transaction)', value=1000.0)
        # Date input for the transaction (converted to datetime in transform)
        transaction_date = st.date_input('Transaction Date', value=pd.Timestamp.now().date())

    if st.button('Predict Spending Category'):
        # Create a DataFrame with the input data
        # CRITICAL: Include 'Transaction_ID' here as it might be part of the model's feature set.
        input_data = pd.DataFrame({
            'Transaction_ID': [transaction_id],
            'Transaction_date': [pd.to_datetime(transaction_date)], # Convert date input to datetime
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
            # --- Apply the same feature engineering pipeline used during training ---
            transformed = engineer.transform(input_data)

            # --- Select features used during training (exclude target, ID, original date) ---
            # This must EXACTLY match the column selection done during training (e.g., in X_sample_imputed.columns)
            # CRITICAL: Do NOT drop 'Transaction_ID' here if the model expects it. Only drop 'Amount_spent' (target) and 'Transaction_date'.
            features_df = transformed.select_dtypes(include=[np.number]).drop([
                'Amount_spent', 'Transaction_date' # Drop the target and original date; keep 'Transaction_ID' if expected
            ], axis=1, errors='ignore')

            # --- Align columns with the order used during training and apply imputation ---
            # Get the column names the imputer was fitted on (these define the required feature set and order)
            feature_names_for_training = imputer.feature_names_in_

            # --- CRITICAL FIX: Ensure features_df has exactly the columns expected by the imputer/model ---
            # Check for missing or extra columns before reindexing
            missing_cols = set(feature_names_for_training) - set(features_df.columns)
            extra_cols = set(features_df.columns) - set(feature_names_for_training)

            if missing_cols:
                 raise ValueError(f"Missing features after transformation: {{missing_cols}}. Ensure FeatureEngineer output matches training features.")
            if extra_cols:
                 # Log a warning but proceed by dropping extra columns if present
                 st.warning(f"Extra features found after transformation (will be dropped): {{extra_cols}}")
                 features_df = features_df.drop(columns=extra_cols)

            # Reindex the features_df to ensure the same column order and presence as during training.
            # Fill_value=np.nan is appropriate here as the imputer will handle NaNs.
            features_df_aligned = features_df.reindex(columns=feature_names_for_training, fill_value=np.nan)

            # --- Apply the LOADED imputer to handle any potential NaNs using training statistics ---
            # This transforms the DataFrame into a numpy array
            features_imputed_array = imputer.transform(features_df_aligned)

            # Convert back to a DataFrame with the correct column names and index for clarity (optional but good practice)
            features_for_prediction = pd.DataFrame(
                features_imputed_array,
                columns=feature_names_for_training,
                index=features_df_aligned.index
            )

            # --- Make prediction using the loaded model ---
            pred_array = model.predict(features_for_prediction) # Returns ndarray
            prediction = pred_array[0] # Get first element

            # Safely convert to string for display
            if hasattr(prediction, 'item'):
                prediction = prediction.item() # Convert np.int64/np.str_ -> Python int/str
            prediction_str = str(prediction).strip()

            prediction_proba = model.predict_proba(features_for_prediction)[0]
            class_names = model.classes_

            # Convert probabilities to native floats for JSON serialization
            probas_dict = {str(cls): float(prob) for cls, prob in zip(class_names, prediction_proba)}

            st.success(f"Predicted Spending Category: **{{prediction_str}}**")
            st.write("**Prediction Probabilities:**")
            st.json(probas_dict)

        except Exception as e:
            st.error(f"Prediction failed: {{type(e).__name__}}: {{e}}")
            st.code(traceback.format_exc())

else:
    st.warning("Waiting for model, engineer, and imputer assets to load...")