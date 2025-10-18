# ==============================================================================
# CELL 5: STREAMLIT APPLICATION CODE (SAVE AS app.py)
# ==============================================================================

APP_CODE = """
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- Configuration for Loading Assets ---
ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')
MODEL_PATH = os.path.join(ASSETS_DIR, 'xgb_co2_model.pkl')
SCALER_PATH = os.path.join(ASSETS_DIR, 'scaler.pkl')

# Feature list MUST match the order used during training
FEATURES = [
    'GDP', 'Elec_Cons_PC', 'Population', 'Renewable_Share', 'Coal_Share', 
    'CO2_Emissions_kt_Lag1'
]

# --- Custom CSS for Theming (Futura Font & Colorful Style) ---
st.markdown(\"""
<style>
    /* NOTE: Futura requires font hosting or local installation. We use a standard sans-serif 
    fallback, and set colors based on your scheme. */
    body {
        font-family: 'Arial', sans-serif; 
        color: #1A431A; /* Dark Green */
        background-color: #FDF9F3; /* Light Cream */
    }

    h1, h2, h3 {
        color: #1A431A; /* Dark Green for headings */
    }
    
    .stButton>button {
        background-color: #FFBF00; /* Golden Yellow for buttons */
        color: #1A431A; /* Dark Green text on buttons */
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #e6a800; /* Slightly darker yellow on hover */
    }

    /* Streamlit Metric Styling (Peach background) */
    [data-testid="stMetric"] > div {
        background-color: #FFDAB9; 
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        color: #1A431A; 
    }
    [data-testid="stMetricValue"] {
        color: #1A431A; 
        font-size: 2.5em;
        font-weight: bold;
    }
    /* Set positive delta to Lime Green, and let Streamlit handle the "inverse" (negative) color */
    [data-testid="stMetricDelta"] > div {
        color: #9ACD32 !important; 
    }

    /* Feature Importance Bar Chart (Lime Green bars) */
    .st-dl .st-dp {
        background-color: #9ACD32; 
    }
    
    /* Info/Interpretation block styling */
    .stAlert {
        background-color: #FDF9F3; 
        border-left: 5px solid #FFBF00; 
        color: #1A431A;
    }

</style>
\""", unsafe_allow_html=True)
# --- End of Custom CSS ---


# --- Load Model Assets ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError:
        st.error(f"Error: Model files not found. Please ensure 'xgb_co2_model.pkl' and 'scaler.pkl' are in the '{ASSETS_DIR}' folder.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading assets: {e}")
        st.stop()

model, scaler = load_assets()

st.title("🌍 SDG 13: Proactive Carbon Emission Forecaster")
st.subheader("Simulate future CO₂ emissions based on key socio-economic indicators (in kt).")

# -----------------
# User Input (Policy Scenario Generator)
# -----------------
with st.sidebar:
    st.header("Policy Levers")
    st.write("Adjust the projected values for the next year to simulate policy impact.")
    
    # User inputs for the features, using realistic example values
    input_data = {
        'GDP': st.number_input("Projected GDP (US$ Trillions)", value=2.0, min_value=0.1, key='gdp_input') * 1e12, # Convert Trillions to standard unit
        'Elec_Cons_PC': st.number_input("Elec. Consumption (kWh/capita)", value=5000.0, min_value=100.0, key='elec_input'),
        'Population': st.number_input("Projected Population (Millions)", value=300.0, min_value=10.0, key='pop_input') * 1e6, # Convert Millions to standard unit
        'Renewable_Share': st.slider("Renewable Share in Energy (%)", value=15.0, min_value=0.0, max_value=100.0, step=0.1, key='renew_input'),
        'Coal_Share': st.slider("Coal Share in Energy (%)", value=25.0, min_value=0.0, max_value=100.0, step=0.1, key='coal_input'),
        'CO2_Emissions_kt_Lag1': st.number_input("Previous Year's CO₂ Emissions (kt)", value=3000000.0, min_value=0.0, key='lag_input')
    }

# --- Prediction Logic ---
if st.button("Forecast Emissions"):
    # Create DataFrame in the exact feature order
    X_pred = pd.DataFrame([[
        input_data['GDP'],
        input_data['Elec_Cons_PC'],
        input_data['Population'],
        input_data['Renewable_Share'],
        input_data['Coal_Share'],
        input_data['CO2_Emissions_kt_Lag1']
    ]], columns=FEATURES)

    # Apply the trained scaler
    X_pred_scaled = scaler.transform(X_pred)

    # Make prediction
    predicted_co2 = model.predict(X_pred_scaled)[0]
    
    # Calculate the change from the previous year
    delta_value = predicted_co2 - input_data['CO2_Emissions_kt_Lag1']

    # -----------------
    # Results Display
    # -----------------
    st.markdown("---")
    st.metric(
        label="Projected CO₂ Emissions (kt)",
        value=f"{predicted_co2:,.0f}",
        delta=f"{delta_value:,.0f} kt",
        delta_color="inverse" # Use 'inverse' to highlight positive change (increase) as bad/red/orange
    )

    # -----------------
    # Explainability (XAI)
    # -----------------
    st.markdown("## 🔍 Policy Impact Analysis (Feature Importance)")
    st.write("This chart shows which input factors were weighted most heavily for the prediction:")
    
    # Get Feature Importance from XGBoost
    importance_df = pd.DataFrame({
        'Feature': FEATURES,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True) # Ascending for bar_chart visualization

    # Use horizontal bar chart for better readability
    st.bar_chart(importance_df.set_index('Feature'), color="#9ACD32") 
    
    st.info(
        "**Interpretation:** Factors like **Coal Share** and **Previous Year's Emissions** often have the highest influence. Use this analysis to identify the most effective policy levers (e.g., reducing the Coal Share, which has a high importance).", 
        icon="💡"
    )

# --- Save this content as app.py in your main directory ---
"""

# Save the Streamlit app content to a file in the same directory for execution
with open('app.py', 'w') as f:
    f.write(APP_CODE)
    
print("\n" + "="*80)
print("✅ Project setup complete.")
print("The training script ran and saved model artifacts to the 'assets/' folder.")
print("The Streamlit app code has been saved as 'app.py'.")
print("\n--- Next Steps ---")
print("1. Open your terminal or Anaconda Prompt.")
print("2. Navigate to this project directory.")
print("3. Run the application using the command:")
print("\n    streamlit run app.py")
print("="*80)