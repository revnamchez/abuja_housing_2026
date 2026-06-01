import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Abuja 2026 Real Estate AI",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Abuja Dual Real Estate AI Portal")
st.markdown("""
**<span style="color: red;">RESEARCH TOPIC:</span>**
""", unsafe_allow_html=True)
st.write("Curated Scraped Web Data for Abuja House Price Prediction: A Practical Machine Learning Framework Using Regression Algorithms")
st.subheader("By: Nkiru Odoh and Dr. Nnaemeka U. Ezeonyi")
st.write("---")

tab1, tab2 = st.tabs(["🔮 AI Price Predictor", "📊 Market Metrics Explorer"])

# --- Cache Resource Loading for BOTH Models ---
@st.cache_resource
def load_all_artifacts():
    try:
        model_sale = joblib.load('winning_abuja_lr_model.pkl')
        feat_sale = joblib.load('model_features.pkl')
        model_rent = joblib.load('winning_abuja_rental_model.pkl')
        feat_rent = joblib.load('rental_features.pkl')
        return model_sale, feat_sale, model_rent, feat_rent
    except FileNotFoundError:
        return None, None, None, None

model_sale, feat_sale, model_rent, feat_rent = load_all_artifacts()

# --- TAB 1: AI PRICE PREDICTOR ---
with tab1:
    if model_sale is None or model_rent is None:
        st.error("❌ Error: Missing model artifacts! Ensure BOTH sales and rental '.pkl' files are in this directory.")
    else:
        # Configuration Controls
        st.sidebar.header("🛠️ Market & Property Setup")
        
        # NEW: Let the user select the type of transaction
        market_type = st.sidebar.radio("💼 Choose Market Type", ["Outright Sale", "Annual Rent"])
        
        # Use the specific features array based on chosen market type
        active_features = feat_sale if market_type == "Outright Sale" else feat_rent
        active_model = model_sale if market_type == "Outright Sale" else model_rent
        
        neigh_options = sorted([col.replace('neighbourhood_', '') for col in active_features if col.startswith('neighbourhood_')])
        prop_options = sorted([col.replace('property_type_', '') for col in active_features if col.startswith('property_type_')])

        selected_neigh = st.sidebar.selectbox("📍 Select Neighborhood", neigh_options)
        selected_prop = st.sidebar.selectbox("🏗️ Property Type", prop_options)
        
        st.sidebar.markdown("---")
        beds = st.sidebar.slider("🛏️ Bedrooms", min_value=1, max_value=10, value=4)
        baths = st.sidebar.slider("🚿 Bathrooms", min_value=1, max_value=10, value=4)
        toilets = st.sidebar.slider("🚽 Toilets", min_value=1, max_value=12, value=5)

        # Main Page Display
        st.subheader("📋 Selected Specifications Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Market Segment:** {market_type}")
            st.write(f"**Neighborhood:** {selected_neigh}")
        with col2:
            st.write(f"**Property Type:** {selected_prop}")
            st.write(f"**Rooms:** {beds} Beds | {baths} Baths")

        st.write("---")

        # Calculation Logic
        if st.button("🔮 Calculate Fair Market Value", type="primary"):
            input_row = pd.DataFrame(0, index=[0], columns=active_features)
            
            if 'bedrooms' in input_row.columns: input_row['bedrooms'] = beds
            if 'bathrooms' in input_row.columns: input_row['bathrooms'] = baths
            if 'toilets' in input_row.columns: input_row['toilets'] = toilets
            
            neigh_col = f"neighbourhood_{selected_neigh}"
            prop_col = f"property_type_{selected_prop}"
            
            if neigh_col in input_row.columns: input_row[neigh_col] = 1
            if prop_col in input_row.columns: input_row[prop_col] = 1
            
            prediction = max(0, float(active_model.predict(input_row)[0]))
            
            # Dynamic output text based on selection
            suffix = "Total Valuation" if market_type == "Outright Sale" else "Per Annum Rent"
            color = "#ff4b4b" if market_type == "Outright Sale" else "#2ca02c"
            
            st.success(f"🎯 AI {market_type} Valuation Complete!")
            st.metric(label=f"Estimated Cost ({suffix} - Millions NGN)", value=f"₦{prediction:,.2f} M")
            
            html_display = f"""
            <div style="background-color:#f0f2f6; padding:15px; border-radius:10px; border-left: 5px solid {color};">
                <h4 style="margin:0; color:#31333F;">Raw Currency Value ({suffix}):</h4>
                <p style="font-size:24px; font-weight:bold; margin:5px 0 0 0; color:{color};">
                    ₦{prediction * 1_000_000:,.0f} NGN
                </p>
            </div>
            """
            st.markdown(html_display, unsafe_allow_html=True)

# --- TAB 2: MARKET METRICS EXPLORER ---
with tab2:
    st.subheader("📊 Cleaned Investment Ratios Dashboard")
    static_metrics = pd.DataFrame({
        "Neighborhood": ["Utako", "Apo", "Gwarinpa", "Guzape", "Wuse", "Asokoro", "Life-Camp", "Lugbe", "Jabi", "Katampe", "Maitama"],
        "Median Sale (M)": [640.0, 232.5, 260.0, 650.0, 775.0, 550.0, 240.0, 120.0, 280.0, 350.0, 750.0],
        "Median Rent (M)": [12.0, 4.75, 6.00, 16.00, 22.50, 20.00, 9.00, 4.70, 13.00, 23.00, 50.00],
        "Price-to-Rent Ratio": [53.33, 48.95, 43.33, 40.62, 34.44, 27.50, 26.67, 25.53, 21.54, 15.22, 15.00],
        "Strategy": ["🔴 Rent", "🔴 Rent", "🔴 Rent", "🔴 Rent", "🟡 Balanced", "🟡 Balanced", "🟡 Balanced", "🟡 Balanced", "🟡 Balanced", "🟢 Buy (Goldmine)", "🟢 Buy (Goldmine)"]
    })
    st.dataframe(static_metrics, use_container_width=True, hide_index=True)
