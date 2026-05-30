import streamlit as st
import pandas as pd
import joblib

# Set browser page configuration
st.set_page_config(
    page_title="Abuja 2026 Housing AI",
    page_icon="🏠",
    layout="centered"
)

# App Title
st.title("🏠 Abuja Real Estate Intelligence Portal")
st.write("---")

# Setup Navigation Tabs
tab1, tab2 = st.tabs(["🔮 AI Price Predictor", "📊 Market Metrics Explorer"])

# --- Artifact Loading ---
@st.cache_resource
def load_model_artifacts():
    """Safely loads model files and caches them to stay super fast."""
    try:
        model = joblib.load('winning_abuja_lr_model.pkl')
        features = joblib.load('model_features.pkl')
        return model, features
    except FileNotFoundError:
        return None, None

model, model_features = load_model_artifacts()

# --- TAB 1: AI PRICE PREDICTOR ---
with tab1:
    if model is None:
        st.error("❌ Error: Missing model artifacts! Ensure 'winning_abuja_lr_model.pkl' and 'model_features.pkl' are in the same directory.")
    else:
        # Extract specific options from your training schema
        neigh_options = sorted([col.replace('neighbourhood_', '') for col in model_features if col.startswith('neighbourhood_')])
        prop_options = sorted([col.replace('property_type_', '') for col in model_features if col.startswith('property_type_')])

        # Fallback lists if training set columns are bare
        if not neigh_options:
            neigh_options = ['Utako', 'Apo', 'Gwarinpa', 'Guzape', 'Lokogoma', 'Wuse', 'Galadimawa', 'Dawaki', 'Kubwa', 'Asokoro', 'Life-Camp', 'Karsana', 'Lugbe', 'Jabi', 'Katampe', 'Maitama']
        if not prop_options:
            prop_options = ['Duplex', 'Flat/Apartment', 'Bungalow']

        # Sidebar Configuration Controls
        st.sidebar.header("🛠️ Property Configurations")
        selected_neigh = st.sidebar.selectbox("📍 Select Neighborhood", neigh_options)
        selected_prop = st.sidebar.selectbox("🏗️ Property Type", prop_options)
        
        st.sidebar.markdown("---")
        beds = st.sidebar.slider("🛏️ Bedrooms", min_value=1, max_value=10, value=4)
        baths = st.sidebar.slider("🚿 Bathrooms", min_value=1, max_value=10, value=4)
        toilets = st.sidebar.slider("🚽 Toilets", min_value=1, max_value=12, value=5)

        # Main Page Display Layout Card
        st.subheader("📋 Selected Specifications Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Neighborhood:** {selected_neigh}")
            st.write(f"**Property Type:** {selected_prop}")
        with col2:
            st.write(f"**Rooms:** {beds} Beds | {baths} Baths | {toilets} Toilets")

        st.write("---")

        # Calculation Trigger Button
        if st.button("🔮 Calculate Fair Market Value", type="primary"):
            input_row = pd.DataFrame(0, index=[0], columns=model_features)
            
            if 'bedrooms' in input_row.columns: input_row['bedrooms'] = beds
            if 'bathrooms' in input_row.columns: input_row['bathrooms'] = baths
            if 'toilets' in input_row.columns: input_row['toilets'] = toilets
            
            neigh_col = f"neighbourhood_{selected_neigh}"
            prop_col = f"property_type_{selected_prop}"
            
            if neigh_col in input_row.columns: input_row[neigh_col] = 1
            if prop_col in input_row.columns: input_row[prop_col] = 1
            
            prediction_array = model.predict(input_row)
            prediction = prediction_array[0]
            prediction = max(0, prediction)
            
            st.success("🎯 AI Model Valuation Complete!")
            st.metric(label="Estimated Market Value (Millions NGN)", value=f"₦{prediction:,.2f} M")
            
            html_display = f"""
            <div style="background-color:#f0f2f6; padding:15px; border-radius:10px; border-left: 5px solid #ff4b4b;">
                <h4 style="margin:0; color:#31333F;">Raw Currency Value:</h4>
                <p style="font-size:24px; font-weight:bold; margin:5px 0 0 0; color:#ff4b4b;">
                    ₦{prediction * 1_000_000:,.0f} NGN
                </p>
            </div>
            """
            st.markdown(html_display, unsafe_allow_html=True)

# --- TAB 2: MARKET METRICS EXPLORER ---
with tab2:
    st.subheader("📈 Abuja Neighborhood Metrics Dashboard")
    st.markdown("Explore real-time valuation metrics and underlying trends extracted from your validated datasets.")

    @st.cache_data
    def load_clean_data():
        """Attempts to load saved residential sales dataset for local rendering."""
        try:
            # Look for your exact clean dataset filename
            return pd.read_csv("abuja_2026_single_standard_sales.csv")
        except FileNotFoundError:
            return None

    df_sales_view = load_clean_data()

    if df_sales_view is not None:
        # Display high-level KPIs
        kpi1, kpi2 = st.columns(2)
        with kpi1:
            st.metric(label="📊 Cleaned Database Listings Size", value=f"{len(df_sales_view)} Homes")
        with kpi2:
            avg_market_price = df_sales_view['price_million_ngn'].median()
            st.metric(label="💰 Market-Wide Median Sale Price", value=f"₦{avg_market_price:,.1f} M")

        st.write("---")
        
        # Interactive Search Filter Block
        st.write("🔍 **Interactive Database Viewer**")
        search_query = st.text_input("Filter database listings by typing a neighborhood name (e.g. Wuse, Gwarinpa):")
        
        if search_query:
            filtered_df = df_sales_view[df_sales_view['neighbourhood'].str.contains(search_query, case=False, na=False)]
        else:
            filtered_df = df_sales_view

        # Display interactive grid table layout without default index column showing
        st.dataframe(filtered_df[['title', 'neighbourhood', 'property_type', 'bedrooms', 'price_formatted']], use_container_width=True, hide_index=True)
    else:
        # Fallback display info using your final audited Price-to-Rent hardcoded insights matrix
        st.warning("ℹ️ File 'abuja_2026_standard_sales.csv' not found. Displaying hardcoded audited investment metrics matrix:")
        
        # Recreating your precise clean tabular data frame matrix metrics representation
        static_metrics = pd.DataFrame({
            "Neighborhood": ["Utako", "Apo", "Gwarinpa", "Guzape", "Wuse", "Asokoro", "Life-Camp", "Lugbe", "Jabi", "Katampe", "Maitama"],
            "Median Sale (M)": [640.0, 232.5, 260.0, 650.0, 775.0, 550.0, 240.0, 120.0, 280.0, 350.0, 750.0],
            "Price-to-Rent Ratio": [53.33, 48.95, 43.33, 40.62, 34.44, 27.50, 26.67, 25.53, 21.54, 15.22, 15.00],
            "Investment Strategy": ["🔴 Rent (Speculative)", "🔴 Rent (Speculative)", "🔴 Rent (Speculative)", "🔴 Rent (Speculative)", "🟡 Balanced Market", "🟡 Balanced Market", "🟡 Balanced Market", "🟡 Balanced Market", "🟡 Balanced Market", "🟢 Buy (Goldmine)", "🟢 Buy (Goldmine)"]
        })
        st.dataframe(static_metrics, use_container_width=True, hide_index=True)
