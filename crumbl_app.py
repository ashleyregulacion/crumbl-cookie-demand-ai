import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np


# =============================================
# 🎨 APP STYLING
# =============================================
def set_app_style():
    st.set_page_config(layout="wide", page_title="Crumbl Cookies Sales Predictor", page_icon="🍪")
    st.markdown("""
    <style>
        .main {
            background-color: #FFF5E6;
        }
        .stSidebar {
            background-color: #FFEBCD !important;
        }
        .header-text {
            font-size: 24px !important;
            color: #D2691E !important;
            font-weight: bold !important;
        }
        .prediction-card {
            padding: 15px;
            margin-bottom: 10px;
        }
        .action-card {
            padding: 15px;
            margin: 5px 0;
        }
        .metric-card {
            background-color: #FFE4B5;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

set_app_style()

# =============================================
# 📊 DATA AND MODEL LOADING
# =============================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("crumbl_mock_data.csv")
        if 'sales' in df.columns:
            df = df.rename(columns={'sales': 'units_sold'})
        elif 'units_sold' not in df.columns:
            df['units_sold'] = np.random.randint(50, 200, size=len(df))
        return df
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        st.stop()

def load_model():
    try:
        model = joblib.load("crumbl_sales_model.pkl")
        feature_columns = joblib.load("feature_columns.pkl")  
        return model, feature_columns
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

df = load_data()
model, feature_columns = load_model()


# =============================================
# 🎛️ USER INPUT SIDEBAR
# =============================================
with st.sidebar:
    st.markdown("<div class='header-text'>🍪 Crumbl Sales Picker</div>", unsafe_allow_html=True)
   
    # Flavor selection
    flavor_options = {
        "Oreo": "🍪 Oreo",
        "Chocolate Chip": "🍫 Chocolate Chip",
        "Pink Sugar": "🌸 Pink Sugar",
        "Peanut Butter": "🥜 Peanut Butter",
        "Lemon Glaze": "🍋 Lemon Glaze"
    }
    flavor_selected = st.selectbox(
        "Cookie Flavor",
        options=sorted(df['flavor'].unique()),
        format_func=lambda x: flavor_options.get(x, x)
    )
   
    # Weather selection
    weather_icons = {
        "Sunny": "☀️ Sunny",
        "Cloudy": "☁️ Cloudy",
        "Rainy": "🌧️ Rainy"
    }
    weather_selected = st.selectbox(
        "Weather Condition",
        options=sorted(df['weather'].unique()),
        format_func=lambda x: weather_icons.get(x, x)
    )
   
    # Location selection
    location_selected = st.selectbox(
        "Store Location",
        options=sorted(df['location'].unique())
    )
   
    # Holidays and Social mentions
    col1, col2 = st.columns(2)
    with col1:
        is_holiday = st.checkbox("Holiday Week", help="Check if it's a holiday week")
    with col2:
        social_mentions = st.slider("Social Mentions", 0, 100, 0,
                                  help="Number of social media mentions expected")


# =============================================
# 📈 PREDICTION ENGINE
# =============================================
def make_prediction():
    try:
        input_data = pd.DataFrame(0, index=[0], columns=feature_columns)
        
        input_data['holiday'] = int(is_holiday)
        input_data['social_media_mentions'] = social_mentions
        
        # Set weather 
        weather_col = f"weather_{weather_selected}"
        if weather_col in input_data.columns:
            input_data[weather_col] = 1
        
        # Set flavor 
        flavor_col = f"flavor_{flavor_selected}"
        if flavor_col in input_data.columns:
            input_data[flavor_col] = 1
        
        # Set location
        location_col = f"location_{location_selected}"
        if location_col in input_data.columns:
            input_data[location_col] = 1

        input_data = input_data[feature_columns]
        
        return model.predict(input_data)[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None


# =============================================
# 🖥️ MAIN APP DISPLAY
# =============================================

st.markdown("""
            <h1 style="text-align: center; font-size: 50px; color: #D2691E; margin-bottom: 30px;">
            🍪 Welcome to the Crumbl Cookie Sales Predictor!
            </h1>
            """, unsafe_allow_html=True)
st.markdown("<div class='header-text'> Crumbl Cookie Sales Predictor</div>", unsafe_allow_html=True)


if st.button("✨ Generate Prediction", use_container_width=True):
    with st.spinner("Crunching cookie data..."):
        prediction = make_prediction()

        if prediction:
            avg_sales = df.groupby('flavor')['units_sold'].mean()[flavor_selected]
            change_pct = (prediction / avg_sales - 1) * 100
           
            # ========== PREDICTION CARD ==========
            with st.container():
                st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
               
                # Scenario format
                scenario_text = f"""
                ➡️ {flavor_selected} cookies during {weather_selected} weather at {location_selected} location
                {'➡️ (Holiday Week)' if is_holiday else ''}
                """

                st.markdown(f"""
                <h3>📋 Scenario Summary</h3>
                <p style="font-size: 16px; margin-bottom: 10px;">{scenario_text}</p>
                """, unsafe_allow_html=True)
               
                # Columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>Predicted Sales</h3>
                        <h2>{prediction:,.0f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
               
                with col2:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>Vs Average</h3>
                        <h2>{change_pct:+.0f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
               
                with col3:
                    batches = int(prediction//12)
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>Batches Needed</h3>
                        <h2>{batches:,.0f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
               
                st.markdown("</div>", unsafe_allow_html=True)
           
            # ========== ACTION RECOMMENDATIONS ==========
            with st.container():
                st.markdown("<div class='action-card'>", unsafe_allow_html=True)
                st.markdown("---")
                st.subheader("📌 Recommended Actions")
               
                if change_pct > 15:
                    st.success(f"""
                    - **🛒 Order Ingredients:** {int(prediction//12):,.0f} batches needed
                    - **👨‍🍳 Staffing:** Schedule {int(prediction//150)} extra bakers
                    - **📱 Marketing:** Boost posts about {flavor_selected}
                    - **📦 Inventory:** Prepare 20% extra packaging
                    """)
                elif change_pct < -10:
                    st.warning("""
                    - **📉 Reduce Orders:** Lower ingredient purchase by 15%
                    - **🔄 Cross-Train Staff:** Use slow period for training
                    """)
                else:
                    st.info("""
                    - **🔄 Maintain Standard Operations**
                    - **👀 Monitor Sales:** Be ready to adjust
                    """)
               
                st.markdown("</div>", unsafe_allow_html=True)
           
            # ========== DATA VISUALIZATIONS ==========
            st.markdown("---")
            st.subheader("📊 Performance Insights")
           
            tab1, tab2, tab3 = st.tabs(["Trend Analysis", "Weather Impact", "Location Comparison"])
           
            with tab1:
                if 'week' in df.columns:
                    weekly_data = df[df['flavor'] == flavor_selected].groupby('week')['units_sold'].mean().reset_index()
                    fig = px.line(
                        weekly_data,
                        x='week',
                        y='units_sold',
                        title=f"Weekly Sales Trend for {flavor_selected}",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
           
            with tab2:
                fig2 = px.box(
                    df[df['flavor'] == flavor_selected],
                    x='weather',
                    y='units_sold',
                    color='weather',
                    title=f"Weather Impact on {flavor_selected} Sales",
                    template="plotly_white"
                )
                st.plotly_chart(fig2, use_container_width=True)
           
            with tab3:
                loc_data = df[df['flavor'] == flavor_selected].groupby('location')['units_sold'].mean().reset_index()
                fig3 = px.bar(
                    loc_data,
                    x='location',
                    y='units_sold',
                    color='location',
                    title=f"{flavor_selected} Sales by Location",
                    template="plotly_white"
                )
                st.plotly_chart(fig3, use_container_width=True)


# =============================================
# 🔍 DEBUG SECTION (COLLAPSIBLE)
# =============================================
with st.expander("⚙️ Data Summary & Debug Info"):
    st.write("### Dataset Columns:", df.columns.tolist())
    st.write("### Sample Data:", df.head(3))
    st.write("### Model Features Expected:", feature_columns)