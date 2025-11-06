import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score

# Set page configuration
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and scaler
@st.cache_resource
def load_model():
    with open('wine_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# Feature descriptions
feature_descriptions = {
    'fixed acidity': 'Primary fixed acids in wine (tartaric, malic, citric)',
    'volatile acidity': 'Amount of acetic acid in wine',
    'citric acid': 'Amount of citric acid (adds freshness and flavor)',
    'residual sugar': 'Amount of sugar remaining after fermentation',
    'chlorides': 'Amount of salt in wine',
    'free sulfur dioxide': 'Free form of SO2 (prevents microbial growth)',
    'total sulfur dioxide': 'Total amount of SO2 (free + bound forms)',
    'density': 'Density of wine (close to water density)',
    'pH': 'pH level of wine (measure of acidity)',
    'sulphates': 'Amount of potassium sulphate (additive)',
    'alcohol': 'Alcohol content percentage'
}

# Sidebar
st.sidebar.title("🍷 Wine Quality Predictor")
st.sidebar.markdown("""
This app predicts wine quality based on physicochemical properties using Machine Learning.
""")

# Main content
st.title("🍷 Wine Quality Prediction")
st.markdown("Predict wine quality (0-10 scale) based on physicochemical properties")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📊 Data Analysis", "📈 Model Info", "ℹ️ About"])

with tab1:
    st.header("Wine Quality Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Features")
        
        # User input fields
        fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 8.0, 0.1, help=feature_descriptions['fixed acidity'])
        volatile_acidity = st.slider("Volatile Acidity", 0.1, 2.0, 0.5, 0.01, help=feature_descriptions['volatile acidity'])
        citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.25, 0.01, help=feature_descriptions['citric acid'])
        residual_sugar = st.slider("Residual Sugar", 0.5, 16.0, 2.5, 0.1, help=feature_descriptions['residual sugar'])
        chlorides = st.slider("Chlorides", 0.01, 0.2, 0.08, 0.001, help=feature_descriptions['chlorides'])
        
    with col2:
        st.subheader("")
        free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1, 72, 15, 1, help=feature_descriptions['free sulfur dioxide'])
        total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6, 289, 45, 1, help=feature_descriptions['total sulfur dioxide'])
        density = st.slider("Density", 0.99, 1.01, 0.996, 0.001, help=feature_descriptions['density'])
        pH = st.slider("pH", 2.7, 4.0, 3.3, 0.01, help=feature_descriptions['pH'])
        sulphates = st.slider("Sulphates", 0.3, 2.0, 0.65, 0.01, help=feature_descriptions['sulphates'])
        alcohol = st.slider("Alcohol", 8.0, 15.0, 10.5, 0.1, help=feature_descriptions['alcohol'])
    
    # Create input array
    input_features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                              chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                              density, pH, sulphates, alcohol]])
    
    # Scale features and make prediction
    input_scaled = scaler.transform(input_features)
    prediction = model.predict(input_scaled)[0]
    
    # Display prediction
    st.markdown("---")
    col1, col2, col3 = st.columns([1,2,1])
    
    with col2:
        st.subheader("Prediction Result")
        
        # Create a quality meter
        quality_score = max(0, min(10, prediction))
        quality_percentage = (quality_score / 10) * 100
        
        # Quality interpretation
        if quality_score >= 7:
            quality_text = "High Quality 🎉"
            color = "green"
        elif quality_score >= 5:
            quality_text = "Medium Quality 👍"
            color = "orange"
        else:
            quality_text = "Low Quality 👎"
            color = "red"
        
        # Display quality meter
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = quality_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Wine Quality Score: {quality_text}"},
            gauge = {
                'axis': {'range': [None, 10]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 3], 'color': "lightgray"},
                    {'range': [3, 7], 'color': "gray"},
                    {'range': [7, 10], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 9}
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"**Predicted Quality Score:** {prediction:.2f}/10")

with tab2:
    st.header("Data Analysis")
    
    # Load sample data for visualization
    @st.cache_data
    def load_data():
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        return pd.read_csv(url, sep=';')
    
    wine_data = load_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Overview")
        st.dataframe(wine_data.head(10))
        
        st.metric("Total Samples", len(wine_data))
        st.metric("Number of Features", len(wine_data.columns) - 1)
        
    with col2:
        st.subheader("Quality Distribution")
        fig = px.histogram(wine_data, x='quality', 
                          title='Distribution of Wine Quality Scores',
                          color_discrete_sequence=['#FF6B6B'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature correlations
    st.subheader("Feature Correlations with Quality")
    correlations = wine_data.corr()['quality'].sort_values(ascending=False)[1:]
    
    fig = px.bar(x=correlations.values, y=correlations.index,
                orientation='h',
                title='Feature Correlation with Wine Quality',
                color=correlations.values,
                color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Details")
        st.write("**Algorithm:** Random Forest Regressor")
        st.write("**Number of Trees:** 100")
        st.write("**Features Used:** 11 physicochemical properties")
        
        st.subheader("Feature Importance")
        feature_names = list(feature_descriptions.keys())
        importance_scores = model.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(importance_df, x='Importance', y='Feature',
                    orientation='h', title='Feature Importance',
                    color='Importance')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Model Performance")
        st.info("""
        **Evaluation Metrics (on test set):**
        - R² Score: ~0.45-0.50
        - Mean Absolute Error: ~0.45
        - Mean Squared Error: ~0.38
        """)
        
        st.subheader("Quality Interpretation")
        st.write("""
        - **7-10:** High Quality Wine
        - **5-7:** Medium Quality Wine  
        - **0-5:** Low Quality Wine
        """)

with tab4:
    st.header("About This Project")
    
    st.markdown("""
    ### 🍷 Wine Quality Prediction ML Project
    
    This machine learning project predicts wine quality based on physicochemical properties using a Random Forest Regressor.
    
    **Dataset Features:**
    - Fixed acidity
    - Volatile acidity  
    - Citric acid
    - Residual sugar
    - Chlorides
    - Free sulfur dioxide
    - Total sulfur dioxide
    - Density
    - pH
    - Sulphates
    - Alcohol
    
    **Model:**
    - Algorithm: Random Forest Regressor
    - Training data: Wine Quality Dataset from UCI ML Repository
    - Target: Quality score (0-10 scale)
    
    **Deployment:**
    - Framework: Streamlit
    - Deployment: Streamlit Community Cloud
    """)
    
    st.success("""
    🚀 **How to Use:**
    1. Go to the **Prediction** tab
    2. Adjust the wine parameters using sliders
    3. View the predicted quality score
    4. Explore data analysis in other tabs
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Machine Learning Project")