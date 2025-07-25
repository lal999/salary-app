import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded"
)


# --- Theme Toggle ---
theme = st.sidebar.radio("🌗 Theme", ["Light Mode", "Dark Mode"])

# --- Dynamic CSS Injection ---
if theme == "Dark Mode":
    st.markdown("""
        <style>
            body {
                background-color: #9ECAD6;
                color: #ffffff;
            }
            .stApp {
                background-color: #9ECAD6;
            }
            .css-1d391kg, .css-18e3th9 {
                background-color: #2e2e2e !important;
                color: #ffffff !important;
            }
            .stSelectbox>div>div {
                color: black;
            }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            body {
                background-color: #EDE8DC;
                color: #000000;
            }
            .stApp {
                background-color: #EDE8DC;
            }
            .css-1d391kg, .css-18e3th9 {
                background-color: #f9f9f9 !important;
                color: #000000 !important;
            }
        </style>
    """, unsafe_allow_html=True)


# Sidebar Welcome Message
st.sidebar.markdown("## 👋 Welcome!")
st.sidebar.write("This app predicts the salary of an employee based on several inputs like age, experience, education, and job title.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Instructions:**")
st.sidebar.write("""
- Fill in your personal and job details in the form.
- Click on **Predict Salary** to see the estimated salary.
- Scroll down to view model performance and insights.
""")


# Custom CSS

st.markdown("""
<style>
    .header {
        font-size: 32px;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 10px;
        text-align: center;
    }
    .block-container {
        padding-top: 1rem;
    }
    .metric-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-title {
        font-size: 14px;
        color: #7f8c8d;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #2c3e50;
    }
    .plot-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .form-box {
        background-color: #f9f9f9;
        padding: 30px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        margin-bottom: 30px;
    }       
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
    <div style="text-align: center;">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcREp9mfNgIxRKyajEc7cu7btGj2HgV6NKDBUA&s" alt="logo" width="80">
        <h1>Employee Salary Prediction</h1>
    </div>
""", unsafe_allow_html=True)
st.write("Fill in your details to get an estimated salary prediction")
st.markdown("---")

# Load and preprocess data
@st.cache_data
def load_and_preprocess():
    df = pd.read_csv("Salary Data.csv")
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    # Create label encoders
    le_gender = LabelEncoder()
    le_education = LabelEncoder()
    le_job = LabelEncoder()
    
    df['Gender'] = le_gender.fit_transform(df['Gender'])
    df['Education Level'] = le_education.fit_transform(df['Education Level'])
    df['Job Title'] = le_job.fit_transform(df['Job Title'])
    
    return df, le_gender, le_education, le_job

df, le_gender, le_education, le_job = load_and_preprocess()

# Train model and calculate metrics
@st.cache_data
def train_and_evaluate():
    X = df.drop("Salary", axis=1)
    y = df["Salary"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_poly)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, poly, mae, mse, r2, X_test, y_test, y_pred

model, poly, mae, mse, r2, X_test, y_test, y_pred = train_and_evaluate()

# Get unique values for dropdowns
original_df = pd.read_csv("Salary Data.csv").dropna()
education_levels = sorted(original_df['Education Level'].unique())
job_titles = sorted(original_df['Job Title'].unique())
genders = sorted(original_df['Gender'].unique())

# Input sections
# 💠 Stylish Form Box


st.markdown("### 📥 Enter Your Details")

# Form inputs in two columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Education Level**")
    education = st.selectbox(
        "Select education level", 
        options=[""] + education_levels,
        index=0,
        key="education",
        help="Select your highest education level",
        format_func=lambda x: 'Select education' if x == "" else x
    )

    st.markdown("**Job Title**")
    job_title = st.selectbox(
        "Select job title", 
        options=[""] + job_titles,
        index=0,
        key="job_title",
        help="Select your current job title",
        format_func=lambda x: 'Select job title' if x == "" else x
    )

with col2:
    st.markdown("**Years of Experience**")
    experience = st.number_input(
        "Years of experience", 
        min_value=0, 
        max_value=50, 
        value=None,
        placeholder="Enter years",
        key="experience"
    )

    st.markdown("**Age**")
    age = st.number_input(
        "Your age", 
        min_value=18, 
        max_value=70, 
        value=None,
        placeholder="Enter age",
        key="age"
    )

# Second row
col3, col4 = st.columns(2)

with col3:
    st.markdown("**Gender**")
    gender = st.selectbox(
        "Select gender", 
        options=[""] + genders,
        index=0,
        key="gender",
        help="Select your gender",
        format_func=lambda x: 'Select gender' if x == "" else x
    )

with col4:
    st.markdown("**Location**")
    location = st.selectbox(
        "Select location", 
        options=[""] + ["New York", "San Francisco", "Chicago", "Boston", "Seattle"],
        index=0,
        key="location",
        help="Select your work location",
        format_func=lambda x: 'Select location' if x == "" else x
    )

# ✅ Prediction button
all_fields_filled = all([
    education != "",
    job_title != "",
    experience is not None,
    age is not None,
    gender != "",
    location != ""
])

predict_button = st.button("💰 Predict Salary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)  # 🔒 Close form box

# 🔮 Prediction Logic
if predict_button and all_fields_filled:
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Education Level': [education],
        'Job Title': [job_title],
        'Years of Experience': [experience]
    })

    input_data['Gender'] = le_gender.transform(input_data['Gender'])
    input_data['Education Level'] = le_education.transform(input_data['Education Level'])
    input_data['Job Title'] = le_job.transform(input_data['Job Title'])

    input_poly = poly.transform(input_data)
    prediction = model.predict(input_poly)

    st.success(f"### 🤑 Predicted Salary: ${prediction[0]:,.2f}")
elif predict_button and not all_fields_filled:
    st.warning("⚠️ Please fill in all fields before predicting.")


# Model Performance Section
st.markdown("---")
st.markdown("## Model Performance Metrics")

# Metrics in columns
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-box"><div class="metric-title">R² Score</div><div class="metric-value">{:.3f}</div></div>'.format(r2), 
                unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-box"><div class="metric-title">Mean Absolute Error</div><div class="metric-value">${:,.0f}</div></div>'.format(mae), 
                unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-box"><div class="metric-title">Mean Squared Error</div><div class="metric-value">{:,.0f}</div></div>'.format(mse), 
                unsafe_allow_html=True)
st.markdown("---")
# Visualizations
st.set_page_config(layout="wide")
st.markdown("## Model Visualizations")

with st.container():
    st.markdown("### 👁️‍🗨️ Visualization Dashboard")
    
    col1, col2 = st.columns(2)
    
    # Actual vs Predicted Plot
    with col1:
        st.markdown("#### 🎯 Feature Importance")
        coefficients = pd.DataFrame({
            'Feature': poly.get_feature_names_out(X_test.columns),
            'Coefficient': model.coef_
        }).sort_values('Coefficient', ascending=False).head(10)

        fig1, ax1 = plt.subplots(figsize=(6,5))
        sns.barplot(x='Coefficient', y='Feature', data=coefficients, ax=ax1, palette='viridis')
        ax1.set_title('Top 10 Important Features')
        ax1.set_xlabel('Coefficient Value')
        ax1.set_ylabel('Feature')
        st.pyplot(fig1)

    # Feature Importance Plot
    with col2:
        st.markdown("#### 🔥 Feature Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(6,5))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax2)
        ax2.set_title('Correlation Matrix of Features')
        st.pyplot(fig2)

    # Correlation Heatmap (Full Width)
    st.markdown("#### 📌 Actual vs Predicted Salaries")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.scatter(y_test, y_pred, alpha=0.5, color='#3498db')
    ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax3.set_xlabel('Actual Salary')
    ax3.set_ylabel('Predicted Salary')
    ax3.set_title('Actual vs Predicted')
    ax3.grid(True)
    st.pyplot(fig3)

st.markdown("---")
st.markdown("""
    <div style="background-color:#748DAE;padding:20px;border-radius:12px;color:white;margin-top:50px;">
        <h3>🎓 Internship Project</h3>
        <p><strong>Company:</strong> Edunet Foundation in collaboration with AICTE and IBM</p>
        <p><strong>Creator:</strong> Palikila Likhita Reddy</p>
        <p><strong>Project:</strong> Machine Learning Salary Predictor</p>
        <p><strong>Duration:</strong> 6 weeks</p>
    </div>
""", unsafe_allow_html=True)



       