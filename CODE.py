import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Set Project Title
st.title("Performance Analysis")

# Load Data
uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
   

    # Filters for Data
    st.sidebar.header("Filter Data")
    gender_filter = st.sidebar.multiselect("Select Gender", options=df['Gender'].unique(), default=df['Gender'].unique())
    attendance_filter = st.sidebar.slider("Select Attendance Range", min_value=int(df['Attendance'].min()), max_value=int(df['Attendance'].max()), value=(int(df['Attendance'].min()), int(df['Attendance'].max())))
    study_hours_filter = st.sidebar.slider("Select Hours Studied Range", min_value=int(df['Hours_Studied'].min()), max_value=int(df['Hours_Studied'].max()), value=(int(df['Hours_Studied'].min()), int(df['Hours_Studied'].max())))

    # Apply Filters
    filtered_df = df[(df['Gender'].isin(gender_filter)) & 
                     (df['Attendance'] >= attendance_filter[0]) & (df['Attendance'] <= attendance_filter[1]) & 
                     (df['Hours_Studied'] >= study_hours_filter[0]) & (df['Hours_Studied'] <= study_hours_filter[1])]
    st.write(filtered_df)

    # Convert categorical columns to numeric where necessary
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Ensure columns are numeric and handle non-numeric values
    df["Hours_Studied"] = pd.to_numeric(df["Hours_Studied"], errors='coerce')
    df["Attendance"] = pd.to_numeric(df["Attendance"], errors='coerce')
    df["Extracurricular_Activities"] = pd.to_numeric(df["Extracurricular_Activities"], errors='coerce')

    # Replace NaN values with 0 or handle as necessary
    df["Hours_Studied"].fillna(0, inplace=True)
    df["Attendance"].fillna(0, inplace=True)
    df["Extracurricular_Activities"].fillna(0, inplace=True)

    # Feature Engineering: Calculate Engagement_Score
    if "Engagement_Score" not in df.columns:
        df["Engagement_Score"] = df["Hours_Studied"] * 0.5 + df["Attendance"] * 0.3 + df["Extracurricular_Activities"] * 0.2
    df["Engagement_Score"] = df["Engagement_Score"]

    # Data Preprocessing: Label Encoding and Scaling
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include=['object']).columns:
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

    scaler = MinMaxScaler()
    df_encoded[df_encoded.columns] = scaler.fit_transform(df_encoded[df_encoded.columns])

    # Train-test split for X and y
    X = df_encoded[['Hours_Studied', 'Attendance', 'Engagement_Score']]  # Select the features
    y = df_encoded['Exam_Score']  # Select the target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3D Scatter Plot for feature interaction
    st.header("📊 3D Scatter Plot of Key Features")
    fig = px.scatter_3d(df, x='Hours_Studied', y='Attendance', z='Engagement_Score', color='Exam_Score', title="3D Scatter Plot of Hours Studied, Attendance, and Engagement Score")
    st.plotly_chart(fig)

    # Interactive Feature Importance Bar Chart
    st.header("🔍 Feature Importance")
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": model_rf.feature_importances_})
    st.bar_chart(feature_importance.set_index("Feature"))

    # Correlation Heatmap with Improved Color Scale and Filters
    st.header("🔍 Correlation Heatmap")
    corr_matrix = df.corr()
    # Filter out low correlations and adjust color scale
    corr_matrix = corr_matrix[abs(corr_matrix) > 0.5]  # Show only correlations with absolute value above 0.5

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, annot_kws={"size": 12})
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    st.pyplot(plt)

    # Interactive Visualization: Exam Score Distribution
    st.header("📈 Exam Score Distribution")
    num_bins = st.slider("Select Number of Bins for Histogram", min_value=5, max_value=50, value=20)
    fig = px.histogram(df, x="Exam_Score", nbins=num_bins, title="Exam Score Distribution")
    st.plotly_chart(fig)

    # Visualizing Impact of Attendance on Exam Scores
    st.header("📉 Impact of Attendance on Performance")
    fig = px.scatter(df, x='Attendance', y='Exam_Score', color='Gender', title="Attendance vs Exam Score")
    st.plotly_chart(fig)

    # Model Selection (Multiple Models)
    models_to_compare = st.multiselect("Select Models to Compare", ['Linear Regression', 'Random Forest'])

    performance = {}

    # Train models and compare
    if 'Linear Regression' in models_to_compare:
        model_lr = LinearRegression()
        model_lr.fit(X_train, y_train)
        y_pred_lr = model_lr.predict(X_test)
        mae_lr = mean_absolute_error(y_test, y_pred_lr)
        r2_lr = r2_score(y_test, y_pred_lr)
        performance['Linear Regression'] = {"MAE": mae_lr, "R² Score": r2_lr}

    if 'Random Forest' in models_to_compare:
        model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        model_rf.fit(X_train, y_train)
        y_pred_rf = model_rf.predict(X_test)
        mae_rf = mean_absolute_error(y_test, y_pred_rf)
        r2_rf = r2_score(y_test, y_pred_rf)
        performance['Random Forest'] = {"MAE": mae_rf, "R² Score": r2_rf}

    # Display Model Comparison Results
    st.write("### Model Performance Comparison")
    st.write(pd.DataFrame(performance).T)

    # Downloadable CSV
    csv = filtered_df.to_csv(index=False)
    st.download_button(label="Download Filtered Data", data=csv, file_name="filtered_data.csv", mime="text/csv")

    # Recommendations Based on Attendance and Study Hours
    if df["Attendance"].mean() < 75:
        st.warning("⚠️ Low attendance detected! Increasing attendance can improve performance.")
    if df["Hours_Studied"].mean() < 5:
        st.info("📚 Recommended: Study at least 7 hours weekly for better performance.")

    st.success("✅ Dashboard Ready! Explore and analyze student performance data.")
