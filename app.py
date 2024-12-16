import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set page configuration at the top, before any Streamlit components
st.set_page_config(page_title="ChurnSage: AI-Powered Customer Insights", layout="wide")

# Load background image
background_image = "E:\churn\images\background.jpg"  # Ensure the path matches where the image is saved

# Apply background image using CSS
st.markdown(f"""
    <style>
        .reportview-container {{
            background-image: url({background_image});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .block-container {{
            background-color: rgba(0, 0, 0, 0.5);  /* Semi-transparent background for readability */
            border-radius: 15px;
            padding: 30px;
            max-width: 800px;
            margin: 0 auto;
        }}
    </style>
""", unsafe_allow_html=True)

# Function to preprocess the data
def preprocess_data(df):
    """Handle missing values and perform necessary preprocessing."""
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df[column].fillna(df[column].mean(), inplace=True)

    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = df['SeniorCitizen'].map({0: "No", 1: "Yes"})
    
    return df

# Function to encode categorical columns
def encode_categorical_columns(df):
    """Encode categorical columns into numeric format."""
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = LabelEncoder().fit_transform(df[column])
    return df

# Function to generate churn rate pie chart
def plot_churn_rate(df):
    """Generate a pie chart of churn rate distribution."""
    churn_column = identify_churn_column(df)
    
    if churn_column is None:
        st.warning("No churn-related column detected in the dataset!")
        return None
    
    churn_values = df[churn_column].value_counts()
    
    # If churn values are numeric (0/1), convert them to Yes/No labels
    if set(churn_values.index) == {0, 1}:
        churn_labels = ['Churn: No', 'Churn: Yes']
    else:
        churn_labels = churn_values.index.tolist()

    fig = go.Figure(data=[go.Pie(labels=churn_labels, values=churn_values, hole=.3)])
    fig.update_layout(title_text="Churn Rate Distribution", showlegend=True)
    return fig

# Function to identify the churn column in the dataset
def identify_churn_column(df):
    """Automatically identify the churn column from the dataset."""
    # List of potential churn column names to look for
    potential_columns = ['Churn', 'Exited', 'Target', 'Attrition', 'Churned', 'IsChurn', 'customer_churn']

    for column in df.columns:
        if column in potential_columns:
            return column
        unique_values = df[column].dropna().unique()
        if set(unique_values) == {0, 1} or set(unique_values) == {'Yes', 'No'}:
            return column
    
    return None

# Function for data analysis
def data_analysis(df):
    """Perform basic data analysis and visualization."""
    st.subheader("Basic Statistical Summary")
    st.write(df.describe())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.subheader("Distribution of Numerical Features")
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[column], kde=True, ax=ax)
        ax.set_title(f'Distribution of {column}')
        st.pyplot(fig)

# Function to plot confusion matrix
def plot_confusion_matrix(cm):
    fig_cm, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False, annot_kws={"size": 16})
    ax.set_title("Confusion Matrix", fontsize=16)
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('Actual', fontsize=14)
    return fig_cm

# Function to plot feature importance
def plot_feature_importance(features, importances):
    """Plot a bar chart for feature importance."""
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    })
    
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax, palette='viridis')
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    
    return fig

# Streamlit app
def main():
    # Sidebar with navigation
    page = st.sidebar.radio("Navigation", ["Home", "Upload Data", "Model Prediction", "Results", "Data Analysis"])

    # --- HOME PAGE ---
    if page == "Home":
        st.markdown("""
        <div style="background-color: rgba(0, 0, 0, 0.5); border-radius: 15px; padding: 30px; max-width: 800px; margin: 0 auto;">
            <h1 style="color: rgb(231, 229, 236); text-align: center; font-size: 42px; font-weight: bold; margin-bottom: 10px;">
                🚀 Welcome to ChurnSage
            </h1>
            <h2 style="color: rgb(255, 255, 255); text-align: center; font-size: 26px; margin-bottom: 30px;">
                AI-Powered Customer Insights & Churn Prediction
            </h2>
            <p style="color: rgb(255, 255, 255); font-size: 18px; line-height: 1.6; text-align: justify;">
                ChurnSage is a powerful AI-based solution that helps businesses predict customer churn with precision. 
                Designed for data-driven professionals, our app leverages advanced machine learning techniques to 
                provide actionable insights, reduce churn, and boost customer retention.
            </p>
            <p style="color: rgb(255, 255, 255); font-size: 18px; line-height: 1.6; text-align: justify;">
                📊 <strong>What you can do with ChurnSage</strong>:
                <ul style="color: rgb(255, 255, 255); font-size: 18px; line-height: 1.6;">
                    <li>Upload and preprocess your churn dataset effortlessly.</li>
                    <li>Visualize customer churn trends and correlations interactively.</li>
                    <li>Train and evaluate machine learning models with a few clicks.</li>
                    <li>Understand feature importance for impactful decision-making.</li>
                </ul>
            </p>
            <p style="color: rgb(255, 255, 255); font-size: 18px; line-height: 1.6; text-align: center; margin-top: 30px;">
                Start your journey to smarter customer retention with <strong>ChurnSage</strong> today! 🌟
            </p>
        </div>
        """, unsafe_allow_html=True)

    # --- UPLOAD DATA ---
    elif page == "Upload Data":
        st.title("Upload Your Dataset")
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.subheader("Dataset Preview")
            st.dataframe(df.head())
            st.session_state.df = df  # Store dataframe in session state

    # --- MODEL PREDICTION ---
    elif page == "Model Prediction":
        if 'df' not in st.session_state:
            st.warning("Please upload a dataset first in the 'Upload Data' page.")
            return
        
        df = st.session_state.df
        st.title("Select Features and Train Model")
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        selected_features = st.multiselect('Choose numerical columns for features', numerical_columns, default=numerical_columns)
        churn_column = identify_churn_column(df)
        
        if churn_column is None:
            st.warning("No churn-related column detected in the dataset.")
            return
        
        target_column = churn_column  # Use the detected churn column
        if selected_features and target_column:
            df = preprocess_data(df)
            df = encode_categorical_columns(df)
            features = df[selected_features]
            target = df[target_column]

            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.30, random_state=42, stratify=target)
            
            # Standardize numerical columns
            scaler = StandardScaler()
            X_train[selected_features] = scaler.fit_transform(X_train[selected_features])
            X_test[selected_features] = scaler.transform(X_test[selected_features])
            
            # Random Forest Classifier
            rf_model = RandomForestClassifier(n_estimators=500, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Store model and features in session state
            st.session_state.rf_model = rf_model
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.selected_features = selected_features  # Store selected features
            
            st.success("Model trained successfully! You can now view the results.")

    # --- RESULTS ---
    elif page == "Results":
        if 'rf_model' not in st.session_state:
            st.warning("Please train the model first on the 'Model Prediction' page.")
            return
        
        rf_model = st.session_state.rf_model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        # Model predictions
        rf_predictions = rf_model.predict(X_test)

        # Accuracy Score
        accuracy = accuracy_score(y_test, rf_predictions)
        
        # Create a 3-column layout for displaying results
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.subheader(f'Accuracy: {accuracy*100:.2f}%')
            st.write("The model's accuracy on the test set.")

        with col2:
            # Display classification report in an expander
            with st.expander("Classification Report"):
                st.text(classification_report(y_test, rf_predictions))

        with col3:
            # Churn rate graph (pie chart)
            with st.expander("Churn Rate Distribution"):
                churn_rate_graph = plot_churn_rate(st.session_state.df)
                if churn_rate_graph:
                    st.plotly_chart(churn_rate_graph)

        # Confusion matrix visualization in an expander
        with st.expander('Confusion Matrix'):
            cm = confusion_matrix(y_test, rf_predictions)
            fig_cm = plot_confusion_matrix(cm)
            st.pyplot(fig_cm)

        # Feature importance visualization in an expander
        with st.expander("Feature Importance"):
            feature_importances = rf_model.feature_importances_
            fig_importance = plot_feature_importance(st.session_state.selected_features, feature_importances)
            st.pyplot(fig_importance)

    # --- DATA ANALYSIS ---
    elif page == "Data Analysis":
        if 'df' not in st.session_state:
            st.warning("Please upload a dataset first in the 'Upload Data' page.")
            return

        df = st.session_state.df
        st.title("Data Analysis")
        data_analysis(df)

# Run the app
if __name__ == "__main__":
    main()
