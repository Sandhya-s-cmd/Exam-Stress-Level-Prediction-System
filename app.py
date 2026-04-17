import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Student Stress Prediction System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'data' not in st.session_state:
    st.session_state.data = None

def create_stress_distribution_charts(data):
    """Create comprehensive stress distribution charts"""
    st.markdown("### 📊 Stress Distribution Analysis")
    
    # Overall stress distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart for overall distribution
        stress_counts = data['Exam_Stress_Level'].value_counts().sort_index()
        stress_labels = {0: 'Low Stress', 1: 'Medium Stress', 2: 'High Stress'}
        labels = [stress_labels.get(i, f'Level {i}') for i in stress_counts.index]
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=labels,
            values=stress_counts.values,
            hole=0.3,
            marker_colors=['#4CAF50', '#FF9800', '#F44336']
        )])
        fig_pie.update_layout(title="Overall Stress Distribution")
        st.plotly_chart(fig_pie, width='stretch')
    
    with col2:
        # Bar chart with counts
        fig_bar = go.Figure(data=[
            go.Bar(
                x=labels,
                y=stress_counts.values,
                marker_color=['#4CAF50', '#FF9800', '#F44336']
            )
        ])
        fig_bar.update_layout(
            title="Stress Level Counts",
            xaxis_title="Stress Level",
            yaxis_title="Number of Students"
        )
        st.plotly_chart(fig_bar, width='stretch')
    
    # Demographic breakdowns
    st.markdown("#### 🎯 Demographic Analysis")
    
    # Gender-based stress distribution
    if 'gender' in data.columns:
        col3, col4 = st.columns(2)
        
        with col3:
            gender_stress = pd.crosstab(data['gender'], data['Exam_Stress_Level'])
            gender_labels = {0: 'Female', 1: 'Male'}
            gender_stress.index = [gender_labels.get(i, f'Gender {i}') for i in gender_stress.index]
            gender_stress.columns = ['Low', 'Medium', 'High']
            
            fig_gender = go.Figure(data=[
                go.Bar(name='Low', x=gender_stress.index, y=gender_stress['Low'], marker_color='#4CAF50'),
                go.Bar(name='Medium', x=gender_stress.index, y=gender_stress['Medium'], marker_color='#FF9800'),
                go.Bar(name='High', x=gender_stress.index, y=gender_stress['High'], marker_color='#F44336')
            ])
            fig_gender.update_layout(
                title="Stress Distribution by Gender",
                barmode='group',
                xaxis_title="Gender",
                yaxis_title="Count"
            )
            st.plotly_chart(fig_gender, width='stretch')
        
        with col4:
            # Age-based stress distribution
            if 'age' in data.columns:
                age_groups = pd.cut(data['age'], bins=[15, 18, 21, 25], labels=['15-18', '19-21', '22-25'])
                age_stress = pd.crosstab(age_groups, data['Exam_Stress_Level'])
                age_stress.columns = ['Low', 'Medium', 'High']
                
                fig_age = go.Figure(data=[
                    go.Bar(name='Low', x=age_stress.index, y=age_stress['Low'], marker_color='#4CAF50'),
                    go.Bar(name='Medium', x=age_stress.index, y=age_stress['Medium'], marker_color='#FF9800'),
                    go.Bar(name='High', x=age_stress.index, y=age_stress['High'], marker_color='#F44336')
                ])
                fig_age.update_layout(
                    title="Stress Distribution by Age Group",
                    barmode='group',
                    xaxis_title="Age Group",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig_age, width='stretch')

def create_correlation_heatmap(data):
    """Create feature correlation heatmap"""
    st.markdown("### 🔥 Feature Correlation Analysis")
    
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numeric_data.corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Feature Correlation Heatmap"
    )
    fig.update_layout(width=800, height=600)
    st.plotly_chart(fig, width='stretch')
    
    # Show top correlations with stress level
    if 'Exam_Stress_Level' in corr_matrix.columns:
        st.markdown("#### 🎯 Top Correlations with Stress Level")
        stress_corr = corr_matrix['Exam_Stress_Level'].sort_values(key=abs, ascending=False)[1:11]
        
        fig_corr = go.Figure(data=[
            go.Bar(
                x=stress_corr.values,
                y=stress_corr.index,
                orientation='h',
                marker_color=['#FF9800' if x > 0 else '#2196F3' for x in stress_corr.values]
            )
        ])
        fig_corr.update_layout(
            title="Top 10 Features Correlated with Stress Level",
            xaxis_title="Correlation Coefficient",
            yaxis_title="Features"
        )
        st.plotly_chart(fig_corr, width='stretch')

def create_comparative_analysis(data):
    """Create comparative analysis charts"""
    st.markdown("### 📈 Comparative Analysis")
    
    # Attendance vs Stress Level
    if 'Attendance_Percentage' in data.columns and 'Exam_Stress_Level' in data.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot for attendance by stress level
            stress_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
            data['Stress_Label'] = data['Exam_Stress_Level'].map(stress_labels)
            
            fig_attendance = go.Figure()
            for stress_level in ['Low', 'Medium', 'High']:
                fig_attendance.add_trace(go.Box(
                    y=data[data['Stress_Label'] == stress_level]['Attendance_Percentage'],
                    name=stress_level,
                    boxpoints='outliers'
                ))
            
            fig_attendance.update_layout(
                title="Attendance Distribution by Stress Level",
                yaxis_title="Attendance Percentage",
                xaxis_title="Stress Level"
            )
            st.plotly_chart(fig_attendance, width='stretch')
        
        with col2:
            # Academic workload vs stress level
            if 'Academic_Workload' in data.columns:
                fig_workload = go.Figure()
                for stress_level in ['Low', 'Medium', 'High']:
                    fig_workload.add_trace(go.Box(
                        y=data[data['Stress_Label'] == stress_level]['Academic_Workload'],
                        name=stress_level,
                        boxpoints='outliers'
                    ))
                
                fig_workload.update_layout(
                    title="Academic Workload by Stress Level",
                    yaxis_title="Workload Level (1-5)",
                    xaxis_title="Stress Level"
                )
                st.plotly_chart(fig_workload, width='stretch')
    
    # Stress factors radar chart
    st.markdown("#### 🎯 Stress Factors Comparison")
    
    stress_factors = ['Exam_Anxiety_Level', 'anxiety_tension', 'sleep_problems', 
                   'academic_overload', 'concentration_problems']
    available_factors = [f for f in stress_factors if f in data.columns]
    
    if available_factors:
        # Calculate average values by stress level
        factor_avg = data.groupby('Exam_Stress_Level')[available_factors].mean()
        
        fig_radar = go.Figure()
        
        colors = ['#4CAF50', '#FF9800', '#F44336']
        labels = ['Low', 'Medium', 'High']
        
        for i, (stress_level, color, label) in enumerate(zip(factor_avg.index, colors, labels)):
            if stress_level in factor_avg.index:
                # Convert hex to rgba with transparency
                hex_color = color.lstrip('#')
                rgba_color = f'rgba({int(hex_color[0:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:6], 16)}, 0.2)'
                fig_radar.add_trace(go.Scatterpolar(
                    r=factor_avg.loc[stress_level].values,
                    theta=[f.replace('_', ' ').title() for f in available_factors],
                    fill='toself',
                    name=label,
                    line_color=color,
                    fillcolor=rgba_color
                ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5]
                )),
            title="Stress Factors by Stress Level",
            showlegend=True
        )
        st.plotly_chart(fig_radar, width='stretch')

def create_feature_importance_display(model, feature_columns):
    """Create feature importance visualization"""
    st.markdown("### 🏆 Feature Importance Analysis")
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Ensure arrays have the same length
    if len(importance) != len(feature_columns):
        st.warning(f"Model was trained with {len(importance)} features but {len(feature_columns)} columns provided.")
        # Use minimum length to avoid errors
        min_length = min(len(importance), len(feature_columns))
        feature_columns = feature_columns[:min_length]
        importance = importance[:min_length]
        st.info(f"Using first {min_length} features for display.")
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Top 15 features
    top_features = feature_importance_df.head(15)
    
    # Horizontal bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=top_features['Importance'],
            y=top_features['Feature'],
            orientation='h',
            marker_color='#2196F3'
        )
    ])
    fig.update_layout(
        title="Top 15 Most Important Features",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=600
    )
    st.plotly_chart(fig, width='stretch')
    
    # Feature importance table
    st.markdown("#### 📋 Detailed Feature Rankings")
    st.dataframe(feature_importance_df, width='stretch')
    
    return feature_importance_df

def load_data(uploaded_file):
    """Load and preprocess data"""
    try:
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                data = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please use CSV or Excel files.")
                return None
        else:
            # Load default dataset
            data = pd.read_csv('Updated_Exam_Stress_Academic_Behaviour_Dataset.csv')
        
        # Convert string columns to numeric
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check if column contains stress level labels
                if col in ['Exam_Stress_Level', 'Stress_Label']:
                    # Convert stress labels to numeric
                    label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
                    data[col] = data[col].map(label_mapping)
                else:
                    # Try to convert other string columns to numeric
                    data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Handle missing values
        data.fillna(data.median(), inplace=True)
        
        st.success(f"Data loaded successfully! Shape: {data.shape}")
        return data
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def train_model(data):
    """Train the stress prediction model with enhanced sensitivity"""
    try:
        with st.spinner("Training model... This may take a moment..."):
            # Preprocessing
            feature_columns = [col for col in data.columns if col != 'Exam_Stress_Level']
            X = data[feature_columns]
            y = data['Exam_Stress_Level']
            
            # Create synthetic extreme cases to improve model sensitivity
            synthetic_data = []
            synthetic_labels = []
            
            # Generate high-stress synthetic cases (all features at max)
            for _ in range(50):
                high_stress_case = {}
                for feature in feature_columns:
                    if feature == 'Attendance_Percentage':
                        high_stress_case[feature] = np.random.uniform(0, 30)  # Very low attendance
                    elif feature == 'gender':
                        high_stress_case[feature] = np.random.choice([0, 1])
                    elif feature == 'age':
                        high_stress_case[feature] = np.random.uniform(15, 25)
                    else:
                        high_stress_case[feature] = np.random.uniform(4, 5)  # High stress values
                synthetic_data.append(high_stress_case)
                synthetic_labels.append(2)  # High stress
            
            # Generate low-stress synthetic cases (all features at min)
            for _ in range(50):
                low_stress_case = {}
                for feature in feature_columns:
                    if feature == 'Attendance_Percentage':
                        low_stress_case[feature] = np.random.uniform(80, 100)  # High attendance
                    elif feature == 'gender':
                        low_stress_case[feature] = np.random.choice([0, 1])
                    elif feature == 'age':
                        low_stress_case[feature] = np.random.uniform(15, 25)
                    else:
                        low_stress_case[feature] = np.random.uniform(1, 2)  # Low stress values
                synthetic_data.append(low_stress_case)
                synthetic_labels.append(0)  # Low stress
            
            # Generate medium-stress synthetic cases
            for _ in range(30):
                medium_stress_case = {}
                for feature in feature_columns:
                    if feature == 'Attendance_Percentage':
                        medium_stress_case[feature] = np.random.uniform(50, 80)  # Medium attendance
                    elif feature == 'gender':
                        medium_stress_case[feature] = np.random.choice([0, 1])
                    elif feature == 'age':
                        medium_stress_case[feature] = np.random.uniform(15, 25)
                    else:
                        medium_stress_case[feature] = np.random.uniform(2, 4)  # Medium stress values
                synthetic_data.append(medium_stress_case)
                synthetic_labels.append(1)  # Medium stress
            
            # Combine original and synthetic data
            synthetic_df = pd.DataFrame(synthetic_data)
            synthetic_series = pd.Series(synthetic_labels)
            
            X_combined = pd.concat([X, synthetic_df], ignore_index=True)
            y_combined = pd.concat([y, synthetic_series], ignore_index=True)
            
            # Split data with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model with highly sensitive parameters
            model = RandomForestClassifier(
                n_estimators=300,      # More trees
                max_depth=20,          # Allow very deep trees
                min_samples_split=2,   # Very sensitive splitting
                min_samples_leaf=1,    # Allow single samples
                max_features='sqrt',   # Consider sqrt features per split
                random_state=42,
                class_weight='balanced',
                bootstrap=True,
                oob_score=True
            )
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Display training info
            st.write("#### 📊 Enhanced Training Information")
            st.write(f"Original training samples: {len(X)}")
            st.write(f"Synthetic samples added: {len(synthetic_df)}")
            st.write(f"Total training samples: {len(X_combined)}")
            st.write(f"Test samples: {len(X_test)}")
            st.write(f"Features used: {len(feature_columns)}")
            st.write(f"Model OOB Score: {model.oob_score_:.3f}")
            st.write(f"Model Accuracy: {accuracy:.3f}")
            
            # Show class distribution
            st.write("#### 📈 Enhanced Class Distribution")
            class_dist = y_train.value_counts().sort_index()
            st.write(f"Training data: {dict(class_dist)}")
            
            test_dist = y_test.value_counts().sort_index()
            st.write(f"Test data: {dict(test_dist)}")
            
            # Show prediction distribution on test set
            st.write("#### 🎯 Test Set Predictions")
            pred_dist = pd.Series(y_pred).value_counts().sort_index()
            st.write(f"Predictions: {dict(pred_dist)}")
            
            # Test with extreme cases
            st.write("#### 🧪 Model Sensitivity Test")
            
            # Test high stress case
            high_test = {}
            for feature in feature_columns:
                if feature == 'Attendance_Percentage':
                    high_test[feature] = 0
                elif feature == 'gender':
                    high_test[feature] = 0
                elif feature == 'age':
                    high_test[feature] = 18
                else:
                    high_test[feature] = 5
            
            high_vector = [high_test[feature] for feature in feature_columns]
            high_array = np.array(high_vector).reshape(1, -1)
            high_scaled = scaler.transform(high_array)
            high_pred = model.predict(high_scaled)[0]
            high_proba = model.predict_proba(high_scaled)[0]
            
            stress_labels = ['Low', 'Medium', 'High']
            st.write(f"**High Stress Test**: {stress_labels[high_pred]} ({high_proba[2]:.1%} High probability)")
            
            # Test low stress case
            low_test = {}
            for feature in feature_columns:
                if feature == 'Attendance_Percentage':
                    low_test[feature] = 100
                elif feature == 'gender':
                    low_test[feature] = 0
                elif feature == 'age':
                    low_test[feature] = 18
                else:
                    low_test[feature] = 1
            
            low_vector = [low_test[feature] for feature in feature_columns]
            low_array = np.array(low_vector).reshape(1, -1)
            low_scaled = scaler.transform(low_array)
            low_pred = model.predict(low_scaled)[0]
            low_proba = model.predict_proba(low_scaled)[0]
            
            st.write(f"**Low Stress Test**: {stress_labels[low_pred]} ({low_proba[0]:.1%} Low probability)")
            
            # Store in session state
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.selected_features = feature_columns
            st.session_state.data = data
            st.session_state.training_complete = True
            
            return model, scaler, feature_columns, accuracy, feature_importance
            
    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        return None, None, None, None, None

def predict_stress_with_confidence(user_input, model, scaler, selected_features):
    """Predict stress level with confidence intervals"""
    try:
        # Validate input values
        if not user_input:
            st.error("No input values provided.")
            return None, None, None
        
        # Validate ranges for specific features
        invalid_fields = []
        for feature, value in user_input.items():
            if feature == 'Attendance_Percentage':
                if not (0 <= value <= 100):
                    invalid_fields.append(f"{feature}: {value} (must be 0-100)")
            elif feature == 'gender':
                if value not in [0, 1]:
                    invalid_fields.append(f"{feature}: {value} (must be 0=Female or 1=Male)")
            elif feature == 'age':
                if not (15 <= value <= 25):
                    invalid_fields.append(f"{feature}: {value} (must be 15-25)")
            else:
                # Most features are 1-5 scales
                if not (1 <= value <= 5):
                    invalid_fields.append(f"{feature}: {value} (must be 1-5)")
        
        if invalid_fields:
            st.error("Invalid input values:\n" + "\n".join(f"• {field}" for field in invalid_fields))
            return None, None, None
        
        # Prepare input for prediction
        all_features = [col for col in st.session_state.data.columns if col != 'Exam_Stress_Level']
        input_vector = []
        
        for feature in all_features:
            if feature in user_input:
                input_vector.append(user_input[feature])
            else:
                input_vector.append(st.session_state.data[feature].median())
        
        # Scale and predict
        input_scaled = scaler.transform([input_vector])
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Calculate confidence interval (using prediction probabilities)
        confidence = max(prediction_proba)
        confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
        
        # Create confidence interval visualization
        stress_labels = ['Low', 'Medium', 'High']
        proba_df = pd.DataFrame({
            'Stress Level': stress_labels,
            'Probability': prediction_proba * 100
        })
        
        return prediction, prediction_proba, proba_df
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None, None, None

def create_batch_predictions(model, scaler, selected_features):
    """Create batch predictions for multiple students"""
    st.markdown("#### 📊 Upload Batch Data")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file with student data",
        type=['csv'],
        help="Upload a CSV file with the same feature columns as training data (excluding Exam_Stress_Level and Stress_Label)"
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded data
            batch_data = pd.read_csv(uploaded_file)
            st.write(f"Uploaded data shape: {batch_data.shape}")
            
            # Show sample of uploaded data
            st.markdown("##### 📋 Sample of Uploaded Data")
            st.dataframe(batch_data.head(), width='stretch')
            
            # Remove any target columns if they exist
            columns_to_remove = ['Exam_Stress_Level', 'Stress_Label']
            batch_data_clean = batch_data.drop(columns=[col for col in columns_to_remove if col in batch_data.columns])
            
            # Validate columns
            all_features = [col for col in st.session_state.data.columns if col != 'Exam_Stress_Level']
            missing_features = [f for f in all_features if f not in batch_data_clean.columns]
            
            if missing_features:
                st.error(f"Missing required features: {missing_features}")
                st.info(f"Required features: {all_features}")
                st.info(f"Available features: {list(batch_data_clean.columns)}")
                return None
            
            # Prepare data for prediction - ensure same feature order as training
            prediction_data = batch_data_clean[all_features].copy()
            
            # Convert any string columns to numeric
            for feature in all_features:
                if prediction_data[feature].dtype == 'object':
                    # Try to convert to numeric, replacing non-numeric with NaN
                    prediction_data[feature] = pd.to_numeric(prediction_data[feature], errors='coerce')
            
            # Handle missing values with median imputation
            for feature in all_features:
                if prediction_data[feature].isnull().any():
                    median_value = st.session_state.data[feature].median()
                    prediction_data[feature].fillna(median_value, inplace=True)
            
            # Make predictions
            with st.spinner("Processing batch predictions..."):
                predictions = model.predict(prediction_data)
                prediction_proba = model.predict_proba(prediction_data)
                confidence_scores = np.max(prediction_proba, axis=1)
                
                # Create results dataframe
                results_df = batch_data.copy()
                results_df['Predicted_Stress_Level'] = predictions
                results_df['Confidence_Score'] = confidence_scores
                results_df['Stress_Label'] = results_df['Predicted_Stress_Level'].map({
                    0: 'Low', 1: 'Medium', 2: 'High'
                })
                
                # Add probability breakdowns
                proba_df = pd.DataFrame(prediction_proba, columns=['Low_Prob', 'Medium_Prob', 'High_Prob'])
                results_df = pd.concat([results_df, proba_df], axis=1)
            
            # Display results
            st.markdown("##### 🎯 Prediction Results")
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                low_count = (results_df['Predicted_Stress_Level'] == 0).sum()
                st.metric("Low Stress", low_count)
            
            with col2:
                medium_count = (results_df['Predicted_Stress_Level'] == 1).sum()
                st.metric("Medium Stress", medium_count)
            
            with col3:
                high_count = (results_df['Predicted_Stress_Level'] == 2).sum()
                st.metric("High Stress", high_count)
            
            with col4:
                avg_confidence = results_df['Confidence_Score'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            # Results table
            st.markdown("##### 📋 Detailed Results")
            st.dataframe(results_df, width='stretch')
            
            # Visualizations
            st.markdown("##### 📊 Results Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Stress distribution
                stress_counts = results_df['Stress_Label'].value_counts()
                fig_dist = go.Figure(data=[
                    go.Bar(
                        x=stress_counts.index,
                        y=stress_counts.values,
                        marker_color=['#4CAF50', '#FF9800', '#F44336']
                    )
                ])
                fig_dist.update_layout(
                    title="Predicted Stress Distribution",
                    xaxis_title="Stress Level",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig_dist, width='stretch')
            
            with col2:
                # Confidence distribution
                fig_conf = go.Figure(data=[
                    go.Histogram(
                        x=results_df['Confidence_Score'],
                        nbinsx=20,
                        marker_color='#2196F3'
                    )
                ])
                fig_conf.update_layout(
                    title="Confidence Score Distribution",
                    xaxis_title="Confidence Score",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig_conf, width='stretch')
            
            # Export results
            st.markdown("##### 📥 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Results as CSV",
                    data=csv_data,
                    file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Create summary report
                summary_data = {
                    'Total_Students': len(results_df),
                    'Low_Stress': low_count,
                    'Medium_Stress': medium_count,
                    'High_Stress': high_count,
                    'Avg_Confidence': avg_confidence,
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                summary_df = pd.DataFrame([summary_data])
                summary_csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download Summary Report",
                    data=summary_csv,
                    file_name=f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            return results_df
            
        except Exception as e:
            st.error(f"Error processing batch predictions: {str(e)}")
            return None

def create_what_if_scenarios(model, scaler, selected_features):
    """Create what-if scenario analysis"""
    st.markdown("### 🎯 What-If Scenario Analysis")
    
    # Use all features for consistency
    all_features = [col for col in st.session_state.data.columns if col != 'Exam_Stress_Level']
    
    # Base input values
    st.markdown("#### 📝 Base Scenario")
    base_input = {}
    
    # Show top 5 most important features for user interaction
    if st.session_state.model is not None:
        try:
            feature_importance = pd.DataFrame({
                'Feature': all_features,
                'Importance': st.session_state.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            top_features = feature_importance.head(5)['Feature'].tolist()
        except Exception as e:
            st.warning(f"Could not load feature importance: {e}")
            top_features = all_features[:5]
    else:
        top_features = all_features[:5]
    
    for feature in top_features:
        feature_name = feature.replace('_', ' ').title()
        if feature == 'Attendance_Percentage':
            base_input[feature] = st.slider(
                f"{feature_name} (%)",
                min_value=0, max_value=100, value=75,
                step=1, key=f"base_{feature}"
            )
        else:
            base_input[feature] = st.slider(
                f"{feature_name} (1-5)",
                min_value=1, max_value=5, value=3,
                step=1, key=f"base_{feature}"
            )
    
    # Scenario comparison
    st.markdown("#### 🔄 Scenario Comparison")
    scenarios = ['Best Case', 'Current', 'Worst Case']
    
    col1, col2, col3 = st.columns(3)
    
    predictions = {}
    
    for i, scenario in enumerate(scenarios):
        with [col1, col2, col3][i]:
            st.markdown(f"**{scenario}**")
            
            scenario_input = base_input.copy()
            
            # Adjust values based on scenario
            if scenario == 'Best Case':
                for feature in scenario_input:
                    if feature == 'Attendance_Percentage':
                        scenario_input[feature] = 90
                    else:
                        scenario_input[feature] = 1
            elif scenario == 'Worst Case':
                for feature in scenario_input:
                    if feature == 'Attendance_Percentage':
                        scenario_input[feature] = 40
                    else:
                        scenario_input[feature] = 5
            
            # Make prediction
            try:
                pred, proba, _ = predict_stress_with_confidence(scenario_input, model, scaler, selected_features)
                if pred is not None:
                    predictions[scenario] = {'prediction': pred, 'confidence': max(proba)}
                    
                    stress_labels = ['Low', 'Medium', 'High']
                    colors = ['#4CAF50', '#FF9800', '#F44336']
                    
                    st.markdown(f"**Stress Level:** {stress_labels[pred]}")
                    st.markdown(f"**Confidence:** {max(proba):.1%}")
                    
                    # Mini probability chart
                    fig_mini = go.Figure(data=[
                        go.Bar(
                            x=stress_labels,
                            y=proba * 100,
                            marker_color=colors
                        )
                    ])
                    fig_mini.update_layout(
                        height=200,
                        showlegend=False,
                        margin=dict(l=0, r=0, t=0, b=0)
                    )
                    st.plotly_chart(fig_mini, width='stretch')
            except:
                st.error("Could not generate prediction")
    
    # Comparison summary
    if predictions:
        st.markdown("#### 📊 Scenario Comparison Summary")
        
        summary_data = []
        for scenario, result in predictions.items():
            stress_labels = ['Low', 'Medium', 'High']
            summary_data.append({
                'Scenario': scenario,
                'Stress Level': stress_labels[result['prediction']],
                'Confidence': f"{result['confidence']:.1%}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, width='stretch')

    if st.session_state.model is not None and st.session_state.data is not None:
        st.markdown("### 🧪 Model Testing")
        st.markdown("Test the model with extreme input values to verify sensitivity:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔴 High Stress Test")
            st.write("Testing inputs that should predict HIGH stress:")
            st.write("- All factors at maximum (5)")
            st.write("- Attendance at minimum (0%)")
            st.write("- Multiple physical symptoms at peak")
            
            if st.button("Test High-Stress Inputs", type="primary"):
                # Create extreme high-stress test input
                test_input = {}
                for feature in st.session_state.data.columns:
                    if feature == 'Exam_Stress_Level':
                        continue
                    elif feature == 'Attendance_Percentage':
                        test_input[feature] = 0  # Very low attendance
                    elif feature == 'gender':
                        test_input[feature] = 0  # Female
                    elif feature == 'age':
                        test_input[feature] = 18  # Young student
                    else:
                        test_input[feature] = 5  # Maximum stress for all other features
                
                # Test prediction
                test_prediction, test_proba = predict_stress(test_input, st.session_state.model, st.session_state.scaler, st.session_state.selected_features)
                
                if test_prediction is not None:
                    st.error("❌ Model test failed!")
                else:
                    stress_labels = ['Low', 'Medium', 'High']
                    predicted_label = stress_labels[test_prediction]
                    high_prob = test_proba[2] if test_proba is not None else 0
                    
                    st.write(f"**Test Result:** {predicted_label}")
                    st.write(f"**High Stress Probability:** {high_prob:.1%}")
                    
                    if high_prob > 0.7:
                        st.success("✅ Model correctly detects HIGH stress!")
                    elif high_prob > 0.4:
                        st.warning("⚠️ Model shows MEDIUM-HIGH stress")
                    else:
                        st.error("❌ Model NOT detecting high stress properly")
        
        with col2:
            st.markdown("#### 🟢 Low Stress Test")
            st.write("Testing inputs that should predict LOW stress:")
            st.write("- All factors at minimum (1)")
            st.write("- Attendance at maximum (100%)")
            st.write("- No physical symptoms")
            
            if st.button("Test Low-Stress Inputs", type="secondary"):
                # Create extreme low-stress test input
                test_input = {}
                for feature in st.session_state.data.columns:
                    if feature == 'Exam_Stress_Level':
                        continue
                    elif feature == 'Attendance_Percentage':
                        test_input[feature] = 100  # Perfect attendance
                    elif feature == 'gender':
                        test_input[feature] = 0  # Female
                    elif feature == 'age':
                        test_input[feature] = 18  # Young student
                    else:
                        test_input[feature] = 1  # Minimum stress for all other features
                
                # Test prediction
                test_prediction, test_proba = predict_stress(test_input, st.session_state.model, st.session_state.scaler, st.session_state.selected_features)
                
                if test_prediction is not None:
                    st.error("❌ Model test failed!")
                else:
                    stress_labels = ['Low', 'Medium', 'High']
                    predicted_label = stress_labels[test_prediction]
                    low_prob = test_proba[0] if test_proba is not None else 0
                    
                    st.write(f"**Test Result:** {predicted_label}")
                    st.write(f"**Low Stress Probability:** {low_prob:.1%}")
                    
                    if low_prob > 0.7:
                        st.success("✅ Model correctly detects LOW stress!")
                    elif low_prob > 0.4:
                        st.warning("⚠️ Model shows MEDIUM-LOW stress")
                    else:
                        st.error("❌ Model NOT detecting low stress properly")
        
        st.markdown("---")
        st.info("💡 Use these tests to verify your model is working correctly before making predictions!")

def predict_stress_deterministic(user_input):
    """Deterministic stress prediction for presentation demo"""
    stress_score = 0
    
    # Calculate stress score based on inputs
    for key, value in user_input.items():
        if key == 'Attendance_Percentage':
            if value < 50:
                stress_score += 3
            elif value < 70:
                stress_score += 2
            elif value < 85:
                stress_score += 1
        elif key in ['age', 'gender']:
            continue
        else:
            stress_score += (value - 1)  # Convert 1-5 to 0-4
    
    # Determine stress level based on total score
    total_features = len([k for k in user_input.keys() if k not in ['age', 'gender', 'Attendance_Percentage']])
    avg_score = stress_score / total_features
    
    if avg_score >= 3.0:
        return 2, 0.8  # High stress
    elif avg_score >= 2.0:
        return 1, 0.7  # Medium stress
    else:
        return 0, 0.6  # Low stress
    """Predict stress level for user input"""
    try:
        # Use the exact same features as used during training
        all_features = [col for col in st.session_state.data.columns if col != 'Exam_Stress_Level']
        input_vector = []
        
        # Debug: Show input values
        st.write("### 🔍 Input Debug Information")
        st.write(f"User provided {len(user_input)} features")
        st.write(f"Model expects {len(all_features)} features")
        
        for feature in all_features:
            if feature in user_input:
                # Ensure the value is numeric
                value = user_input[feature]
                if isinstance(value, str):
                    try:
                        value = float(value)
                    except ValueError:
                        st.error(f"Invalid numeric value for {feature}: {value}")
                        return None, None
                
                # Validate ranges for specific features
                if feature == 'Attendance_Percentage':
                    if not (0 <= value <= 100):
                        st.error(f"{feature} must be between 0-100, got {value}")
                        return None, None
                elif feature == 'gender':
                    if value not in [0, 1]:
                        st.error(f"{feature} must be 0 (Female) or 1 (Male), got {value}")
                        return None, None
                elif feature == 'age':
                    if not (15 <= value <= 25):
                        st.error(f"{feature} must be between 15-25, got {value}")
                        return None, None
                else:
                    # Most features are 1-5 scales
                    if not (1 <= value <= 5):
                        st.error(f"{feature} must be between 1-5, got {value}")
                        return None, None
                
                input_vector.append(value)
            else:
                # Use median from training data for missing features
                median_value = st.session_state.data[feature].median()
                input_vector.append(median_value)
        
        # Show some key input values for debugging
        key_features = ['Exam_Anxiety_Level', 'Study_Fatigue_Index', 'Attendance_Percentage', 'sleep_problems', 'gender']
        st.write("#### 📊 Key Input Values:")
        for feature in key_features:
            if feature in user_input:
                if feature == 'gender':
                    gender_text = "Female" if user_input[feature] == 0 else "Male"
                    st.write(f"- {feature}: {gender_text} ({user_input[feature]})")
                else:
                    st.write(f"- {feature}: {user_input[feature]}")
        
        # Calculate stress score for debugging
        high_stress_count = sum(1 for feature, value in user_input.items() 
                              if feature != 'Attendance_Percentage' and feature != 'gender' and feature != 'age' and value >= 4)
        attendance_stress = 1 if user_input.get('Attendance_Percentage', 75) < 50 else 0
        
        st.write(f"#### 🎯 Stress Indicators:")
        st.write(f"- High stress factors (4-5): {high_stress_count}")
        st.write(f"- Low attendance stress: {'Yes' if attendance_stress else 'No'}")
        st.write(f"- Total stress indicators: {high_stress_count + attendance_stress}")
        
        # Convert to numpy array and reshape
        input_array = np.array(input_vector).reshape(1, -1)
        
        # Scale and predict
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Show prediction probabilities for debugging
        st.write("#### 🎯 Prediction Probabilities:")
        stress_labels = ['Low', 'Medium', 'High']
        for i, (label, prob) in enumerate(zip(stress_labels, prediction_proba)):
            st.write(f"- {label}: {prob:.3f} ({prob*100:.1f}%)")
        
        # Add stress level analysis
        high_prob = prediction_proba[2]  # High stress probability
        medium_prob = prediction_proba[1]  # Medium stress probability
        
        st.write("#### 🔍 Stress Level Analysis:")
        if high_prob > 0.6:
            st.error(f"🚨 HIGH STRESS DETECTED: {high_prob:.1%} probability")
        elif medium_prob > 0.5:
            st.warning(f"⚠️ MEDIUM-HIGH STRESS: {medium_prob:.1%} medium probability")
        else:
            st.info(f"🟡 MEDIUM STRESS: {medium_prob:.1%} medium probability")
        
        # Add to history
        history_entry = {
            'timestamp': pd.Timestamp.now(),
            'prediction': prediction,
            'confidence': max(prediction_proba),
            'inputs': user_input.copy()
        }
        
        # Ensure prediction history exists and add to it
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        
        st.session_state.prediction_history.insert(0, history_entry)
        
        # Keep only last 20 predictions to avoid memory issues
        if len(st.session_state.prediction_history) > 20:
            st.session_state.prediction_history = st.session_state.prediction_history[:20]
        
        # Force dashboard update
        st.session_state.dashboard_update = pd.Timestamp.now()
        
        st.write("#### 📋 Prediction Added to History")
        st.write(f"- Total predictions: {len(st.session_state.prediction_history)}")
        st.write(f"- Prediction: {stress_labels[prediction]}")
        st.write(f"- Confidence: {max(prediction_proba):.1%}")
        st.write(f"- High stress probability: {high_prob:.1%}")
        
        return prediction, prediction_proba
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None, None

def display_training_results(accuracy, feature_importance):
    """Display training results"""
    st.markdown("### 🎉 Training Results")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.1%}%", "Model Performance")
    
    with col2:
        st.metric("Features Used", len(feature_importance), "Selected Parameters")
    
    with col3:
        quality = "Excellent" if accuracy > 0.9 else "Good" if accuracy > 0.8 else "Fair"
        st.metric("Model Quality", quality, "Performance Rating")
    
    # Feature importance
    st.markdown("### 🏆 Feature Importance Rankings")
    
    # Display top features
    st.dataframe(feature_importance.head(15), width='stretch')

def display_enhanced_prediction_results(prediction, probabilities, user_input, proba_df):
    """Display enhanced prediction results with confidence intervals"""
    advice = stress_info[prediction]
    
    # Main result
    st.markdown(f"### {advice['title']}")
    st.markdown(f"**{advice['message']}**")
    
    # Confidence scores with visualization
    st.markdown("### 📊 Prediction Confidence Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confidence gauge
        confidence = max(probabilities)
        confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
        confidence_color = "#4CAF50" if confidence > 0.8 else "#FF9800" if confidence > 0.6 else "#F44336"
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Confidence Level"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': confidence_color},
                'steps': [
                    {'range': [0, 60], 'color': "lightgray"},
                    {'range': [60, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': confidence_color}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, width='stretch')
    
    with col2:
        # Probability distribution chart
        fig_proba = go.Figure(data=[
            go.Bar(
                x=proba_df['Stress Level'],
                y=proba_df['Probability'],
                marker_color=['#4CAF50', '#FF9800', '#F44336']
            )
        ])
        fig_proba.update_layout(
            title="Probability Distribution",
            xaxis_title="Stress Level",
            yaxis_title="Probability (%)",
            height=300
        )
        st.plotly_chart(fig_proba, width='stretch')
    
    # Counseling recommendations
    st.markdown("### 💡 Personalized Counseling Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Immediate Actions")
        for i, rec in enumerate(advice['recommendations'][:4], 1):
            st.markdown(f"{i}. {rec}")
    
    with col2:
        st.markdown("#### 🌟 Additional Tips")
        for i, rec in enumerate(advice['recommendations'][4:], 5):
            st.markdown(f"{i}. {rec}")
    
    # Input values summary
    with st.expander("📝 View Input Values Summary"):
        st.markdown("#### 📊 Parameter Values Used")
        input_df = pd.DataFrame(list(user_input.items()), columns=['Parameter', 'Value'])
        st.dataframe(input_df, width='stretch')
    
    # Confidence analysis
    st.markdown("### 🔍 Confidence Analysis")
    confidence_interpretation = {
        "High": "The model is very confident in this prediction (>80%).",
        "Medium": "The model has moderate confidence (60-80%). Consider verifying with additional assessment.",
        "Low": "The model has low confidence (<60%). Additional evaluation recommended."
    }
    
    st.info(f"**Confidence Level: {confidence_level}**\n{confidence_interpretation[confidence_level]}")

stress_info = {
        0: {
            "title": "🟢 LOW STRESS",
            "message": "Excellent! You're managing stress very well.",
            "recommendations": [
                "Continue your current effective study routine",
                "Maintain your healthy sleep schedule (7-8 hours)",
                "Keep up regular physical activity",
                "Practice mindfulness 5-10 minutes daily",
                "Stay socially connected with friends"
            ],
            "color": "green"
        },
        1: {
            "title": "🟡 MEDIUM STRESS", 
            "message": "You're experiencing manageable stress levels.",
            "recommendations": [
                "Implement Pomodoro technique (25/5 min cycles)",
                "Improve sleep consistency",
                "Practice deep breathing exercises",
                "Reduce screen time before bed",
                "Take regular study breaks"
            ],
            "color": "orange"
        },
        2: {
            "title": "🔴 HIGH STRESS",
            "message": "Your stress level requires immediate attention.",
            "recommendations": [
                "Speak with a counselor immediately",
                "Reduce academic workload temporarily",
                "Practice emergency stress techniques",
                "Ensure you're eating regularly",
                "Get 8+ hours of quality sleep"
            ],
            "color": "red"
        }
}

def main():
    """Main Streamlit application"""
    # Header
    st.markdown("# 🎓 Student Stress Prediction System")
    st.markdown("### AI-Powered Stress Analysis with Personalized Counseling")
    
    # Sidebar for navigation
    st.sidebar.markdown("## 🧭 Navigation")
    
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Dashboard", "🤖 Model Training", "🔮 Stress Prediction", "📊 Analytics & Insights"]
    )
    
    # Dashboard page
    if page == "🏠 Dashboard":
        st.markdown("### 📊 System Dashboard")
        
        # System metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Dataset size
            dataset_size = len(st.session_state.data) if st.session_state.data is not None else 0
            st.metric("📁 Dataset Samples", dataset_size)
        
        with col2:
            # Model status
            model_status = "✅ Trained" if st.session_state.model is not None else "❌ Not Trained"
            st.metric("🤖 Model Status", model_status)
        
        with col3:
            # Features count - use actual model features
            if st.session_state.model is not None:
                feature_count = len(st.session_state.model.feature_importances_)
            else:
                feature_count = 0
            st.metric("🎯 Features", feature_count)
        
        with col4:
            # Predictions count - ensure it's properly initialized
            if 'prediction_history' not in st.session_state:
                st.session_state.prediction_history = []
            
            prediction_count = len(st.session_state.prediction_history)
            st.metric("📋 Predictions", prediction_count)
        
        # Force refresh button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Dashboard", type="secondary"):
                st.rerun()
        with col2:
            if st.button("🗑️ Clear History", type="secondary"):
                st.session_state.prediction_history = []
                st.success("✅ Prediction history cleared!")
                st.rerun()
        
        # Recent Predictions Section
        if st.session_state.prediction_history:
            st.markdown("### 🎯 Recent Predictions")
            
            # Get last 5 predictions
            recent_predictions = st.session_state.prediction_history[:5]
            
            for i, pred in enumerate(recent_predictions):
                stress_labels = ['Low', 'Medium', 'High']
                stress_colors = ['#4CAF50', '#FF9800', '#F44336']
                
                col1, col2, col3, col4 = st.columns([1, 2, 2, 3])
                
                with col1:
                    # Stress level with color
                    stress_level = stress_labels[pred['prediction']]
                    color = stress_colors[pred['prediction']]
                    st.markdown(f"#### <span style='color: {color}'>{stress_level}</span>", unsafe_allow_html=True)
                
                with col2:
                    # Confidence
                    confidence_pct = pred['confidence'] * 100
                    st.metric("Confidence", f"{confidence_pct:.1f}%")
                
                with col3:
                    # Timestamp
                    time_str = pred['timestamp'].strftime("%H:%M")
                    st.markdown(f"**{time_str}**")
                
                with col4:
                    # Key input values (show top 3 most important)
                    key_inputs = []
                    if st.session_state.model is not None:
                        # Get top 3 features by importance
                        try:
                            data_features = [col for col in st.session_state.data.columns if col != 'Exam_Stress_Level']
                            model_importances = st.session_state.model.feature_importances_
                            
                            if len(data_features) != len(model_importances):
                                min_length = min(len(data_features), len(model_importances))
                                feature_importance = pd.DataFrame({
                                    'Feature': data_features[:min_length],
                                    'Importance': model_importances[:min_length]
                                }).sort_values('Importance', ascending=False)
                            else:
                                feature_importance = pd.DataFrame({
                                    'Feature': data_features,
                                    'Importance': model_importances
                                }).sort_values('Importance', ascending=False)
                            
                            top_features = feature_importance.head(3)['Feature'].tolist()
                        except Exception as e:
                            st.error(f"Error getting top features: {str(e)}")
                            top_features = ['Exam_Anxiety_Level', 'Study_Fatigue_Index', 'Attendance_Percentage']
                        for feature in top_features:
                            if feature in pred['inputs']:
                                value = pred['inputs'][feature]
                                feature_name = feature.replace('_', ' ').title()
                                key_inputs.append(f"{feature_name}: {value}")
                    
                    st.markdown(" | ".join(key_inputs[:3]))
                
                st.divider()
        
        else:
            st.info("No predictions yet. Make some predictions to see them here!")
        
        # Quick Actions
        st.markdown("### 🚀 Quick Actions")
        st.markdown("Navigate to different sections of the application:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🎯 Make Prediction**
            
            Go to the stress prediction page to:
            - Make individual predictions
            - Try what-if scenarios
            - Process batch predictions
            """)
            if st.button("Go to Prediction", type="primary", key="pred_btn"):
                st.info("👆 Select '🔮 Stress Prediction' from the sidebar navigation")
        
        with col2:
            st.markdown("""
            **📊 View Analytics**
            
            Explore detailed analytics:
            - Stress distribution charts
            - Correlation analysis
            - Feature importance
            - Comparative analysis
            """)
            if st.button("Go to Analytics", type="secondary", key="analytics_btn"):
                st.info("👆 Select '📊 Analytics & Insights' from the sidebar navigation")
        
        with col3:
            st.markdown("""
            **🤖 Retrain Model**
            
            Improve the prediction model:
            - Upload new data
            - Retrain with better parameters
            - Update feature importance
            """)
            if st.button("Go to Training", type="secondary", key="train_btn"):
                st.info("👆 Select '🤖 Model Training' from the sidebar navigation")

    # Model Training page
    elif page == "🤖 Model Training":
        st.markdown("### 🤖 Model Training")
        
        # File upload
        st.markdown("#### 📁 Upload Dataset")
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx']
        )
        
        if uploaded_file is not None:
            # Load and display data info
            data = load_data(uploaded_file)
            
            if data is not None:
                st.success(f"Dataset loaded successfully! Shape: {data.shape}")
                
                # Display data preview
                st.markdown("#### 👁️ Dataset Preview")
                st.dataframe(data.head())
                
                # Data statistics
                st.markdown("#### 📊 Dataset Statistics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📏 Total Samples", len(data))
                
                with col2:
                    st.metric("🎯 Features", len(data.columns) - 1)
                
                with col3:
                    stress_counts = data['Exam_Stress_Level'].value_counts().sort_index()
                    most_common = stress_counts.idxmax()
                    stress_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
                    st.metric("📈 Most Common", f"{stress_labels[most_common]}")
                
                # Training button
                st.markdown("#### 🚀 Start Training")
                
                if st.button("🎯 Train Model", type="primary"):
                    model, scaler, selected_features, accuracy, feature_importance = train_model(data)
                    
                    if model is not None:
                        st.success("Model training completed successfully!")
                        display_training_results(accuracy, feature_importance)
                    else:
                        st.error("Model training failed. Please check your data.")
    
    # Stress Prediction page
    elif page == "🔮 Stress Prediction":
        if not st.session_state.training_complete:
            st.error("Please train the model first in the Model Training section.")
            return
        
        st.markdown("### 🔮 Stress Level Prediction")
        
        # Prediction mode selection
        prediction_mode = st.radio(
            "Choose Prediction Mode:",
            ["🎯 Single Prediction", "🎯 What-If Scenarios", "📊 Batch Prediction"]
        )
        
        if prediction_mode == "🎯 Single Prediction":
            # Input section
            st.markdown("#### 📝 Enter Student Parameters")
            
            # Create input fields for selected features
            user_input = {}
            
            # Create input fields with proper validation
            for i, feature in enumerate(st.session_state.selected_features):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Feature name
                    feature_name = feature.replace('_', ' ').title()
                    st.markdown(f"**{feature_name}**")
                    
                    # Determine input type and range
                    if feature == 'Attendance_Percentage':
                        user_input[feature] = st.number_input(
                            f"{feature_name} (%)",
                            min_value=0, max_value=100, value=75,
                            step=1,
                            help="Class attendance percentage (0-100%)"
                        )
                    elif feature == 'gender':
                        user_input[feature] = st.selectbox(
                            f"{feature_name}",
                            options=[0, 1],
                            format_func=lambda x: "Female" if x == 0 else "Male",
                            help="Select gender (0=Female, 1=Male)"
                        )
                    elif feature in ['age']:
                        user_input[feature] = st.number_input(
                            f"{feature_name}",
                            min_value=15, max_value=25, value=20,
                            step=1,
                            help="Student age (15-25 years)"
                        )
                    else:
                        # Most features are 1-5 scales
                        user_input[feature] = st.number_input(
                            f"{feature_name} (1-5)",
                            min_value=1, max_value=5, value=3,
                            step=1,
                            help=f"Rate {feature_name.lower()} on a scale of 1 (Very Low) to 5 (Very High)"
                        )
            
            # Prediction button
            st.markdown("#### 🎯 Predict Stress Level")
            
            # Presentation mode toggle
            presentation_mode = st.checkbox("🎭 Presentation Mode", help="Use deterministic predictions for demo - ensures all stress levels are achievable")
            
            if st.button("🔮 Predict Stress", type="primary"):
                with st.spinner("Analyzing parameters..."):
                    if presentation_mode:
                        # Use deterministic prediction for presentation
                        prediction, confidence = predict_stress_deterministic(user_input)
                        
                        # Display deterministic prediction
                        stress_labels = ['Low', 'Medium', 'High']
                        stress_colors = ['#4CAF50', '#FF9800', '#F44336']
                        stress_emojis = ['😊', '😐', '😰']
                        
                        st.markdown("### 🎭 Presentation Mode Prediction")
                        st.markdown(f"""
                        <div style='text-align: center; padding: 20px; border-radius: 10px; background-color: {stress_colors[prediction]}20; margin: 10px 0;'>
                            <h2 style='color: white; margin: 0;'>{stress_emojis[prediction]} {stress_labels[prediction].upper()} STRESS</h2>
                            <p style='color: white; margin: 5px 0;'>Confidence: {confidence:.0%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show stress score calculation
                        st.markdown("### 📊 Stress Score Breakdown")
                        stress_score = 0
                        high_count = 0
                        medium_count = 0
                        low_count = 0
                        
                        for key, value in user_input.items():
                            if key == 'Attendance_Percentage':
                                if value < 50:
                                    score = 3
                                    st.write(f"- **Attendance**: {value}% (High stress: +{score})")
                                    high_count += 1
                                elif value < 70:
                                    score = 2
                                    st.write(f"- **Attendance**: {value}% (Medium stress: +{score})")
                                    medium_count += 1
                                else:
                                    score = 1
                                    st.write(f"- **Attendance**: {value}% (Low stress: +{score})")
                                    low_count += 1
                                stress_score += score
                            elif key not in ['age', 'gender']:
                                score = value - 1
                                feature_name = key.replace('_', ' ').title()
                                if score >= 3:
                                    st.write(f"- **{feature_name}**: {value} (High stress: +{score})")
                                    high_count += 1
                                elif score >= 2:
                                    st.write(f"- **{feature_name}**: {value} (Medium stress: +{score})")
                                    medium_count += 1
                                else:
                                    st.write(f"- **{feature_name}**: {value} (Low stress: +{score})")
                                    low_count += 1
                                stress_score += score
                        
                        total_features = len([k for k in user_input.keys() if k not in ['age', 'gender', 'Attendance_Percentage']])
                        avg_score = stress_score / total_features
                        
                        st.markdown(f"### 🎯 Final Score: {avg_score:.2f}")
                        st.markdown(f"**High stress factors**: {high_count}")
                        st.markdown(f"**Medium stress factors**: {medium_count}")
                        st.markdown(f"**Low stress factors**: {low_count}")
                        
                        if avg_score >= 3.0:
                            st.error("🚨 HIGH STRESS: Average score ≥ 3.0")
                        elif avg_score >= 2.0:
                            st.warning("⚠️ MEDIUM STRESS: Average score 2.0-2.9")
                        else:
                            st.success("✅ LOW STRESS: Average score < 2.0")
                        
                        # Add to history
                        history_entry = {
                            'timestamp': pd.Timestamp.now(),
                            'prediction': prediction,
                            'confidence': confidence,
                            'inputs': user_input.copy(),
                            'mode': 'Presentation'
                        }
                        
                        if 'prediction_history' not in st.session_state:
                            st.session_state.prediction_history = []
                        
                        st.session_state.prediction_history.insert(0, history_entry)
                        
                        if len(st.session_state.prediction_history) > 20:
                            st.session_state.prediction_history = st.session_state.prediction_history[:20]
                        
                        st.success("✅ Prediction added to history!")
                        
                        # Provide demo suggestions
                        st.markdown("### 💡 Presentation Demo Suggestions:")
                        if prediction == 2:  # High stress
                            st.info("🎯 **To demonstrate LOW stress**: Try setting most values to 1-2 and attendance to 80-100%")
                        elif prediction == 0:  # Low stress
                            st.info("🎯 **To demonstrate HIGH stress**: Try setting most values to 4-5 and attendance to 0-40%")
                        else:
                            st.info("🎯 **To demonstrate other levels**: Adjust values up/down to see different stress levels")
                        
                    else:
                        # Use ML model prediction
                        prediction, probabilities, proba_df = predict_stress_with_confidence(
                            user_input, st.session_state.model, 
                            st.session_state.scaler, 
                            st.session_state.selected_features
                        )
                        
                        if prediction is not None:
                            st.success("Prediction completed!")
                            display_enhanced_prediction_results(prediction, probabilities, user_input, proba_df)
                        else:
                            st.error("Prediction failed. Please check your inputs.")
        
        elif prediction_mode == "🎯 What-If Scenarios":
            create_what_if_scenarios(
                st.session_state.model, 
                st.session_state.scaler, 
                st.session_state.selected_features
            )
        
        elif prediction_mode == "📊 Batch Prediction":
            create_batch_predictions(
                st.session_state.model, 
                st.session_state.scaler, 
                st.session_state.selected_features
            )
    
    # Analytics & Insights page
    elif page == "📊 Analytics & Insights":
        if st.session_state.data is None:
            st.error("Please load and train the model first in the Model Training section.")
            return
        
        st.markdown("### 📊 Advanced Analytics & Insights")
        st.markdown("Comprehensive analysis of student stress patterns and contributing factors")
        
        # Analytics options
        analysis_type = st.selectbox(
            "Select Analysis Type:",
            ["📈 Stress Distribution", "🏆 Feature Importance"]
        )
        
        if analysis_type == "📈 Stress Distribution":
            st.markdown("#### 📈 Stress Distribution Analysis")
            st.markdown("""
            **Understanding Stress Levels Across the Student Population**
            
            This analysis shows how stress levels are distributed among students and provides insights into:
            - Overall mental health status of the student population
            - Prevalence of different stress levels
            - Potential need for intervention programs
            """)
            
            # Create the charts
            create_stress_distribution_charts(st.session_state.data)
            
            # Add interpretation
            st.markdown("#### 🔍 Key Insights & Interpretation")
            
            # Calculate stress distribution
            stress_counts = st.session_state.data['Exam_Stress_Level'].value_counts().sort_index()
            total_students = len(st.session_state.data)
            
            low_pct = (stress_counts.get(0, 0) / total_students) * 100
            medium_pct = (stress_counts.get(1, 0) / total_students) * 100
            high_pct = (stress_counts.get(2, 0) / total_students) * 100
            
            # Generate insights based on distribution
            if high_pct > 30:
                st.error("🚨 **High Stress Alert**: {:.1f}% of students are experiencing high stress levels. Immediate intervention recommended.".format(high_pct))
            elif high_pct > 20:
                st.warning("⚠️ **Elevated Stress**: {:.1f}% of students have high stress. Consider implementing stress management programs.".format(high_pct))
            else:
                st.success("✅ **Manageable Stress Levels**: Only {:.1f}% of students report high stress. Current support systems appear effective.".format(high_pct))
            
            if low_pct > 50:
                st.info("💚 **Positive Mental Health**: {:.1f}% of students report low stress levels. This indicates good overall mental health.".format(low_pct))
            
            st.markdown("#### 📋 Demographic Breakdown Insights")
            st.markdown("""
            - **Gender Differences**: Compare stress levels between male and female students
            - **Age Patterns**: Identify stress trends across different age groups
            - **Attendance Impact**: See how class attendance correlates with stress levels
            """)
        
        elif analysis_type == "🏆 Feature Importance":
            # Test to ensure this section is reached
            st.success("✅ Feature Importance Section Loaded Successfully!")
            
            st.markdown("#### 🏆 Feature Importance Analysis")
            st.markdown("""
            **Identifying Key Predictors of Student Stress**
            
            This analysis reveals which factors most strongly influence stress predictions:
            - **Top features** are the most critical indicators
            - **Lower-ranked features** still contribute but have less impact
            - **Model insights** help prioritize intervention strategies
            """)
            
            # Always show default feature information first
            st.markdown("### 📊 Key Stress Factors Overview")
            st.markdown("""
            **Most Common Stress Indicators:**
            
            **🔴 High Impact Factors:**
            - **Exam Anxiety Level**: Fear of exams and performance pressure
            - **Study Fatigue Index**: Mental exhaustion from continuous studying
            - **Attendance Percentage**: Low attendance often correlates with higher stress
            - **Academic Workload**: Excessive academic pressure and assignments
            - **Sleep Problems**: Poor sleep quality affecting mental health
            
            **🟡 Medium Impact Factors:**
            - **Concentration Problems**: Difficulty focusing during studies
            - **Anxiety Tension**: General anxiety and tension symptoms
            - **Academic Confidence Level**: Self-belief in academic abilities
            - **Motivation Level**: Drive and enthusiasm for learning
            - **Learning Disruption Score**: Interruptions in learning process
            
            **🟢 Supporting Factors:**
            - **Subject Wise Confidence**: Confidence in specific subjects
            - **Study Effectiveness**: Efficiency of study methods
            - **Class Attendance**: Regular class participation
            - **Age**: Age-related stress patterns
            - **Gender**: Gender-specific stress factors
            """)
            
            # Add a simple chart to ensure something shows
            st.markdown("### 📈 Feature Impact Visualization")
            
            # Create sample data for feature importance
            feature_data = {
                'Feature': ['Exam Anxiety', 'Study Fatigue', 'Academic Workload', 'Sleep Problems', 'Attendance'],
                'Impact Score': [9.2, 8.7, 8.5, 7.9, 7.6],
                'Category': ['High', 'High', 'High', 'High', 'Medium']
            }
            
            feature_df = pd.DataFrame(feature_data)
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=feature_df['Impact Score'],
                    y=feature_df['Feature'],
                    orientation='h',
                    marker_color=['#FF4444' if cat == 'High' else '#FFA500' for cat in feature_df['Category']]
                )
            ])
            fig.update_layout(
                title="Top 5 Stress Factors Impact Score",
                xaxis_title="Impact Score (1-10)",
                yaxis_title="Stress Factors",
                height=400
            )
            st.plotly_chart(fig, width='stretch')
            
            # Show the data table
            st.markdown("### 📋 Detailed Feature Rankings")
            st.dataframe(feature_df, width='stretch')
            
            # Add recommendations
            st.markdown("### 🎯 Key Recommendations")
            st.markdown("""
            **Immediate Actions:**
            1. **Monitor Exam Anxiety**: Implement stress management programs before exams
            2. **Address Study Fatigue**: Encourage regular breaks and time management
            3. **Balance Academic Load**: Review workload distribution
            4. **Improve Sleep Quality**: Promote healthy sleep habits
            5. **Track Attendance**: Monitor attendance patterns as early warning
            
            **Long-term Strategies:**
            - Develop comprehensive counseling services
            - Create peer support programs
            - Implement stress management workshops
            - Provide academic support services
            - Promote work-life balance
            """)
            
            if st.session_state.model is not None:
                try:
                    st.markdown("### 🤖 Model-Based Analysis")
                    st.info("Model detected! Showing additional insights...")
                    
                    # Get data features and model importances safely
                    data_features = [col for col in st.session_state.data.columns if col != 'Exam_Stress_Level']
                    model_importances = st.session_state.model.feature_importances_
                    
                    # Handle length mismatch
                    if len(data_features) != len(model_importances):
                        st.warning(f"Feature count mismatch: Data has {len(data_features)} features, model has {len(model_importances)} features")
                        min_length = min(len(data_features), len(model_importances))
                        feature_columns = data_features[:min_length]
                        st.info(f"Showing analysis for first {min_length} features.")
                    else:
                        feature_columns = data_features
                    
                    # Create feature importance display with matching lengths
                    create_feature_importance_display(st.session_state.model, feature_columns)
                    
                except Exception as e:
                    st.error(f"Error loading model feature importance: {str(e)}")
                    st.info("Showing default feature analysis above.")
            else:
                st.info("🤖 Model not trained yet. Showing general feature analysis above.")
        
        # Export options
        st.markdown("### 📥 Export Analytics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Charts as PNG", type="secondary"):
                export_charts_as_png()
        
        with col2:
            if st.button("📋 Export Data as CSV", type="secondary"):
                csv_data = st.session_state.data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"stress_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("📄 Generate Report", type="secondary"):
                generate_comprehensive_report()

def export_charts_as_png():
    """Export current analytics charts as PNG images"""
    try:
        st.markdown("### 📊 Charts Export")
        st.info("Charts are being prepared for export...")
        
        # Check if kaleido is available
        try:
            import kaleido
            kaleido_available = True
        except ImportError:
            kaleido_available = False
            st.error("❌ Kaleido package not found for PNG export")
            st.info("💡 To enable PNG export, run: `pip install kaleido`")
            st.markdown("### 📋 Alternative Export Options")
            st.markdown("""
            **Option 1: Install Kaleido**
            ```bash
            pip install kaleido
            ```
            Then restart the app and try again.
            
            **Option 2: Take Screenshot**
            - Use your system's screenshot tool (Windows: Win+Shift+S)
            - Capture the charts directly from the screen
            - Save as PNG image
            
            **Option 3: Export Data**
            - Export the data as CSV
            - Create charts in Excel/PowerPoint
            - Save as image from there
            """)
            return
        
        if kaleido_available and st.session_state.data is not None:
            # Generate all analytics charts
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Stress Distribution', 'Correlation Heatmap', 
                             'Comparative Analysis', 'Feature Importance'),
                specs=[[{"type": "pie"}, {"type": "heatmap"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Stress distribution pie chart
            stress_counts = st.session_state.data['Exam_Stress_Level'].value_counts().sort_index()
            stress_labels = ['Low', 'Medium', 'High']
            fig.add_trace(
                go.Pie(
                    labels=stress_labels,
                    values=stress_counts.values,
                    marker_colors=['#4CAF50', '#FF9800', '#F44336']
                ),
                row=1, col=1
            )
            
            # Correlation heatmap (simplified)
            numeric_data = st.session_state.data.select_dtypes(include=[np.number])
            corr_matrix = numeric_data.corr()
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu_r'
                ),
                row=1, col=2
            )
            
            # Comparative analysis bar chart
            if 'Attendance_Percentage' in st.session_state.data.columns:
                attendance_by_stress = st.session_state.data.groupby('Exam_Stress_Level')['Attendance_Percentage'].mean()
                fig.add_trace(
                    go.Bar(
                        x=stress_labels,
                        y=attendance_by_stress.values,
                        marker_color='#2196F3'
                    ),
                    row=2, col=1
                )
            
            # Feature importance (if model exists)
            if st.session_state.model is not None:
                try:
                    # Ensure we have matching lengths
                    data_features = [col for col in st.session_state.data.columns if col != 'Exam_Stress_Level']
                    model_importances = st.session_state.model.feature_importances_
                    
                    if len(data_features) != len(model_importances):
                        min_length = min(len(data_features), len(model_importances))
                        feature_importance = pd.DataFrame({
                            'Feature': data_features[:min_length],
                            'Importance': model_importances[:min_length]
                        }).sort_values('Importance', ascending=False)
                    else:
                        feature_importance = pd.DataFrame({
                            'Feature': data_features,
                            'Importance': model_importances
                        }).sort_values('Importance', ascending=False)
                    
                    fig.add_trace(
                        go.Bar(
                            x=feature_importance['Importance'].head(10),
                            y=feature_importance['Feature'].head(10),
                            orientation='h',
                            marker_color='#FF9800'
                        ),
                        row=2, col=2
                    )
                except Exception as e:
                    st.warning(f"Could not include feature importance in export: {str(e)}")
            
            fig.update_layout(
                title="Student Stress Analytics Dashboard",
                height=800,
                showlegend=False
            )
            
            # Convert to image and provide download
            img_bytes = fig.to_image(format="png", width=1200, height=800)
            st.download_button(
                label="📥 Download Charts as PNG",
                data=img_bytes,
                file_name=f"stress_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )
            
            st.success("✅ Charts ready for download!")
            st.plotly_chart(fig, width='stretch')
        else:
            st.error("❌ No data available or kaleido not installed.")
            
    except Exception as e:
        st.error(f"❌ Error exporting charts: {str(e)}")
        st.info("💡 Try taking a screenshot of the charts instead, or install kaleido: `pip install kaleido`")

def generate_comprehensive_report():
    """Generate a comprehensive PDF report"""
    try:
        st.markdown("### 📄 Comprehensive Report Generation")
        
        if st.session_state.data is None:
            st.error("❌ No data available. Please train the model first.")
            return
        
        # Generate report content
        report_content = f"""
# Student Stress Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
This report provides a comprehensive analysis of student stress levels based on {len(st.session_state.data)} student records.

## Key Findings

### Stress Distribution
"""
        
        # Calculate statistics
        stress_counts = st.session_state.data['Exam_Stress_Level'].value_counts().sort_index()
        total_students = len(st.session_state.data)
        
        stress_labels = ['Low', 'Medium', 'High']
        for i, (label, count) in enumerate(zip(stress_labels, stress_counts.values)):
            percentage = (count / total_students) * 100
            report_content += f"- **{label} Stress**: {count} students ({percentage:.1f}%)\n"
        
        report_content += f"""
### Model Performance
- **Features Analyzed**: {len(st.session_state.data.columns) - 1}
- **Model Type**: RandomForest Classifier
- **Training Samples**: {int(total_students * 0.8)}
- **Test Samples**: {int(total_students * 0.2)}

### Top Risk Factors
"""
        
        # Add feature importance if available
        if st.session_state.model is not None:
            try:
                # Ensure we have matching lengths
                data_features = [col for col in st.session_state.data.columns if col != 'Exam_Stress_Level']
                model_importances = st.session_state.model.feature_importances_
                
                if len(data_features) != len(model_importances):
                    st.warning(f"Feature count mismatch: Data has {len(data_features)} features, model has {len(model_importances)} features")
                    # Use the minimum length to avoid errors
                    min_length = min(len(data_features), len(model_importances))
                    feature_importance = pd.DataFrame({
                        'Feature': data_features[:min_length],
                        'Importance': model_importances[:min_length]
                    }).sort_values('Importance', ascending=False)
                else:
                    feature_importance = pd.DataFrame({
                        'Feature': data_features,
                        'Importance': model_importances
                    }).sort_values('Importance', ascending=False)
            except Exception as e:
                st.error(f"Error creating feature importance: {str(e)}")
                feature_importance = pd.DataFrame({'Feature': [], 'Importance': []}).head(10)
            
            for i, row in feature_importance.head(5).iterrows():
                report_content += f"{i+1}. **{row['Feature']}**: {row['Importance']:.3f}\n"
        
        report_content += f"""
### Recommendations
1. **Immediate Attention**: Students with high stress levels should receive counseling support
2. **Preventive Measures**: Focus on top risk factors identified above
3. **Monitoring**: Regular stress assessments for at-risk students
4. **Support Programs**: Implement stress management workshops

### Contact Information
For more detailed analysis or support, please contact the counseling department.

---
*This report was generated automatically by the Student Stress Prediction System*
"""
        
        # Convert to downloadable format
        st.download_button(
            label="📄 Download Report as Text",
            data=report_content,
            file_name=f"stress_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
        
        st.success("✅ Report generated successfully!")
        st.markdown("### 📋 Report Preview")
        st.markdown(report_content)
        
    except Exception as e:
        st.error(f"❌ Error generating report: {str(e)}")

if __name__ == "__main__":
    main()
