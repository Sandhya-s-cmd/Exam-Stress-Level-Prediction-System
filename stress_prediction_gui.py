import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import joblib
import threading
import warnings
warnings.filterwarnings('ignore')

class StressPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Stress Level Prediction System")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.model = None
        self.scaler = None
        self.selected_features = []
        self.data = None
        self.feature_entries = {}
        
        # Counseling recommendations
        self.counseling_advice = {
            0: {
                "title": "LOW STRESS",
                "message": "You are managing your academic life well.",
                "recommendations": [
                    "Maintain your study routine and healthy lifestyle.",
                    "Continue balancing study and relaxation."
                ],
                "color": "#4CAF50"  # Green
            },
            1: {
                "title": "MEDIUM STRESS",
                "message": "You may be experiencing moderate stress.",
                "recommendations": [
                    "Improve sleep schedule",
                    "Take short study breaks",
                    "Practice time management",
                    "Reduce screen time"
                ],
                "color": "#FF9800"  # Orange
            },
            2: {
                "title": "HIGH STRESS",
                "message": "Your stress level is high.",
                "recommendations": [
                    "Take regular breaks",
                    "Practice meditation or yoga",
                    "Improve sleep habits",
                    "Seek academic guidance",
                    "Consider talking to a counselor"
                ],
                "color": "#F44336"  # Red
            }
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the main UI components"""
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Data Loading and Training
        self.training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text="Model Training")
        self.setup_training_tab()
        
        # Tab 2: Stress Prediction
        self.prediction_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_frame, text="Stress Prediction")
        self.setup_prediction_tab()
        
        # Tab 3: Model Analysis
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="Model Analysis")
        self.setup_analysis_tab()
    
    def setup_training_tab(self):
        """Setup the training tab"""
        
        # Title
        title_label = ttk.Label(self.training_frame, text="Model Training", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # File selection frame
        file_frame = ttk.LabelFrame(self.training_frame, text="Dataset Selection", padding=10)
        file_frame.pack(fill='x', padx=20, pady=10)
        
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=50).pack(side='left', padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side='left', padx=5)
        ttk.Button(file_frame, text="Load Dataset", command=self.load_dataset).pack(side='left', padx=5)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(self.training_frame, text="Training Progress", padding=10)
        progress_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                          maximum=100, length=400)
        self.progress_bar.pack(pady=10)
        
        # Status label
        self.status_label = ttk.Label(progress_frame, text="Ready to train model", 
                                    font=('Arial', 10))
        self.status_label.pack(pady=5)
        
        # Training button
        self.train_button = ttk.Button(progress_frame, text="Start Training", 
                                      command=self.start_training, state='disabled')
        self.train_button.pack(pady=10)
        
        # Results text area
        results_frame = ttk.LabelFrame(self.training_frame, text="Training Results", padding=10)
        results_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.results_text = tk.Text(results_frame, height=15, width=80, wrap='word')
        scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
    
    def setup_prediction_tab(self):
        """Setup the prediction tab"""
        
        # Title
        title_label = ttk.Label(self.prediction_frame, text="Stress Level Prediction", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Input frame
        input_frame = ttk.LabelFrame(self.prediction_frame, text="Enter Student Parameters", 
                                    padding=10)
        input_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create scrollable frame for inputs
        canvas = tk.Canvas(input_frame, height=400)
        scrollbar = ttk.Scrollbar(input_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Input fields will be added dynamically after training
        self.input_fields_frame = scrollable_frame
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Prediction button
        self.predict_button = ttk.Button(self.prediction_frame, text="Predict Stress Level", 
                                       command=self.predict_stress, state='disabled')
        self.predict_button.pack(pady=10)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.prediction_frame, text="Prediction Results", 
                                      padding=10)
        results_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Stress level display
        self.stress_level_label = ttk.Label(results_frame, text="", font=('Arial', 14, 'bold'))
        self.stress_level_label.pack(pady=10)
        
        # Counseling advice
        self.counseling_text = tk.Text(results_frame, height=10, width=80, wrap='word')
        counseling_scrollbar = ttk.Scrollbar(results_frame, orient='vertical', 
                                           command=self.counseling_text.yview)
        self.counseling_text.configure(yscrollcommand=counseling_scrollbar.set)
        
        self.counseling_text.pack(side='left', fill='both', expand=True)
        counseling_scrollbar.pack(side='right', fill='y')
    
    def setup_analysis_tab(self):
        """Setup the enhanced analysis tab"""
        
        # Title
        title_label = ttk.Label(self.analysis_frame, text="Enhanced Model Analysis", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(self.analysis_frame, text="Analysis Controls", padding=10)
        control_frame.pack(fill='x', padx=20, pady=5)
        
        # Row 1: Basic analysis buttons
        basic_frame = ttk.Frame(control_frame)
        basic_frame.pack(fill='x', pady=5)
        
        ttk.Button(basic_frame, text="Feature Importance", 
                  command=self.show_feature_importance).pack(side='left', padx=3)
        ttk.Button(basic_frame, text="Correlation Heatmap", 
                  command=self.show_correlation_heatmap).pack(side='left', padx=3)
        ttk.Button(basic_frame, text="Confusion Matrix", 
                  command=self.show_confusion_matrix).pack(side='left', padx=3)
        
        # Row 2: Advanced analysis buttons
        advanced_frame = ttk.Frame(control_frame)
        advanced_frame.pack(fill='x', pady=5)
        
        ttk.Button(advanced_frame, text="Stress Distribution", 
                  command=self.show_stress_distribution).pack(side='left', padx=3)
        ttk.Button(advanced_frame, text="Prediction Confidence", 
                  command=self.show_prediction_confidence).pack(side='left', padx=3)
        ttk.Button(advanced_frame, text="Feature Comparison", 
                  command=self.show_feature_comparison).pack(side='left', padx=3)
        
        # Row 3: Performance metrics
        metrics_frame = ttk.Frame(control_frame)
        metrics_frame.pack(fill='x', pady=5)
        
        ttk.Button(metrics_frame, text="Performance Metrics", 
                  command=self.show_performance_metrics).pack(side='left', padx=3)
        ttk.Button(metrics_frame, text="Learning Curves", 
                  command=self.show_learning_curves).pack(side='left', padx=3)
        ttk.Button(metrics_frame, text="ROC Curves", 
                  command=self.show_roc_curves).pack(side='left', padx=3)
        
        # Row 4: Export options
        export_frame = ttk.Frame(control_frame)
        export_frame.pack(fill='x', pady=5)
        
        ttk.Button(export_frame, text="Export Report", 
                  command=self.export_analysis_report).pack(side='left', padx=3)
        ttk.Button(export_frame, text="Save All Plots", 
                  command=self.save_all_plots).pack(side='left', padx=3)
        
        # Analysis info panel
        info_frame = ttk.LabelFrame(self.analysis_frame, text="Analysis Information", padding=10)
        info_frame.pack(fill='x', padx=20, pady=5)
        
        self.analysis_info_text = tk.Text(info_frame, height=3, width=100, wrap='word')
        self.analysis_info_text.pack(fill='x')
        self.analysis_info_text.insert(tk.END, "Select an analysis option to view detailed model insights and visualizations.")
        self.analysis_info_text.config(state='disabled')
        
        # Plot area with scroll
        plot_container = ttk.Frame(self.analysis_frame)
        plot_container.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create canvas for scrolling
        self.analysis_canvas = tk.Canvas(plot_container)
        self.analysis_scrollbar = ttk.Scrollbar(plot_container, orient="vertical", 
                                                command=self.analysis_canvas.yview)
        self.analysis_plot_frame = ttk.Frame(self.analysis_canvas)
        
        self.analysis_plot_frame.bind(
            "<Configure>",
            lambda e: self.analysis_canvas.configure(scrollregion=self.analysis_canvas.bbox("all"))
        )
        
        self.analysis_canvas.create_window((0, 0), window=self.analysis_plot_frame, anchor="nw")
        self.analysis_canvas.configure(yscrollcommand=self.analysis_scrollbar.set)
        
        self.analysis_canvas.pack(side="left", fill="both", expand=True)
        self.analysis_scrollbar.pack(side="right", fill="y")
    
    def browse_file(self):
        """Browse for dataset file"""
        filename = filedialog.askopenfilename(
            title="Select Dataset File",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if filename:
            self.file_path_var.set(filename)
    
    def load_dataset(self):
        """Load and display dataset information"""
        file_path = self.file_path_var.get()
        
        if not file_path:
            messagebox.showerror("Error", "Please select a dataset file first.")
            return
        
        try:
            # Load dataset
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                self.data = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Please use CSV or Excel files.")
            
            # Display dataset info
            info_text = f"Dataset loaded successfully!\n\n"
            info_text += f"Shape: {self.data.shape}\n"
            info_text += f"Features: {len(self.data.columns) - 1}\n"
            info_text += f"Target: Exam_Stress_Level\n\n"
            
            info_text += "Features:\n"
            for i, col in enumerate(self.data.columns):
                if col != 'Exam_Stress_Level':
                    info_text += f"  {i+1}. {col}\n"
            
            info_text += f"\nTarget Distribution:\n"
            stress_counts = self.data['Exam_Stress_Level'].value_counts().sort_index()
            stress_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
            for level, count in stress_counts.items():
                info_text += f"  {stress_labels.get(level, level)} ({level}): {count} samples\n"
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, info_text)
            
            # Enable training button
            self.train_button.config(state='normal')
            self.status_label.config(text="Dataset loaded. Ready to train model.")
            
            messagebox.showinfo("Success", "Dataset loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
    
    def start_training(self):
        """Start model training in a separate thread"""
        if self.data is None:
            messagebox.showerror("Error", "Please load dataset first.")
            return
        
        # Disable UI during training
        self.train_button.config(state='disabled')
        self.status_label.config(text="Training in progress...")
        
        # Start training in separate thread
        training_thread = threading.Thread(target=self.train_model)
        training_thread.daemon = True
        training_thread.start()
    
    def train_model(self):
        """Train the model"""
        try:
            # Update progress
            self.update_progress(10, "Preprocessing data...")
            
            # Preprocessing
            feature_columns = [col for col in self.data.columns if col != 'Exam_Stress_Level']
            X = self.data[feature_columns]
            y = self.data['Exam_Stress_Level']
            
            # Handle missing values
            X = X.fillna(X.median())
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.update_progress(30, "Training Random Forest model...")
            
            # Train model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train_scaled, y_train)
            
            self.update_progress(70, "Evaluating model...")
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Select top features
            self.selected_features = feature_importance.head(15)['Feature'].tolist()
            
            self.update_progress(90, "Finalizing...")
            
            # Display results
            results_text = "Training completed successfully!\n\n"
            results_text += f"Model: Random Forest Classifier\n"
            results_text += f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)\n\n"
            
            results_text += "Classification Report:\n"
            results_text += classification_report(y_test, y_pred, 
                                                target_names=['Low', 'Medium', 'High'])
            
            results_text += f"\nTop 15 Selected Features:\n"
            for i, (feature, importance) in enumerate(zip(feature_importance['Feature'][:15], 
                                                        feature_importance['Importance'][:15])):
                results_text += f"  {i+1}. {feature}: {importance:.4f}\n"
            
            # Update UI
            self.root.after(0, self.update_training_results, results_text)
            
            # Save model
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'selected_features': self.selected_features,
                'feature_columns': feature_columns
            }
            joblib.dump(model_data, "stress_model_gui.pkl")
            
            self.update_progress(100, "Training completed!")
            
            # Setup prediction interface
            self.root.after(0, self.setup_prediction_interface)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Training Error", str(e)))
            self.root.after(0, lambda: self.train_button.config(state='normal'))
    
    def update_progress(self, value, message):
        """Update progress bar and status"""
        self.progress_var.set(value)
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def update_training_results(self, results_text):
        """Update training results in UI"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, results_text)
        self.train_button.config(state='normal')
        self.predict_button.config(state='normal')
    
    def setup_prediction_interface(self):
        """Setup input fields for prediction"""
        # Clear existing fields
        for widget in self.input_fields_frame.winfo_children():
            widget.destroy()
        
        self.feature_entries = {}
        
        # Input prompts for features
        input_prompts = {
            'Exam_Anxiety_Level': "Exam Anxiety Level (1-5):",
            'anxiety_tension': "Anxiety/Tension Level (1-5):",
            'sleep_problems': "Sleep Problems Level (1-5):",
            'academic_overload': "Academic Overload Level (1-5):",
            'concentration_problems': "Concentration Problems Level (1-5):",
            'Motivation_Level': "Motivation Level (1-5):",
            'low_academic_confidence': "Low Academic Confidence Level (1-5):",
            'sadness_low_mood': "Sadness/Low Mood Level (1-5):",
            'Attendance_Percentage': "Attendance Percentage (0-100):",
            'Academic_Workload': "Academic Workload Level (1-5):",
            'Study_Fatigue_Index': "Study Fatigue Index (1-5):",
            'Learning_Disruption_Score': "Learning Disruption Score (1-5):",
            'heartbeat_palpitations': "Heartbeat Palpitations Level (1-5):",
            'restlessness': "Restlessness Level (1-5):",
            'headaches': "Headaches Level (1-5):"
        }
        
        # Create input fields
        for i, feature in enumerate(self.selected_features):
            frame = ttk.Frame(self.input_fields_frame)
            frame.pack(fill='x', padx=5, pady=2)
            
            prompt = input_prompts.get(feature, f"{feature.replace('_', ' ').title()}:")
            ttk.Label(frame, text=prompt, width=30).pack(side='left', padx=5)
            
            entry = ttk.Entry(frame, width=15)
            entry.pack(side='left', padx=5)
            self.feature_entries[feature] = entry
            
            # Set default value (median)
            if feature in self.data.columns:
                default_value = self.data[feature].median()
                entry.insert(0, str(default_value))
        
        messagebox.showinfo("Ready", "Model trained successfully! You can now predict stress levels.")
    
    def predict_stress(self):
        """Predict stress level for user input"""
        if self.model is None:
            messagebox.showerror("Error", "Please train the model first.")
            return
        
        try:
            # Collect user input with enhanced validation
            user_input = {}
            missing_fields = []
            invalid_fields = []
            
            for feature, entry in self.feature_entries.items():
                value = entry.get().strip()
                
                # Check for missing values
                if not value:
                    missing_fields.append(feature)
                    continue
                
                # Validate numeric input
                try:
                    numeric_value = float(value)
                    
                    # Validate ranges based on feature type
                    if feature == 'Attendance_Percentage':
                        if not (0 <= numeric_value <= 100):
                            invalid_fields.append(f"{feature}: {numeric_value} (must be 0-100)")
                            continue
                    else:
                        if not (1 <= numeric_value <= 5):
                            invalid_fields.append(f"{feature}: {numeric_value} (must be 1-5)")
                            continue
                    
                    user_input[feature] = numeric_value
                    
                except ValueError:
                    invalid_fields.append(f"{feature}: '{value}' (must be a number)")
                    continue
            
            # Show validation errors
            if missing_fields:
                messagebox.showerror("Missing Values", 
                    f"Please enter values for:\n" + "\n".join(f"• {field}" for field in missing_fields))
                return
            
            if invalid_fields:
                messagebox.showerror("Invalid Values", 
                    f"Please correct these values:\n" + "\n".join(f"• {field}" for field in invalid_fields))
                return
            
            if not user_input:
                messagebox.showerror("Error", "No valid input values provided.")
                return
            
            # Prepare input for prediction
            all_features = [col for col in self.data.columns if col != 'Exam_Stress_Level']
            input_vector = []
            
            for feature in all_features:
                if feature in user_input:
                    input_vector.append(user_input[feature])
                else:
                    input_vector.append(self.data[feature].median())
            
            # Scale and predict
            input_scaled = self.scaler.transform([input_vector])
            prediction = self.model.predict(input_scaled)[0]
            prediction_proba = self.model.predict_proba(input_scaled)[0]
            
            # Display results
            self.display_prediction_results(prediction, prediction_proba, user_input)
            
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
    
    def display_prediction_results(self, prediction, probabilities, user_input):
        """Display prediction results"""
        advice = self.counseling_advice[prediction]
        
        # Update stress level label
        self.stress_level_label.config(text=f"{advice['title']}", foreground=advice['color'])
        
        # Display counseling advice
        counseling_text = f"{advice['message']}\n\n"
        counseling_text += "Recommendations:\n"
        
        for i, rec in enumerate(advice['recommendations'], 1):
            counseling_text += f"{i}. {rec}\n"
        
        counseling_text += f"\nPrediction Confidence:\n"
        stress_labels = ['Low', 'Medium', 'High']
        for i, (label, prob) in enumerate(zip(stress_labels, probabilities)):
            counseling_text += f"{label}: {prob:.1%}\n"
        
        counseling_text += f"\nInput Values:\n"
        for feature, value in user_input.items():
            counseling_text += f"{feature}: {value}\n"
        
        self.counseling_text.delete(1.0, tk.END)
        self.counseling_text.insert(tk.END, counseling_text)
    
    def show_feature_importance(self):
        """Display enhanced feature importance plot"""
        if self.model is None:
            messagebox.showerror("Error", "Please train the model first.")
            return
        
        self.clear_analysis_plots()
        self.update_analysis_info("Feature Importance Analysis", 
                                 "Shows which features contribute most to stress level predictions.")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(14, 8))
        
        # Subplot 1: Horizontal bar chart
        ax1 = plt.subplot(2, 2, 1)
        feature_importance = pd.DataFrame({
            'Feature': self.selected_features,
            'Importance': [self.model.feature_importances_[i] for i in range(len(self.selected_features))]
        }).sort_values('Importance', ascending=True)
        
        bars = ax1.barh(feature_importance['Feature'], feature_importance['Importance'], 
                       color='skyblue', edgecolor='navy', alpha=0.7)
        ax1.set_xlabel('Importance Score', fontsize=10)
        ax1.set_title('Feature Importance Ranking', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=8)
        
        # Subplot 2: Top 10 features pie chart
        ax2 = plt.subplot(2, 2, 2)
        top_10 = feature_importance.tail(10)
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_10)))
        wedges, texts, autotexts = ax2.pie(top_10['Importance'], labels=top_10['Feature'], 
                                          autopct='%1.2f%%', colors=colors, startangle=90)
        ax2.set_title('Top 10 Features Distribution', fontweight='bold')
        
        # Subplot 3: Cumulative importance
        ax3 = plt.subplot(2, 2, 3)
        cumulative_importance = np.cumsum(feature_importance['Importance'].values[::-1])
        features_reversed = feature_importance['Feature'].values[::-1]
        
        ax3.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 
                'o-', color='red', linewidth=2, markersize=4)
        ax3.axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='95% Threshold')
        ax3.set_xlabel('Number of Features')
        ax3.set_ylabel('Cumulative Importance')
        ax3.set_title('Cumulative Feature Importance', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Subplot 4: Feature importance statistics
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        
        stats_text = f"""
        Feature Importance Statistics:
        ─────────────────────────
        Total Features: {len(self.selected_features)}
        Most Important: {feature_importance.iloc[-1]['Feature']}
        Importance Score: {feature_importance.iloc[-1]['Importance']:.4f}
        
        Least Important: {feature_importance.iloc[0]['Feature']}
        Importance Score: {feature_importance.iloc[0]['Importance']:.4f}
        
        Mean Importance: {feature_importance['Importance'].mean():.4f}
        Std Deviation: {feature_importance['Importance'].std():.4f}
        
        Features for 95% variance: {np.argmax(cumulative_importance >= 0.95) + 1}
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.analysis_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Add scrollbar if needed
        self.analysis_plot_frame.update_idletasks()
        if self.analysis_plot_frame.winfo_height() > 600:
            self.analysis_canvas.config(scrollregion=self.analysis_canvas.bbox("all"))
    
    def show_stress_distribution(self):
        """Display stress level distribution analysis"""
        if self.data is None:
            messagebox.showerror("Error", "Please load dataset first.")
            return
        
        self.clear_analysis_plots()
        self.update_analysis_info("Stress Distribution Analysis", 
                                 "Analyzes the distribution of stress levels across the dataset.")
        
        fig = plt.figure(figsize=(14, 10))
        
        # Subplot 1: Overall distribution
        ax1 = plt.subplot(2, 3, 1)
        stress_counts = self.data['Exam_Stress_Level'].value_counts().sort_index()
        colors = ['#4CAF50', '#FF9800', '#F44336']
        labels = ['Low Stress', 'Medium Stress', 'High Stress']
        
        bars = ax1.bar(labels, [stress_counts[0], stress_counts[1], stress_counts[2]], 
                      color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Overall Stress Distribution', fontweight='bold')
        ax1.set_ylabel('Number of Students')
        
        # Add value labels
        for bar, count in zip(bars, [stress_counts[0], stress_counts[1], stress_counts[2]]):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # Subplot 2: Pie chart
        ax2 = plt.subplot(2, 3, 2)
        wedges, texts, autotexts = ax2.pie([stress_counts[0], stress_counts[1], stress_counts[2]], 
                                          labels=labels, colors=colors, autopct='%1.1f%%',
                                          startangle=90, explode=(0.05, 0.05, 0.05))
        ax2.set_title('Stress Level Percentage', fontweight='bold')
        
        # Subplot 3: Gender-based distribution
        ax3 = plt.subplot(2, 3, 3)
        if 'gender' in self.data.columns:
            gender_stress = pd.crosstab(self.data['gender'], self.data['Exam_Stress_Level'])
            gender_labels = ['Female', 'Male']
            gender_stress.index = gender_labels
            gender_stress.columns = labels
            
            gender_stress.plot(kind='bar', ax=ax3, color=colors, alpha=0.7)
            ax3.set_title('Stress by Gender', fontweight='bold')
            ax3.set_ylabel('Number of Students')
            ax3.legend(title='Stress Level')
            ax3.tick_params(axis='x', rotation=0)
        
        # Subplot 4: Age distribution
        ax4 = plt.subplot(2, 3, 4)
        if 'age' in self.data.columns:
            for stress_level in [0, 1, 2]:
                subset = self.data[self.data['Exam_Stress_Level'] == stress_level]
                ax4.hist(subset['age'], alpha=0.6, label=labels[stress_level], 
                        bins=15, color=colors[stress_level])
            ax4.set_title('Age Distribution by Stress Level', fontweight='bold')
            ax4.set_xlabel('Age')
            ax4.set_ylabel('Frequency')
            ax4.legend()
        
        # Subplot 5: Attendance vs Stress
        ax5 = plt.subplot(2, 3, 5)
        if 'Attendance_Percentage' in self.data.columns:
            for stress_level in [0, 1, 2]:
                subset = self.data[self.data['Exam_Stress_Level'] == stress_level]
                ax5.scatter(subset['Attendance_Percentage'], [stress_level]*len(subset), 
                          alpha=0.6, color=colors[stress_level], s=30)
            ax5.set_title('Attendance vs Stress Level', fontweight='bold')
            ax5.set_xlabel('Attendance Percentage')
            ax5.set_ylabel('Stress Level')
            ax5.set_yticks([0, 1, 2])
            ax5.set_yticklabels(labels)
        
        # Subplot 6: Statistics summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        total_students = len(self.data)
        stats_text = f"""
        Dataset Statistics:
        ─────────────────────────
        Total Students: {total_students}
        
        Stress Distribution:
        • Low Stress: {stress_counts[0]} ({stress_counts[0]/total_students*100:.1f}%)
        • Medium Stress: {stress_counts[1]} ({stress_counts[1]/total_students*100:.1f}%)
        • High Stress: {stress_counts[2]} ({stress_counts[2]/total_students*100:.1f}%)
        
        Imbalance Ratio: {stress_counts[1]/stress_counts[0]:.2f}:1 (Medium:Low)
        
        Average Age: {self.data['age'].mean():.1f} years
        Avg Attendance: {self.data['Attendance_Percentage'].mean():.1f}%
        """
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.analysis_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def show_prediction_confidence(self):
        """Display prediction confidence analysis"""
        if self.model is None:
            messagebox.showerror("Error", "Please train the model first.")
            return
        
        self.clear_analysis_plots()
        self.update_analysis_info("Prediction Confidence Analysis", 
                                 "Analyzes model confidence and prediction reliability.")
        
        # Generate predictions for test set
        feature_columns = [col for col in self.data.columns if col != 'Exam_Stress_Level']
        X = self.data[feature_columns]
        y = self.data['Exam_Stress_Level']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)
        
        fig = plt.figure(figsize=(14, 10))
        
        # Subplot 1: Confidence distribution
        ax1 = plt.subplot(2, 3, 1)
        max_confidence = np.max(y_proba, axis=1)
        
        for stress_level in [0, 1, 2]:
            mask = (y_test == stress_level)
            ax1.hist(max_confidence[mask], alpha=0.6, 
                    label=f'{["Low", "Medium", "High"][stress_level]} Stress',
                    bins=20, color=['#4CAF50', '#FF9800', '#F44336'][stress_level])
        
        ax1.set_title('Prediction Confidence Distribution', fontweight='bold')
        ax1.set_xlabel('Maximum Prediction Probability')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Confidence vs Correctness
        ax2 = plt.subplot(2, 3, 2)
        correct = (y_pred == y_test)
        
        ax2.scatter(max_confidence[~correct], np.sum(~correct), 
                   color='red', alpha=0.6, s=50, label='Incorrect')
        ax2.scatter(max_confidence[correct], np.sum(correct), 
                   color='green', alpha=0.6, s=50, label='Correct')
        ax2.set_title('Confidence vs Prediction Accuracy', fontweight='bold')
        ax2.set_xlabel('Prediction Confidence')
        ax2.set_ylabel('Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Class-wise confidence
        ax3 = plt.subplot(2, 3, 3)
        class_confidence = []
        class_labels = ['Low', 'Medium', 'High']
        
        for i in range(3):
            mask = (y_test == i)
            if np.any(mask):
                class_confidence.append(np.mean(max_confidence[mask]))
            else:
                class_confidence.append(0)
        
        bars = ax3.bar(class_labels, class_confidence, 
                      color=['#4CAF50', '#FF9800', '#F44336'], alpha=0.7)
        ax3.set_title('Average Confidence by Class', fontweight='bold')
        ax3.set_ylabel('Average Confidence')
        ax3.set_ylim(0, 1)
        
        # Add value labels
        for bar, conf in zip(bars, class_confidence):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{conf:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Subplot 4: Confusion matrix with confidence
        ax4 = plt.subplot(2, 3, 4)
        cm = confusion_matrix(y_test, y_pred)
        
        # Normalize by true labels
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax4,
                   xticklabels=class_labels, yticklabels=class_labels)
        ax4.set_title('Normalized Confusion Matrix', fontweight='bold')
        ax4.set_ylabel('Actual')
        ax4.set_xlabel('Predicted')
        
        # Subplot 5: Confidence thresholds
        ax5 = plt.subplot(2, 3, 5)
        thresholds = np.arange(0.5, 1.0, 0.05)
        accuracies = []
        
        for threshold in thresholds:
            high_confidence_mask = max_confidence >= threshold
            if np.any(high_confidence_mask):
                accuracy = np.mean(correct[high_confidence_mask])
                accuracies.append(accuracy)
            else:
                accuracies.append(0)
        
        ax5.plot(thresholds, accuracies, 'o-', color='blue', linewidth=2, markersize=6)
        ax5.set_title('Accuracy vs Confidence Threshold', fontweight='bold')
        ax5.set_xlabel('Confidence Threshold')
        ax5.set_ylabel('Accuracy')
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(0.5, 1.0)
        ax5.set_ylim(0, 1)
        
        # Subplot 6: Statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        avg_confidence = np.mean(max_confidence)
        high_confidence_ratio = np.sum(max_confidence > 0.8) / len(max_confidence)
        
        stats_text = f"""
        Confidence Statistics:
        ─────────────────────────
        Test Samples: {len(y_test)}
        
        Overall Accuracy: {np.mean(correct):.3f} ({np.mean(correct)*100:.1f}%)
        Average Confidence: {avg_confidence:.3f}
        
        High Confidence (>0.8): {high_confidence_ratio*100:.1f}%
        Very High Confidence (>0.9): {np.sum(max_confidence > 0.9)/len(max_confidence)*100:.1f}%
        
        Most Confident Class: {class_labels[np.argmax(class_confidence)]}
        Least Confident Class: {class_labels[np.argmin(class_confidence)]}
        """
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.analysis_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def show_performance_metrics(self):
        """Display comprehensive performance metrics"""
        if self.model is None:
            messagebox.showerror("Error", "Please train the model first.")
            return
        
        self.clear_analysis_plots()
        self.update_analysis_info("Performance Metrics Analysis", 
                                 "Comprehensive evaluation of model performance across multiple metrics.")
        
        # Generate predictions
        feature_columns = [col for col in self.data.columns if col != 'Exam_Stress_Level']
        X = self.data[feature_columns]
        y = self.data['Exam_Stress_Level']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)
        
        fig = plt.figure(figsize=(14, 10))
        
        # Subplot 1: Classification metrics
        ax1 = plt.subplot(2, 3, 1)
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(y_test, y_pred, average=None)
        recall = recall_score(y_test, y_pred, average=None)
        f1 = f1_score(y_test, y_pred, average=None)
        
        x = np.arange(3)
        width = 0.25
        
        ax1.bar(x - width, precision, width, label='Precision', color='skyblue', alpha=0.8)
        ax1.bar(x, recall, width, label='Recall', color='lightgreen', alpha=0.8)
        ax1.bar(x + width, f1, width, label='F1-Score', color='salmon', alpha=0.8)
        
        ax1.set_title('Classification Metrics by Class', fontweight='bold')
        ax1.set_xlabel('Stress Level')
        ax1.set_ylabel('Score')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Low', 'Medium', 'High'])
        ax1.legend()
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: ROC curves
        ax2 = plt.subplot(2, 3, 2)
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        colors = ['#4CAF50', '#FF9800', '#F44336']
        
        for i, color in zip(range(3), colors):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax2.plot(fpr, tpr, color=color, lw=2, 
                    label=f'{["Low", "Medium", "High"][i]} (AUC = {roc_auc:.3f})')
        
        ax2.plot([0, 1], [0, 1], 'k--', lw=2)
        ax2.set_title('ROC Curves', fontweight='bold')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Precision-Recall curves
        ax3 = plt.subplot(2, 3, 3)
        from sklearn.metrics import precision_recall_curve
        
        for i, color in zip(range(3), colors):
            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
            pr_auc = auc(recall, precision)
            ax3.plot(recall, precision, color=color, lw=2,
                    label=f'{["Low", "Medium", "High"][i]} (AUC = {pr_auc:.3f})')
        
        ax3.set_title('Precision-Recall Curves', fontweight='bold')
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Learning curves
        ax4 = plt.subplot(2, 3, 4)
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, val_scores = learning_curve(
            self.model, X_train, y_train, cv=5, 
            train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        ax4.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        ax4.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
        
        ax4.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        ax4.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='red')
        
        ax4.set_title('Learning Curves', fontweight='bold')
        ax4.set_xlabel('Training Set Size')
        ax4.set_ylabel('Accuracy Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Subplot 5: Feature importance distribution
        ax5 = plt.subplot(2, 3, 5)
        importance_scores = self.model.feature_importances_
        
        ax5.hist(importance_scores, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax5.axvline(np.mean(importance_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(importance_scores):.4f}')
        ax5.set_title('Feature Importance Distribution', fontweight='bold')
        ax5.set_xlabel('Importance Score')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Subplot 6: Performance summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        overall_accuracy = accuracy_score(y_test, y_pred)
        macro_precision = precision_score(y_test, y_pred, average='macro')
        macro_recall = recall_score(y_test, y_pred, average='macro')
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        summary_text = f"""
        Performance Summary:
        ─────────────────────────
        Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.1f}%)
        
        Macro-Averaged Metrics:
        • Precision: {macro_precision:.4f}
        • Recall: {macro_recall:.4f}
        • F1-Score: {macro_f1:.4f}
        
        Class-wise Performance:
        • Low Stress:    P={precision[0]:.3f}, R={recall[0]:.3f}, F1={f1[0]:.3f}
        • Medium Stress: P={precision[1]:.3f}, R={recall[1]:.3f}, F1={f1[1]:.3f}
        • High Stress:   P={precision[2]:.3f}, R={recall[2]:.3f}, F1={f1[2]:.3f}
        
        Model Complexity: {len(self.selected_features)} features
        Training Samples: {len(X_train)}
        Test Samples: {len(X_test)}
        """
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.analysis_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def clear_analysis_plots(self):
        """Clear all plots from analysis frame"""
        for widget in self.analysis_plot_frame.winfo_children():
            widget.destroy()
    
    def update_analysis_info(self, title, description):
        """Update the analysis information panel"""
        self.analysis_info_text.config(state='normal')
        self.analysis_info_text.delete(1.0, tk.END)
        self.analysis_info_text.insert(tk.END, f"{title}\n{'='*len(title)}\n\n{description}")
        self.analysis_info_text.config(state='disabled')
    
    def show_correlation_heatmap(self):
        """Display enhanced correlation heatmap"""
        if self.data is None:
            messagebox.showerror("Error", "Please load dataset first.")
            return
        
        self.clear_analysis_plots()
        self.update_analysis_info("Correlation Analysis", 
                                 "Shows relationships between features and stress levels.")
        
        fig = plt.figure(figsize=(16, 12))
        
        # Subplot 1: Full correlation matrix
        ax1 = plt.subplot(2, 2, 1)
        
        # Select top features for correlation
        correlation_data = self.data[self.selected_features + ['Exam_Stress_Level']]
        correlation_matrix = correlation_data.corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f', ax=ax1, 
                   annot_kws={'size': 8}, cbar_kws={'shrink': 0.8})
        ax1.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=12)
        
        # Subplot 2: Target correlation
        ax2 = plt.subplot(2, 2, 2)
        
        target_corr = correlation_matrix['Exam_Stress_Level'].drop('Exam_Stress_Level').sort_values(ascending=True)
        colors = ['red' if x > 0 else 'blue' for x in target_corr.values]
        
        bars = ax2.barh(range(len(target_corr)), target_corr.values, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(target_corr)))
        ax2.set_yticklabels(target_corr.index, fontsize=8)
        ax2.set_xlabel('Correlation with Stress Level')
        ax2.set_title('Feature-Stress Correlation', fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, corr) in enumerate(zip(bars, target_corr.values)):
            ax2.text(bar.get_width() + 0.01 if corr > 0 else bar.get_width() - 0.01, 
                    bar.get_y() + bar.get_height()/2, f'{corr:.3f}', 
                    ha='left' if corr > 0 else 'right', va='center', fontsize=8)
        
        # Subplot 3: High correlation pairs
        ax3 = plt.subplot(2, 2, 3)
        
        # Find high correlation pairs (excluding target)
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if correlation_matrix.columns[i] != 'Exam_Stress_Level' and \
                   correlation_matrix.columns[j] != 'Exam_Stress_Level':
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:  # High correlation threshold
                        high_corr_pairs.append((correlation_matrix.columns[i], 
                                             correlation_matrix.columns[j], corr_val))
        
        if high_corr_pairs:
            pairs_df = pd.DataFrame(high_corr_pairs, columns=['Feature 1', 'Feature 2', 'Correlation'])
            pairs_df = pairs_df.sort_values('Correlation', key=abs, ascending=False)
            
            ax3.axis('tight')
            ax3.axis('off')
            table_data = [pairs_df.columns.tolist()] + pairs_df.values.tolist()
            table = ax3.table(cellText=table_data[1:], colLabels=table_data[0], 
                            cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            
            # Color code correlations
            for i in range(len(pairs_df)):
                corr_val = pairs_df.iloc[i]['Correlation']
                color = 'lightcoral' if corr_val > 0 else 'lightblue'
                for j in range(3):
                    table[(i+1, j)].set_facecolor(color)
        else:
            ax3.text(0.5, 0.5, 'No high correlations\nfound (|r| > 0.5)', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        
        ax3.set_title('High Correlation Pairs (|r| > 0.5)', fontweight='bold')
        
        # Subplot 4: Correlation statistics
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        
        # Calculate statistics
        abs_corr_with_target = np.abs(target_corr.values)
        strong_corr = np.sum(abs_corr_with_target > 0.5)
        moderate_corr = np.sum((abs_corr_with_target > 0.3) & (abs_corr_with_target <= 0.5))
        weak_corr = np.sum(abs_corr_with_target <= 0.3)
        
        stats_text = f"""
        Correlation Statistics:
        ─────────────────────────
        Total Features: {len(target_corr)}
        
        Correlation with Stress Level:
        • Strong (|r| > 0.5): {strong_corr} features
        • Moderate (0.3 < |r| ≤ 0.5): {moderate_corr} features
        • Weak (|r| ≤ 0.3): {weak_corr} features
        
        Most Positive Correlation:
        {target_corr.index[-1]} ({target_corr.values[-1]:.3f})
        
        Most Negative Correlation:
        {target_corr.index[0]} ({target_corr.values[0]:.3f})
        
        Average |Correlation|: {np.mean(abs_corr_with_target):.3f}
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.analysis_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def export_analysis_report(self):
        """Export comprehensive analysis report"""
        if self.model is None:
            messagebox.showerror("Error", "Please train model first.")
            return
        
        try:
            # Generate report content
            report = self.generate_analysis_report()
            
            # Save to file
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Save Analysis Report"
            )
            
            if filename:
                with open(filename, 'w') as f:
                    f.write(report)
                messagebox.showinfo("Success", f"Analysis report saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export report: {str(e)}")
    
    def generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        if self.model is None:
            return "No model trained yet."
        
        # Generate predictions for metrics
        feature_columns = [col for col in self.data.columns if col != 'Exam_Stress_Level']
        X = self.data[feature_columns]
        y = self.data['Exam_Stress_Level']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)
        
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        # Generate report
        report = "="*80 + "\n"
        report += "STUDENT STRESS PREDICTION - MODEL ANALYSIS REPORT\n"
        report += "="*80 + "\n\n"
        
        report += f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Dataset Information
        report += "DATASET INFORMATION\n"
        report += "-"*40 + "\n"
        report += f"Total Samples: {len(self.data)}\n"
        report += f"Features Used: {len(self.selected_features)}\n"
        report += f"Training Samples: {len(X_train)}\n"
        report += f"Test Samples: {len(X_test)}\n\n"
        
        # Stress Level Distribution
        stress_counts = self.data['Exam_Stress_Level'].value_counts().sort_index()
        report += "STRESS LEVEL DISTRIBUTION\n"
        report += "-"*40 + "\n"
        report += f"Low Stress (0): {stress_counts[0]} ({stress_counts[0]/len(self.data)*100:.1f}%)\n"
        report += f"Medium Stress (1): {stress_counts[1]} ({stress_counts[1]/len(self.data)*100:.1f}%)\n"
        report += f"High Stress (2): {stress_counts[2]} ({stress_counts[2]/len(self.data)*100:.1f}%)\n\n"
        
        # Model Performance
        report += "MODEL PERFORMANCE\n"
        report += "-"*40 + "\n"
        report += f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f} ({accuracy_score(y_test, y_pred)*100:.1f}%)\n\n"
        
        report += "Classification Report:\n"
        report += classification_report(y_test, y_pred, 
                                    target_names=['Low', 'Medium', 'High']) + "\n"
        
        # Feature Importance
        report += "TOP 10 FEATURE IMPORTANCE\n"
        report += "-"*40 + "\n"
        feature_importance = pd.DataFrame({
            'Feature': self.selected_features,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        for i, (feature, importance) in enumerate(zip(feature_importance['Feature'][:10], 
                                                    feature_importance['Importance'][:10])):
            report += f"{i+1:2d}. {feature}: {importance:.4f}\n"
        
        report += "\n"
        
        # Confusion Matrix
        report += "CONFUSION MATRIX\n"
        report += "-"*40 + "\n"
        cm = confusion_matrix(y_test, y_pred)
        report += "Predicted →\n"
        report += "Actual ↓   Low  Medium  High\n"
        for i, row in enumerate(cm):
            report += f"{'Low' if i==0 else 'Medium' if i==1 else 'High':^7} {row[0]:5d} {row[1]:7d} {row[2]:5d}\n"
        
        report += "\n"
        
        # Recommendations
        report += "RECOMMENDATIONS\n"
        report += "-"*40 + "\n"
        
        # Analyze model performance
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > 0.9:
            report += "• Model performance is excellent. Ready for deployment.\n"
        elif accuracy > 0.8:
            report += "• Model performance is good. Consider fine-tuning for better results.\n"
        else:
            report += "• Model performance needs improvement. Consider feature engineering or different algorithms.\n"
        
        # Analyze class balance
        imbalance_ratio = stress_counts[1] / stress_counts[0]
        if imbalance_ratio > 5:
            report += "• Dataset is imbalanced. Consider using balanced sampling or weighted loss.\n"
        
        # Analyze feature importance
        if feature_importance['Importance'].std() > 0.05:
            report += "• Feature importance varies significantly. Consider feature selection.\n"
        
        report += "\n"
        
        report += "="*80 + "\n"
        report += "END OF REPORT\n"
        report += "="*80 + "\n"
        
        return report
    
    def save_all_plots(self):
        """Save all analysis plots to files"""
        if self.model is None:
            messagebox.showerror("Error", "Please train model first.")
            return
        
        try:
            # Ask for directory
            directory = filedialog.askdirectory(title="Select Directory to Save Plots")
            if not directory:
                return
            
            # Generate and save each plot
            plot_functions = [
                (self.show_feature_importance, "feature_importance.png"),
                (self.show_correlation_heatmap, "correlation_heatmap.png"),
                (self.show_confusion_matrix, "confusion_matrix.png"),
                (self.show_stress_distribution, "stress_distribution.png"),
                (self.show_prediction_confidence, "prediction_confidence.png"),
                (self.show_performance_metrics, "performance_metrics.png"),
                (self.show_learning_curves, "learning_curves.png"),
                (self.show_roc_curves, "roc_curves.png")
            ]
            
            saved_plots = []
            for plot_func, filename in plot_functions:
                try:
                    plot_func()
                    
                    # Get the current figure and save it
                    fig = plt.gcf()
                    filepath = f"{directory}/{filename}"
                    fig.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    saved_plots.append(filename)
                    
                except Exception as e:
                    print(f"Failed to save {filename}: {str(e)}")
            
            if saved_plots:
                messagebox.showinfo("Success", f"Saved {len(saved_plots)} plots to {directory}")
            else:
                messagebox.showwarning("Warning", "No plots were saved")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save plots: {str(e)}")
    
    def show_confusion_matrix(self):
        """Display enhanced confusion matrix"""
        if self.model is None:
            messagebox.showerror("Error", "Please train model first.")
            return
        
        self.clear_analysis_plots()
        self.update_analysis_info("Confusion Matrix Analysis", 
                                 "Detailed analysis of model prediction accuracy.")
        
        # Generate predictions
        feature_columns = [col for col in self.data.columns if col != 'Exam_Stress_Level']
        X = self.data[feature_columns]
        y = self.data['Exam_Stress_Level']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        fig = plt.figure(figsize=(14, 10))
        
        # Subplot 1: Raw confusion matrix
        ax1 = plt.subplot(2, 3, 1)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['Low', 'Medium', 'High'],
                   yticklabels=['Low', 'Medium', 'High'])
        ax1.set_title('Raw Confusion Matrix', fontweight='bold')
        ax1.set_ylabel('Actual')
        ax1.set_xlabel('Predicted')
        
        # Subplot 2: Normalized confusion matrix
        ax2 = plt.subplot(2, 3, 2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax2,
                   xticklabels=['Low', 'Medium', 'High'],
                   yticklabels=['Low', 'Medium', 'High'])
        ax2.set_title('Normalized Confusion Matrix', fontweight='bold')
        ax2.set_ylabel('Actual')
        ax2.set_xlabel('Predicted')
        
        # Subplot 3: Classification report heatmap
        ax3 = plt.subplot(2, 3, 3)
        from sklearn.metrics import classification_report
        
        report = classification_report(y_test, y_pred, 
                                    target_names=['Low', 'Medium', 'High'],
                                    output_dict=True)
        
        # Create metrics heatmap
        metrics = ['precision', 'recall', 'f1-score']
        classes = ['Low', 'Medium', 'High']
        
        metrics_matrix = []
        for metric in metrics:
            row = []
            for cls in classes:
                row.append(report[cls][metric])
            metrics_matrix.append(row)
        
        sns.heatmap(metrics_matrix, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax3,
                   xticklabels=classes, yticklabels=metrics)
        ax3.set_title('Performance Metrics Heatmap', fontweight='bold')
        ax3.set_ylabel('Metric')
        ax3.set_xlabel('Stress Level')
        
        # Subplot 4: Error analysis
        ax4 = plt.subplot(2, 3, 4)
        
        # Calculate error types
        errors = {}
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j:
                    error_type = f"{['Low', 'Medium', 'High'][i]} → {['Low', 'Medium', 'High'][j]}"
                    errors[error_type] = cm[i, j]
        
        if errors:
            error_labels = list(errors.keys())
            error_values = list(errors.values())
            
            colors = ['red' if 'High' in label else 'orange' for label in error_labels]
            bars = ax4.bar(error_labels, error_values, color=colors, alpha=0.7)
            ax4.set_title('Error Types Distribution', fontweight='bold')
            ax4.set_ylabel('Number of Errors')
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, val in zip(bars, error_values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        str(val), ha='center', va='bottom')
        else:
            ax4.text(0.5, 0.5, 'No errors!', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=14, color='green')
        
        # Subplot 5: Per-class accuracy
        ax5 = plt.subplot(2, 3, 5)
        
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        colors_class = ['#4CAF50', '#FF9800', '#F44336']
        
        bars = ax5.bar(['Low', 'Medium', 'High'], class_accuracy, 
                      color=colors_class, alpha=0.7)
        ax5.set_title('Per-Class Accuracy', fontweight='bold')
        ax5.set_ylabel('Accuracy')
        ax5.set_ylim(0, 1)
        ax5.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, acc in zip(bars, class_accuracy):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Subplot 6: Summary statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        overall_accuracy = np.trace(cm) / np.sum(cm)
        total_errors = np.sum(cm) - np.trace(cm)
        
        summary_text = f"""
        Confusion Matrix Summary:
        ─────────────────────────
        Test Samples: {np.sum(cm)}
        Correct Predictions: {np.trace(cm)}
        Total Errors: {total_errors}
        
        Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.1f}%)
        
        Per-Class Performance:
        • Low Stress:    {cm[0,0]}/{cm[0].sum()} ({class_accuracy[0]:.1%})
        • Medium Stress: {cm[1,1]}/{cm[1].sum()} ({class_accuracy[1]:.1%})
        • High Stress:   {cm[2,2]}/{cm[2].sum()} ({class_accuracy[2]:.1%})
        
        Most Common Error: {max(errors.items(), key=lambda x: x[1])[0] if errors else 'None'}
        """
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.analysis_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    app = StressPredictionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
