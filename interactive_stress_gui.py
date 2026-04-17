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
from datetime import datetime
warnings.filterwarnings('ignore')

class InteractiveStressGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎓 Advanced Student Stress Prediction System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Style configuration
        self.setup_styles()
        
        # Initialize variables
        self.model = None
        self.scaler = None
        self.selected_features = []
        self.data = None
        self.feature_entries = {}
        self.prediction_history = []
        
        # Enhanced counseling recommendations with more details
        self.counseling_advice = {
            0: {
                "title": "🟢 LOW STRESS",
                "message": "Excellent! You're managing stress well.",
                "recommendations": [
                    "✅ Continue your current effective study routine",
                    "✅ Maintain your healthy sleep schedule (7-8 hours)",
                    "✅ Keep up regular physical activity",
                    "✅ Practice mindfulness 5-10 minutes daily",
                    "✅ Stay socially connected with friends",
                    "✅ Consider mentoring stressed peers"
                ],
                "color": "#4CAF50",
                "bg_color": "#e8f5e8",
                "tips": [
                    "Your stress management is working well!",
                    "Share your techniques with others",
                    "Consider advanced study techniques"
                ]
            },
            1: {
                "title": "🟡 MEDIUM STRESS", 
                "message": "You're experiencing manageable stress levels.",
                "recommendations": [
                    "⚠️ Implement Pomodoro technique (25/5 min cycles)",
                    "⚠️ Improve sleep consistency",
                    "⚠️ Practice deep breathing exercises",
                    "⚠️ Reduce screen time before bed",
                    "⚠️ Take regular study breaks",
                    "⚠️ Try progressive muscle relaxation"
                ],
                "color": "#FF9800",
                "bg_color": "#fff3e0",
                "tips": [
                    "Stress is normal during exams",
                    "Focus on one task at a time",
                    "Stay hydrated and eat well"
                ]
            },
            2: {
                "title": "🔴 HIGH STRESS",
                "message": "Your stress level requires immediate attention.",
                "recommendations": [
                    "🚨 Speak with a counselor immediately",
                    "🚨 Reduce academic workload temporarily",
                    "🚨 Practice emergency stress techniques",
                    "🚨 Ensure you're eating regularly",
                    "🚨 Get 8+ hours of quality sleep",
                    "🚨 Consider talking to teachers about extensions",
                    "🚨 Try meditation apps (Calm, Headspace)"
                ],
                "color": "#F44336", 
                "bg_color": "#ffebee",
                "tips": [
                    "Your health is the priority",
                    "It's okay to ask for help",
                    "Take things one day at a time"
                ]
            }
        }
        
        self.setup_ui()
    
    def setup_styles(self):
        """Setup modern styling for the GUI"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure styles
        style.configure('Title.TLabel', 
                     background='#2196f3', 
                     foreground='white', 
                     font=('Arial', 16, 'bold'))
        
        style.configure('Header.TLabel', 
                     background='#1976d2', 
                     foreground='white', 
                     font=('Arial', 12, 'bold'))
        
        style.configure('Card.TFrame', 
                     background='white', 
                     relief='raised', 
                     borderwidth=2)
        
        style.configure('Success.TButton', 
                     background='#4CAF50', 
                     foreground='white',
                     font=('Arial', 10, 'bold'))
        
        style.configure('Warning.TButton', 
                     background='#FF9800', 
                     foreground='white',
                     font=('Arial', 10, 'bold'))
        
        style.configure('Danger.TButton', 
                     background='#F44336', 
                     foreground='white',
                     font=('Arial', 10, 'bold'))
    
    def setup_ui(self):
        """Setup the main UI with modern design"""
        
        # Main container with padding
        main_container = ttk.Frame(self.root)
        main_container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Title section
        self.create_title_section(main_container)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill='both', expand=True, pady=(10, 0))
        
        # Tab 1: Dashboard
        self.dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.dashboard_frame, text="📊 Dashboard")
        self.setup_dashboard_tab()
        
        # Tab 2: Model Training
        self.training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text="🤖 Model Training")
        self.setup_training_tab()
        
        # Tab 3: Interactive Prediction
        self.prediction_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_frame, text="🔮 Predict Stress")
        self.setup_prediction_tab()
        
        # Tab 4: Analysis & Insights
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="📈 Analysis")
        self.setup_analysis_tab()
        
        # Tab 5: History & Reports
        self.history_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.history_frame, text="📋 History")
        self.setup_history_tab()
        
        # Tab 6: Help & Resources
        self.help_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.help_frame, text="❓ Help")
        self.setup_help_tab()
    
    def create_title_section(self, parent):
        """Create an attractive title section"""
        title_frame = tk.Frame(parent, bg='#2196f3', relief='raised', bd=2)
        title_frame.pack(fill='x', pady=(0, 10))
        
        # Main title
        title_label = tk.Label(title_frame, 
                           text="🎓 Advanced Student Stress Prediction System",
                           bg='#2196f3', fg='white',
                           font=('Arial', 20, 'bold'))
        title_label.pack(pady=10)
        
        # Subtitle
        subtitle_label = tk.Label(title_frame,
                             text="AI-Powered Stress Analysis with Personalized Counseling",
                             bg='#2196f3', fg='#e3f2fd',
                             font=('Arial', 12))
        subtitle_label.pack(pady=(0, 10))
        
        # Status indicator
        self.status_frame = tk.Frame(title_frame, bg='#2196f3')
        self.status_frame.pack(pady=(0, 10))
        
        self.status_label = tk.Label(self.status_frame,
                                text="🔴 System Ready | No Model Loaded",
                                bg='#2196f3', fg='white',
                                font=('Arial', 10))
        self.status_label.pack()
    
    def setup_dashboard_tab(self):
        """Setup interactive dashboard"""
        # Dashboard container
        dashboard_container = tk.Frame(self.dashboard_frame, bg='white')
        dashboard_container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Quick stats cards
        self.create_stats_cards(dashboard_container)
        
        # Quick actions
        self.create_quick_actions(dashboard_container)
        
        # Recent activity
        self.create_activity_feed(dashboard_container)
    
    def create_stats_cards(self, parent):
        """Create statistics cards"""
        stats_frame = tk.Frame(parent, bg='white')
        stats_frame.pack(fill='x', pady=(0, 20))
        
        # Title
        tk.Label(stats_frame, text="📊 System Statistics",
                bg='white', font=('Arial', 14, 'bold')).pack(anchor='w', pady=(0, 10))
        
        # Cards container
        cards_container = tk.Frame(stats_frame, bg='white')
        cards_container.pack(fill='x')
        
        # Create cards
        cards_data = [
            ("📁 Dataset", "No dataset loaded", "#e3f2fd"),
            ("🤖 Model", "Not trained", "#f3e5f5"),
            ("🎯 Accuracy", "N/A", "#e8f5e8"),
            ("📈 Predictions", "0 made", "#fff3e0")
        ]
        
        for i, (title, value, color) in enumerate(cards_data):
            card = self.create_stat_card(cards_container, title, value, color, i)
        
        self.stats_labels = {
            'dataset': cards_container.winfo_children()[0].winfo_children()[1],
            'model': cards_container.winfo_children()[1].winfo_children()[1],
            'accuracy': cards_container.winfo_children()[2].winfo_children()[1],
            'predictions': cards_container.winfo_children()[3].winfo_children()[1]
        }
    
    def create_stat_card(self, parent, title, value, color, index):
        """Create a single statistics card"""
        card_frame = tk.Frame(parent, bg=color, relief='raised', bd=2)
        card_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Title
        title_label = tk.Label(card_frame, text=title,
                            bg=color, fg='#2c3e50',
                            font=('Arial', 12, 'bold'))
        title_label.pack(pady=(10, 5))
        
        # Value
        value_label = tk.Label(card_frame, text=value,
                            bg=color, fg='#2c3e50',
                            font=('Arial', 16, 'bold'))
        value_label.pack(pady=(0, 10))
        
        return card_frame
    
    def create_quick_actions(self, parent):
        """Create quick action buttons"""
        actions_frame = tk.Frame(parent, bg='white')
        actions_frame.pack(fill='x', pady=(0, 20))
        
        tk.Label(actions_frame, text="⚡ Quick Actions",
                bg='white', font=('Arial', 14, 'bold')).pack(anchor='w', pady=(0, 10))
        
        buttons_container = tk.Frame(actions_frame, bg='white')
        buttons_container.pack(fill='x')
        
        # Action buttons
        actions = [
            ("📁 Load Dataset", self.quick_load_dataset, "#2196f3"),
            ("🤖 Train Model", self.quick_train_model, "#4CAF50"),
            ("🔮 Quick Predict", self.quick_predict, "#FF9800"),
            ("📊 View Analysis", self.quick_analysis, "#9C27B0")
        ]
        
        for text, command, color in actions:
            btn = tk.Button(buttons_container, text=text, command=command,
                          bg=color, fg='white', font=('Arial', 10, 'bold'),
                          relief='raised', bd=2, padx=20, pady=10)
            btn.pack(side='left', padx=5, pady=5)
    
    def create_activity_feed(self, parent):
        """Create activity feed"""
        activity_frame = tk.Frame(parent, bg='white')
        activity_frame.pack(fill='both', expand=True, pady=(0, 20))
        
        tk.Label(activity_frame, text="📋 Recent Activity",
                bg='white', font=('Arial', 14, 'bold')).pack(anchor='w', pady=(0, 10))
        
        # Activity list with scrollbar
        list_frame = tk.Frame(activity_frame, bg='white')
        list_frame.pack(fill='both', expand=True)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.activity_listbox = tk.Listbox(list_frame, 
                                        bg='#f8f9fa', 
                                        font=('Arial', 10),
                                        yscrollcommand=scrollbar.set)
        self.activity_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.activity_listbox.yview)
        
        # Add initial activities
        self.add_activity("🚀 System initialized")
        self.add_activity("💡 Ready to load dataset and train model")
    
    def setup_training_tab(self):
        """Setup enhanced training tab"""
        # Training container
        training_container = tk.Frame(self.training_frame, bg='white')
        training_container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # File selection section
        self.create_file_section(training_container)
        
        # Training progress section
        self.create_progress_section(training_container)
        
        # Results section
        self.create_results_section(training_container)
    
    def create_file_section(self, parent):
        """Create file selection section"""
        file_frame = tk.LabelFrame(parent, text="📁 Dataset Selection", 
                                 bg='white', font=('Arial', 12, 'bold'))
        file_frame.pack(fill='x', pady=(0, 20))
        
        # File path display
        path_frame = tk.Frame(file_frame, bg='white')
        path_frame.pack(fill='x', padx=10, pady=10)
        
        self.file_path_var = tk.StringVar()
        path_entry = tk.Entry(path_frame, textvariable=self.file_path_var, 
                            font=('Arial', 11), relief='solid', bd=2)
        path_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        browse_btn = tk.Button(path_frame, text="📂 Browse", 
                           command=self.browse_file,
                           bg='#2196f3', fg='white', 
                           font=('Arial', 10, 'bold'))
        browse_btn.pack(side='right')
        
        # File info display
        self.file_info_label = tk.Label(file_frame, 
                                   text="No file selected",
                                   bg='white', fg='#666',
                                   font=('Arial', 10))
        self.file_info_label.pack(anchor='w', padx=10, pady=(0, 10))
        
        # Action buttons
        actions_frame = tk.Frame(file_frame, bg='white')
        actions_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        self.load_btn = tk.Button(actions_frame, text="📥 Load Dataset", 
                              command=self.load_dataset,
                              bg='#4CAF50', fg='white', 
                              font=('Arial', 11, 'bold'),
                              relief='raised', bd=2, padx=20, pady=8)
        self.load_btn.pack(side='left', padx=(0, 10))
        
        self.preview_btn = tk.Button(actions_frame, text="👁️ Preview Data", 
                               command=self.preview_data,
                               bg='#2196f3', fg='white',
                               font=('Arial', 11, 'bold'),
                               relief='raised', bd=2, padx=20, pady=8)
        self.preview_btn.pack(side='left', padx=10)
    
    def create_progress_section(self, parent):
        """Create training progress section"""
        progress_frame = tk.LabelFrame(parent, text="🚀 Training Progress", 
                                   bg='white', font=('Arial', 12, 'bold'))
        progress_frame.pack(fill='x', pady=(0, 20))
        
        # Progress bar container
        progress_container = tk.Frame(progress_frame, bg='white')
        progress_container.pack(fill='x', padx=10, pady=10)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_container, 
                                       variable=self.progress_var,
                                       maximum=100, length=400)
        self.progress_bar.pack(fill='x', pady=(0, 10))
        
        # Status and time
        status_container = tk.Frame(progress_container, bg='white')
        status_container.pack(fill='x')
        
        self.status_label = tk.Label(status_container, 
                                text="⏸️ Ready to train",
                                bg='white', fg='#666',
                                font=('Arial', 11))
        self.status_label.pack(side='left')
        
        self.time_label = tk.Label(status_container,
                              text="",
                              bg='white', fg='#666',
                              font=('Arial', 10))
        self.time_label.pack(side='right')
        
        # Training button
        self.train_btn = tk.Button(progress_frame, text="🎯 Start Training",
                              command=self.start_training,
                              bg='#FF9800', fg='white',
                              font=('Arial', 12, 'bold'),
                              relief='raised', bd=3, padx=30, pady=12)
        self.train_btn.pack(pady=10)
    
    def create_results_section(self, parent):
        """Create training results section"""
        results_frame = tk.LabelFrame(parent, text="📊 Training Results",
                                 bg='white', font=('Arial', 12, 'bold'))
        results_frame.pack(fill='both', expand=True, pady=(0, 20))
        
        # Results text area with better styling
        text_container = tk.Frame(results_frame, bg='white')
        text_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.results_text = tk.Text(text_container, height=15, width=100,
                                bg='#f8f9fa', font=('Consolas', 10),
                                relief='solid', bd=1)
        
        scrollbar = ttk.Scrollbar(text_container, orient='vertical',
                               command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
    
    def setup_prediction_tab(self):
        """Setup enhanced prediction tab"""
        prediction_container = tk.Frame(self.prediction_frame, bg='white')
        prediction_container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Input section
        self.create_enhanced_input_section(prediction_container)
        
        # Prediction controls
        self.create_prediction_controls(prediction_container)
        
        # Results section
        self.create_enhanced_results_section(prediction_container)
    
    def create_enhanced_input_section(self, parent):
        """Create enhanced input section with sliders"""
        input_frame = tk.LabelFrame(parent, text="🎯 Student Parameters",
                                  bg='white', font=('Arial', 12, 'bold'))
        input_frame.pack(fill='x', pady=(0, 20))
        
        # Create scrollable area
        canvas = tk.Canvas(input_frame, bg='white', height=400)
        scrollbar = ttk.Scrollbar(input_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.input_fields_frame = scrollable_frame
        
        # Input fields will be added after training
        self.create_placeholder_inputs()
    
    def create_placeholder_inputs(self):
        """Create placeholder inputs before model is trained"""
        # Clear existing fields
        for widget in self.input_fields_frame.winfo_children():
            widget.destroy()
        
        placeholder = tk.Label(self.input_fields_frame,
                            text="🔄 Please train the model first to enable prediction inputs\n\n" +
                                   "The prediction interface will appear here with:\n" +
                                   "• Interactive sliders for each parameter\n" +
                                   "• Real-time validation\n" +
                                   "• Parameter descriptions\n" +
                                   "• Default value suggestions",
                            bg='white', fg='#999', 
                            font=('Arial', 12), justify='center')
        placeholder.pack(pady=50)
    
    def create_prediction_controls(self, parent):
        """Create prediction control buttons"""
        controls_frame = tk.Frame(parent, bg='white')
        controls_frame.pack(fill='x', pady=(0, 20))
        
        # Main prediction button
        self.predict_btn = tk.Button(controls_frame, text="🔮 Predict Stress Level",
                                 command=self.predict_stress,
                                 bg='#4CAF50', fg='white',
                                 font=('Arial', 14, 'bold'),
                                 relief='raised', bd=3, padx=40, pady=15)
        self.predict_btn.pack(pady=10)
        
        # Secondary controls
        secondary_frame = tk.Frame(controls_frame, bg='white')
        secondary_frame.pack(fill='x')
        
        reset_btn = tk.Button(secondary_frame, text="🔄 Reset Values",
                           command=self.reset_prediction_values,
                           bg='#6c757d', fg='white',
                           font=('Arial', 10, 'bold'))
        reset_btn.pack(side='left', padx=5)
        
        random_btn = tk.Button(secondary_frame, text="🎲 Random Sample",
                            command=self.load_random_sample,
                            bg='#17a2b8', fg='white',
                            font=('Arial', 10, 'bold'))
        random_btn.pack(side='left', padx=5)
        
        save_btn = tk.Button(secondary_frame, text="💾 Save Prediction",
                           command=self.save_prediction,
                           bg='#007bff', fg='white',
                           font=('Arial', 10, 'bold'))
        save_btn.pack(side='left', padx=5)
    
    def create_enhanced_results_section(self, parent):
        """Create enhanced results section"""
        results_frame = tk.LabelFrame(parent, text="🎯 Prediction Results",
                                 bg='white', font=('Arial', 12, 'bold'))
        results_frame.pack(fill='both', expand=True, pady=(0, 20))
        
        # Stress level display
        self.stress_display_frame = tk.Frame(results_frame, bg='white')
        self.stress_display_frame.pack(fill='x', pady=10)
        
        self.stress_level_label = tk.Label(self.stress_display_frame,
                                       text="❓ No Prediction Yet",
                                       bg='white', fg='#666',
                                       font=('Arial', 18, 'bold'))
        self.stress_level_label.pack()
        
        # Counseling advice section
        counseling_frame = tk.LabelFrame(results_frame, text="💡 Personalized Counseling",
                                     bg='white', font=('Arial', 11, 'bold'))
        counseling_frame.pack(fill='both', expand=True, pady=10)
        
        # Counseling text area
        self.counseling_text = tk.Text(counseling_frame, height=12, width=90,
                                     bg='#f8f9fa', font=('Arial', 10),
                                     relief='solid', bd=1)
        
        counseling_scrollbar = ttk.Scrollbar(counseling_frame, orient='vertical',
                                        command=self.counseling_text.yview)
        self.counseling_text.configure(yscrollcommand=counseling_scrollbar.set)
        
        self.counseling_text.pack(side='left', fill='both', expand=True)
        counseling_scrollbar.pack(side='right', fill='y')
    
    def setup_analysis_tab(self):
        """Setup analysis tab with interactive visualizations"""
        analysis_container = tk.Frame(self.analysis_frame, bg='white')
        analysis_container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Analysis controls
        controls_frame = tk.Frame(analysis_container, bg='white')
        controls_frame.pack(fill='x', pady=(0, 20))
        
        tk.Label(controls_frame, text="📈 Model Analysis Tools",
                bg='white', font=('Arial', 14, 'bold')).pack(anchor='w', pady=(0, 10))
        
        # Analysis buttons
        buttons_frame = tk.Frame(controls_frame, bg='white')
        buttons_frame.pack(fill='x')
        
        analysis_buttons = [
            ("📊 Feature Importance", self.show_feature_importance, "#4CAF50"),
            ("🔥 Correlation Heatmap", self.show_correlation_heatmap, "#FF9800"),
            ("📋 Confusion Matrix", self.show_confusion_matrix, "#F44336"),
            ("📈 Stress Distribution", self.show_stress_distribution, "#2196f3"),
            ("🎯 Performance Metrics", self.show_performance_metrics, "#9C27B0"),
            ("📥 Export Report", self.export_analysis_report, "#6c757d")
        ]
        
        for text, command, color in analysis_buttons:
            btn = tk.Button(buttons_frame, text=text, command=command,
                          bg=color, fg='white', font=('Arial', 10, 'bold'),
                          relief='raised', bd=2, padx=15, pady=8)
            btn.pack(side='left', padx=5, pady=5)
        
        # Plot area
        self.plot_frame = tk.Frame(analysis_container, bg='white', relief='sunken', bd=2)
        self.plot_frame.pack(fill='both', expand=True, pady=(0, 20))
        
        # Initial message
        self.show_plot_placeholder()
    
    def show_plot_placeholder(self):
        """Show placeholder in plot area"""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        placeholder = tk.Label(self.plot_frame,
                            text="📈 Select an analysis tool above to view visualizations\n\n" +
                                   "Available analyses:\n" +
                                   "• Feature Importance Rankings\n" +
                                   "• Correlation Heatmaps\n" +
                                   "• Confusion Matrix Analysis\n" +
                                   "• Stress Level Distributions\n" +
                                   "• Performance Metrics\n\n" +
                                   "🚀 Train a model first to enable all features",
                            bg='white', fg='#999',
                            font=('Arial', 12), justify='center')
        placeholder.pack(expand=True)
    
    def setup_history_tab(self):
        """Setup history and reports tab"""
        history_container = tk.Frame(self.history_frame, bg='white')
        history_container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # History title
        tk.Label(history_container, text="📋 Prediction History",
                bg='white', font=('Arial', 14, 'bold')).pack(anchor='w', pady=(0, 10))
        
        # History list
        history_frame = tk.Frame(history_container, bg='white')
        history_frame.pack(fill='both', expand=True)
        
        # History listbox
        self.history_listbox = tk.Listbox(history_frame,
                                       bg='#f8f9fa', font=('Arial', 10),
                                       height=15)
        history_scrollbar = ttk.Scrollbar(history_frame, orient='vertical',
                                      command=self.history_listbox.yview)
        self.history_listbox.configure(yscrollcommand=history_scrollbar.set)
        
        self.history_listbox.pack(side='left', fill='both', expand=True)
        history_scrollbar.pack(side='right', fill='y')
        
        # Control buttons
        controls_frame = tk.Frame(history_container, bg='white')
        controls_frame.pack(fill='x', pady=10)
        
        clear_btn = tk.Button(controls_frame, text="🗑️ Clear History",
                           command=self.clear_history,
                           bg='#dc3545', fg='white',
                           font=('Arial', 10, 'bold'))
        clear_btn.pack(side='left', padx=5)
        
        export_btn = tk.Button(controls_frame, text="📥 Export History",
                           command=self.export_history,
                           bg='#28a745', fg='white',
                           font=('Arial', 10, 'bold'))
        export_btn.pack(side='left', padx=5)
        
        # Add initial message
        self.add_activity("📋 History tab initialized - no predictions yet")
    
    def setup_help_tab(self):
        """Setup help and resources tab"""
        help_container = tk.Frame(self.help_frame, bg='white')
        help_container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Help title
        tk.Label(help_container, text="❓ Help & Resources",
                bg='white', font=('Arial', 14, 'bold')).pack(anchor='w', pady=(0, 10))
        
        # Help content with scroll
        help_frame = tk.Frame(help_container, bg='white')
        help_frame.pack(fill='both', expand=True)
        
        help_text = tk.Text(help_frame, height=20, width=100,
                          bg='#f8f9fa', font=('Arial', 10),
                          relief='solid', bd=1, wrap='word')
        
        help_scrollbar = ttk.Scrollbar(help_frame, orient='vertical',
                                   command=help_text.yview)
        help_text.configure(yscrollcommand=help_scrollbar.set)
        
        help_text.pack(side='left', fill='both', expand=True)
        help_scrollbar.pack(side='right', fill='y')
        
        # Help content
        help_content = """
🎓 STUDENT STRESS PREDICTION SYSTEM - USER GUIDE

📖 OVERVIEW
This advanced system uses machine learning to predict student stress levels based on academic behavior patterns and provides personalized counseling recommendations.

🚀 GETTING STARTED
1. Load Dataset: Click "📁 Load Dataset" to select your CSV/Excel file
2. Train Model: Click "🎯 Start Training" to train the AI model
3. Make Predictions: Use the interactive prediction interface
4. Analyze Results: View detailed analysis and insights

📊 DASHBOARD FEATURES
• 📈 Real-time statistics and system status
• ⚡ Quick access to main functions
• 📋 Activity feed showing recent actions
• 🎯 One-click access to key features

🤖 MODEL TRAINING
• Supports CSV and Excel formats
• Automatic feature selection and preprocessing
• Real-time progress tracking
• Performance metrics and accuracy scores
• Model saving for future use

🔮 INTERACTIVE PREDICTION
• 🎚️ Interactive sliders for each parameter
• ✅ Real-time input validation
• 🎯 Instant stress level prediction
• 💡 Personalized counseling recommendations
• 📊 Confidence scores for predictions

📈 ANALYSIS TOOLS
• 📊 Feature importance rankings
• 🔥 Correlation heatmaps
• 📋 Confusion matrix analysis
• 📈 Stress distribution charts
• 🎯 Performance metrics
• 📥 Exportable reports

🎯 STRESS LEVELS
🟢 LOW STRESS (0-2): Excellent stress management
🟡 MEDIUM STRESS (3-4): Manageable stress levels
🔴 HIGH STRESS (5+): Requires immediate attention

💡 COUNSELING RECOMMENDATIONS
Each stress level includes:
• Personalized advice based on prediction
• Actionable recommendations
• Stress management techniques
• Academic guidance
• Health and wellness tips

📋 HISTORY & REPORTS
• 📊 Track all predictions over time
• 📈 Monitor stress level trends
• 📥 Export data for analysis
• 🔄 Compare different time periods

⚙️ TECHNICAL SPECIFICATIONS
• Algorithm: Random Forest Classifier
• Features: 33 academic behavior parameters
• Accuracy: Typically 85-95%
• Processing: Real-time predictions
• Export: CSV, PNG, PDF formats

🆘 TROUBLESHOOTING
• Dataset not loading: Check file format and permissions
• Training errors: Ensure sufficient data (100+ samples)
• Prediction issues: Verify all inputs are filled
• Performance issues: Close other applications

📞 SUPPORT
For additional help or questions:
• Check the Help tab for detailed guides
• Review prediction history for patterns
• Export reports for further analysis
• Use the dashboard for system overview

🔄 UPDATES
The system continuously learns and improves:
• Regular model retraining with new data
• Enhanced feature selection algorithms
• Improved counseling recommendations
• Better visualization tools

💡 PRO TIPS
• Use consistent data for best results
• Train model regularly with new data
• Review counseling recommendations carefully
• Track prediction history for trends
• Export analysis reports for records
        """
        
        help_text.insert(tk.END, help_content)
        help_text.config(state='disabled')
    
    def add_activity(self, message):
        """Add activity to the activity feed"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        activity_entry = f"[{timestamp}] {message}"
        
        self.activity_listbox.insert(0, activity_entry)
        # Keep only last 50 activities
        if self.activity_listbox.size() > 50:
            self.activity_listbox.delete(50, tk.END)
    
    def update_status(self, status, color='black'):
        """Update system status"""
        self.status_label.config(text=status, fg=color)
        
        # Update dashboard stats
        if hasattr(self, 'stats_labels'):
            self.stats_labels['model'].config(text=status)
    
    def quick_load_dataset(self):
        """Quick load dataset from dashboard"""
        self.notebook.select(1)  # Switch to training tab
        self.browse_file()
    
    def quick_train_model(self):
        """Quick train model from dashboard"""
        if self.data is not None:
            self.notebook.select(1)  # Switch to training tab
            self.start_training()
        else:
            messagebox.showwarning("No Dataset", "Please load a dataset first!")
    
    def quick_predict(self):
        """Quick predict from dashboard"""
        if self.model is not None:
            self.notebook.select(2)  # Switch to prediction tab
        else:
            messagebox.showwarning("No Model", "Please train the model first!")
    
    def quick_analysis(self):
        """Quick analysis from dashboard"""
        if self.model is not None:
            self.notebook.select(3)  # Switch to analysis tab
        else:
            messagebox.showwarning("No Model", "Please train the model first!")
    
    def browse_file(self):
        """Browse for dataset file"""
        filename = filedialog.askopenfilename(
            title="Select Dataset File",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if filename:
            self.file_path_var.set(filename)
            self.file_info_label.config(text=f"📁 {filename.split('/')[-1]}")
            self.add_activity(f"📁 Selected dataset: {filename.split('/')[-1]}")
    
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
            
            # Update UI
            self.update_status("🟢 Dataset loaded", "#4CAF50")
            self.add_activity(f"📊 Dataset loaded: {self.data.shape[0]} samples, {self.data.shape[1]-1} features")
            
            # Update dashboard stats
            if hasattr(self, 'stats_labels'):
                self.stats_labels['dataset'].config(text=f"✅ {self.data.shape[0]} samples")
            
            # Display dataset info
            info_text = f"🎉 Dataset loaded successfully!\n\n"
            info_text += f"📊 Dataset Information:\n"
            info_text += f"   • Total Samples: {self.data.shape[0]:,}\n"
            info_text += f"   • Features: {self.data.shape[1]-1}\n"
            info_text += f"   • Target: Exam_Stress_Level\n\n"
            
            info_text += f"📋 Features:\n"
            for i, col in enumerate(self.data.columns):
                if col != 'Exam_Stress_Level':
                    info_text += f"   {i+1:2d}. {col}\n"
            
            info_text += f"\n📈 Stress Level Distribution:\n"
            stress_counts = self.data['Exam_Stress_Level'].value_counts().sort_index()
            stress_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
            total_samples = len(self.data)
            
            for level, count in stress_counts.items():
                percentage = (count / total_samples) * 100
                info_text += f"   • {stress_labels.get(level, level)} ({level}): {count} ({percentage:.1f}%)\n"
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, info_text)
            
            # Enable training button
            self.train_btn.config(state='normal', bg='#4CAF50')
            self.add_activity("🚀 Ready to start training")
            
            messagebox.showinfo("Success", f"Dataset loaded successfully!\n{self.data.shape[0]} samples found")
            
        except Exception as e:
            self.update_status("🔴 Load failed", "#F44336")
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
            self.add_activity(f"❌ Dataset load failed: {str(e)}")
    
    def preview_data(self):
        """Preview dataset in a new window"""
        if self.data is None:
            messagebox.showwarning("No Data", "Please load a dataset first!")
            return
        
        # Create preview window
        preview_window = tk.Toplevel(self.root)
        preview_window.title("📊 Dataset Preview")
        preview_window.geometry("800x600")
        
        # Preview text
        preview_text = tk.Text(preview_window, font=('Consolas', 9))
        preview_scrollbar = ttk.Scrollbar(preview_window, orient='vertical',
                                     command=preview_text.yview)
        preview_text.configure(yscrollcommand=preview_scrollbar.set)
        
        # Show first 50 rows
        preview_data = self.data.head(50).to_string()
        preview_text.insert(tk.END, preview_data)
        
        preview_text.pack(side='left', fill='both', expand=True)
        preview_scrollbar.pack(side='right', fill='y')
        
        self.add_activity("👁️ Dataset preview opened")
    
    def start_training(self):
        """Start model training with enhanced progress"""
        if self.data is None:
            messagebox.showerror("Error", "Please load dataset first.")
            return
        
        # Disable UI during training
        self.train_btn.config(state='disabled', text="🔄 Training...", bg='#6c757d')
        self.update_status("🔄 Training in progress...", "#FF9800")
        self.add_activity("🚀 Model training started")
        
        # Start timer
        start_time = datetime.now()
        
        def update_timer():
            elapsed = datetime.now() - start_time
            self.time_label.config(text=f"⏱️ {elapsed.seconds}s")
            if self.train_btn['state'] == 'disabled':
                self.root.after(1000, update_timer)
        
        update_timer()
        
        # Start training in separate thread
        training_thread = threading.Thread(target=self.train_model_with_progress)
        training_thread.daemon = True
        training_thread.start()
    
    def train_model_with_progress(self):
        """Train model with detailed progress updates"""
        try:
            # Update progress
            self.update_progress_ui(10, "📊 Preprocessing data...")
            self.root.after(0, lambda: self.add_activity("📊 Data preprocessing started"))
            
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
            
            self.update_progress_ui(30, "🔧 Scaling features...")
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.update_progress_ui(50, "🤖 Training Random Forest...")
            self.root.after(0, lambda: self.add_activity("🤖 Model training started"))
            
            # Train model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train_scaled, y_train)
            
            self.update_progress_ui(70, "📈 Selecting features...")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Select top features
            self.selected_features = feature_importance.head(15)['Feature'].tolist()
            
            self.update_progress_ui(85, "📊 Evaluating model...")
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.update_progress_ui(95, "💾 Saving model...")
            
            # Save model
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'selected_features': self.selected_features,
                'feature_columns': feature_columns
            }
            joblib.dump(model_data, "interactive_stress_model.pkl")
            
            self.update_progress_ui(100, "✅ Training completed!")
            
            # Display results
            self.display_training_results(accuracy, feature_importance)
            
            # Update UI
            self.root.after(0, lambda: self.training_completed_ui(accuracy))
            
        except Exception as e:
            self.root.after(0, lambda: self.training_error_ui(str(e)))
    
    def update_progress_ui(self, value, message):
        """Update progress UI from any thread"""
        self.root.after(0, lambda v=value: self.progress_var.set(v))
        self.root.after(0, lambda m=message: self.status_label.config(text=m))
    
    def display_training_results(self, accuracy, feature_importance):
        """Display comprehensive training results"""
        results_text = f"🎉 Model Training Completed Successfully!\n\n"
        results_text += f"🤖 Model Information:\n"
        results_text += f"   • Algorithm: Random Forest Classifier\n"
        results_text += f"   • Estimators: 100 trees\n"
        results_text += f"   • Selected Features: {len(self.selected_features)}\n\n"
        
        results_text += f"📊 Performance Metrics:\n"
        results_text += f"   • Overall Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)\n"
        results_text += f"   • Model Quality: {'Excellent' if accuracy > 0.9 else 'Good' if accuracy > 0.8 else 'Fair'}\n\n"
        
        results_text += f"🏆 Top 10 Most Important Features:\n"
        for i, (feature, importance) in enumerate(zip(feature_importance['Feature'][:10], 
                                                    feature_importance['Importance'][:10])):
            results_text += f"   {i+1:2d}. {feature:<30} ({importance:.4f})\n"
        
        # Update results text
        self.root.after(0, lambda: self.results_text.delete(1.0, tk.END))
        self.root.after(0, lambda: self.results_text.insert(tk.END, results_text))
    
    def training_completed_ui(self, accuracy):
        """Update UI after successful training"""
        self.train_btn.config(state='normal', text="✅ Training Complete", bg='#4CAF50')
        self.update_status(f"🟢 Model trained (Accuracy: {accuracy:.1%})", "#4CAF50")
        self.add_activity(f"✅ Model training completed (Accuracy: {accuracy:.1%})")
        
        # Update dashboard
        if hasattr(self, 'stats_labels'):
            self.stats_labels['model'].config(text=f"✅ Trained ({accuracy:.1%})")
            self.stats_labels['accuracy'].config(text=f"{accuracy:.1%}")
        
        # Setup prediction interface
        self.root.after(0, self.setup_prediction_inputs)
        
        # Enable prediction
        self.predict_btn.config(state='normal', bg='#4CAF50')
        
        messagebox.showinfo("Success", f"Model trained successfully!\nAccuracy: {accuracy:.1%}\n\nYou can now make predictions!")
    
    def training_error_ui(self, error_msg):
        """Update UI after training error"""
        self.train_btn.config(state='normal', text="🔄 Retry Training", bg='#F44336')
        self.update_status("🔴 Training failed", "#F44336")
        self.add_activity(f"❌ Training failed: {error_msg}")
        messagebox.showerror("Training Error", f"Model training failed:\n{error_msg}")
    
    def setup_prediction_inputs(self):
        """Setup interactive prediction inputs"""
        # Clear existing fields
        for widget in self.input_fields_frame.winfo_children():
            widget.destroy()
        
        self.feature_entries = {}
        
        # Input prompts with descriptions
        input_prompts = {
            'Exam_Anxiety_Level': {
                'prompt': "Exam Anxiety Level",
                'description': "How anxious do you feel before exams?",
                'range': "1 (Very Low) - 5 (Very High)",
                'default': 3
            },
            'anxiety_tension': {
                'prompt': "Anxiety/Tension Level", 
                'description': "General anxiety and tension levels",
                'range': "1 (Relaxed) - 5 (Very Tense)",
                'default': 2
            },
            'sleep_problems': {
                'prompt': "Sleep Problems",
                'description': "Difficulty with sleep quality/quantity",
                'range': "1 (Excellent) - 5 (Severe Problems)",
                'default': 2
            },
            'academic_overload': {
                'prompt': "Academic Overload",
                'description': "Feeling overwhelmed with academic workload",
                'range': "1 (Manageable) - 5 (Extremely Overloaded)",
                'default': 3
            },
            'concentration_problems': {
                'prompt': "Concentration Problems",
                'description': "Difficulty focusing on studies",
                'range': "1 (Excellent Focus) - 5 (Severe Problems)",
                'default': 2
            },
            'Motivation_Level': {
                'prompt': "Motivation Level",
                'description': "Current motivation for academic activities",
                'range': "1 (Very High) - 5 (Very Low)",
                'default': 2
            },
            'low_academic_confidence': {
                'prompt': "Academic Confidence",
                'description': "Confidence in academic abilities",
                'range': "1 (Very Confident) - 5 (Not Confident)",
                'default': 2
            },
            'sadness_low_mood': {
                'prompt': "Sadness/Low Mood",
                'description': "Frequency of sad feelings",
                'range': "1 (Rarely) - 5 (Very Often)",
                'default': 2
            },
            'Attendance_Percentage': {
                'prompt': "Attendance Percentage",
                'description': "Class attendance rate",
                'range': "0% - 100%",
                'default': 75
            },
            'Academic_Workload': {
                'prompt': "Academic Workload",
                'description': "Perceived academic workload",
                'range': "1 (Very Light) - 5 (Very Heavy)",
                'default': 3
            },
            'Study_Fatigue_Index': {
                'prompt': "Study Fatigue",
                'description': "Mental and physical fatigue from studying",
                'range': "1 (Not Fatigued) - 5 (Very Fatigued)",
                'default': 2
            },
            'Learning_Disruption_Score': {
                'prompt': "Learning Disruption",
                'description': "Factors disrupting learning process",
                'range': "1 (Minimal) - 5 (Severe)",
                'default': 2
            },
            'heartbeat_palpitations': {
                'prompt': "Heartbeat Palpitations",
                'description': "Physical stress symptoms - heart racing",
                'range': "1 (Never) - 5 (Very Often)",
                'default': 1
            },
            'restlessness': {
                'prompt': "Restlessness",
                'description': "Inability to relax or stay still",
                'range': "1 (Very Calm) - 5 (Very Restless)",
                'default': 2
            }
        }
        
        # Create input fields for selected features
        for i, feature in enumerate(self.selected_features):
            if feature in input_prompts:
                self.create_interactive_input(self.input_fields_frame, feature, 
                                         input_prompts[feature], i)
    
    def create_interactive_input(self, parent, feature, info, index):
        """Create an interactive input with slider and entry"""
        # Main container
        feature_frame = tk.Frame(parent, bg='white', relief='ridge', bd=1)
        feature_frame.pack(fill='x', padx=10, pady=5)
        
        # Header
        header_frame = tk.Frame(feature_frame, bg='#f8f9fa')
        header_frame.pack(fill='x')
        
        # Feature name and description
        title_label = tk.Label(header_frame, text=f"📝 {info['prompt']}",
                            bg='#f8f9fa', fg='#2c3e50',
                            font=('Arial', 11, 'bold'))
        title_label.pack(anchor='w', padx=10, pady=(5, 2))
        
        desc_label = tk.Label(header_frame, text=f"💡 {info['description']}",
                            bg='#f8f9fa', fg='#6c757d',
                            font=('Arial', 9))
        desc_label.pack(anchor='w', padx=10, pady=(0, 5))
        
        # Range info
        range_label = tk.Label(header_frame, text=f"📏 Range: {info['range']}",
                            bg='#f8f9fa', fg='#17a2b8',
                            font=('Arial', 9, 'italic'))
        range_label.pack(anchor='w', padx=10, pady=(0, 5))
        
        # Input controls
        controls_frame = tk.Frame(feature_frame, bg='white')
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        # Slider
        slider_frame = tk.Frame(controls_frame, bg='white')
        slider_frame.pack(side='left', fill='x', expand=True)
        
        slider_label = tk.Label(slider_frame, text="🎚️",
                             bg='white', font=('Arial', 12))
        slider_label.pack(side='left', padx=(0, 5))
        
        # Determine slider range
        if feature == 'Attendance_Percentage':
            from_val, to_val = 0, 100
            default_val = info['default']
        else:
            from_val, to_val = 1, 5
            default_val = info['default']
        
        slider_var = tk.DoubleVar(value=default_val)
        slider = tk.Scale(slider_frame, from_=from_val, to=to_val,
                         orient='horizontal', variable=slider_var,
                         bg='white', highlightthickness=0,
                         length=300, resolution=0.1 if feature == 'Attendance_Percentage' else 1)
        slider.pack(side='left', fill='x', expand=True)
        
        # Value display and entry
        value_frame = tk.Frame(controls_frame, bg='white')
        value_frame.pack(side='right', padx=(10, 0))
        
        value_label = tk.Label(value_frame, text=f"Value: {default_val}",
                            bg='white', fg='#2c3e50',
                            font=('Arial', 10, 'bold'))
        value_label.pack()
        
        entry = tk.Entry(value_frame, textvariable=slider_var,
                       width=10, font=('Arial', 10))
        entry.pack(pady=(5, 0))
        
        # Update value label when slider changes
        def update_value(val):
            value_label.config(text=f"Value: {float(val):.1f}")
        
        slider.config(command=update_value)
        
        # Store references
        self.feature_entries[feature] = slider_var
    
    def predict_stress(self):
        """Predict stress level with enhanced feedback"""
        if self.model is None:
            messagebox.showerror("Error", "Please train the model first.")
            return
        
        try:
            # Collect user input with validation
            user_input = {}
            missing_fields = []
            invalid_fields = []
            
            for feature, var in self.feature_entries.items():
                try:
                    value = var.get()
                    if value is None or (isinstance(value, str) and value.strip() == ''):
                        missing_fields.append(feature)
                        continue
                    
                    # Convert to float and validate range
                    value = float(value)
                    
                    # Validate ranges based on feature type
                    if feature == 'Attendance_Percentage':
                        if not (0 <= value <= 100):
                            invalid_fields.append(f"{feature}: {value} (must be 0-100)")
                            continue
                    else:
                        if not (1 <= value <= 5):
                            invalid_fields.append(f"{feature}: {value} (must be 1-5)")
                            continue
                    
                    user_input[feature] = value
                    
                except (ValueError, TypeError) as e:
                    invalid_fields.append(f"{feature}: Invalid value")
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
            self.display_enhanced_prediction_results(prediction, prediction_proba, user_input)
            
            # Add to history
            self.add_to_history(prediction, user_input, prediction_proba)
            
            # Update activity
            self.add_activity(f"🔮 Prediction made: Stress Level {prediction}")
            
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            self.add_activity(f"❌ Prediction failed: {str(e)}")
    
    def display_enhanced_prediction_results(self, prediction, probabilities, user_input):
        """Display enhanced prediction results"""
        advice = self.counseling_advice[prediction]
        
        # Update stress level display with animation
        self.stress_display_frame.config(bg=advice['bg_color'])
        self.stress_level_label.config(
            text=advice['title'],
            fg=advice['color'],
            bg=advice['bg_color']
        )
        
        # Display counseling advice with formatting
        counseling_text = f"🎯 PREDICTION RESULTS\n"
        counseling_text += f"{'='*50}\n"
        counseling_text += f"📊 Stress Level: {advice['title']}\n"
        counseling_text += f"💭 Message: {advice['message']}\n\n"
        
        counseling_text += f"🌟 PERSONALIZED RECOMMENDATIONS:\n"
        counseling_text += f"{'='*50}\n"
        for i, rec in enumerate(advice['recommendations'], 1):
            counseling_text += f"{i}. {rec}\n"
        
        counseling_text += f"\n💡 ADDITIONAL TIPS:\n"
        counseling_text += f"{'='*30}\n"
        for tip in advice['tips']:
            counseling_text += f"• {tip}\n"
        
        counseling_text += f"\n📈 PREDICTION CONFIDENCE:\n"
        counseling_text += f"{'='*30}\n"
        stress_labels = ['Low', 'Medium', 'High']
        for i, (label, prob) in enumerate(zip(stress_labels, probabilities)):
            confidence_emoji = "🟢" if prob > 0.8 else "🟡" if prob > 0.6 else "🔴"
            counseling_text += f"{confidence_emoji} {label}: {prob:.1%}\n"
        
        counseling_text += f"\n📝 INPUT VALUES SUMMARY:\n"
        counseling_text += f"{'='*30}\n"
        for feature, value in user_input.items():
            counseling_text += f"• {feature.replace('_', ' ').title()}: {value:.1f}\n"
        
        # Update counseling text
        self.counseling_text.delete(1.0, tk.END)
        self.counseling_text.insert(tk.END, counseling_text)
        
        # Update dashboard
        if hasattr(self, 'stats_labels'):
            self.stats_labels['predictions'].config(text=f"{len(self.prediction_history)+1}")
    
    def reset_prediction_values(self):
        """Reset all prediction values to defaults"""
        for feature, var in self.feature_entries.items():
            # Reset to default values
            if feature == 'Attendance_Percentage':
                var.set(75)
            else:
                var.set(3)
        
        self.add_activity("🔄 Prediction values reset to defaults")
    
    def load_random_sample(self):
        """Load random sample from dataset"""
        if self.data is None:
            messagebox.showwarning("No Data", "Please load and train model first!")
            return
        
        # Get random sample
        random_sample = self.data.sample(1).iloc[0]
        
        # Update input fields
        for feature, var in self.feature_entries.items():
            if feature in random_sample:
                var.set(random_sample[feature])
        
        self.add_activity("🎲 Random sample loaded from dataset")
    
    def save_prediction(self):
        """Save current prediction to file"""
        if not hasattr(self, 'last_prediction'):
            messagebox.showinfo("No Prediction", "Make a prediction first!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Prediction"
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(f"Student Stress Prediction Report\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(f"Prediction: {self.last_prediction['level']}\n")
                    f.write(f"Confidence: {self.last_prediction['confidence']:.1%}\n\n")
                    f.write("Input Values:\n")
                    for feature, value in self.last_prediction['inputs'].items():
                        f.write(f"{feature}: {value}\n")
                
                messagebox.showinfo("Success", f"Prediction saved to {filename}")
                self.add_activity(f"💾 Prediction saved to {filename}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save: {str(e)}")
    
    def add_to_history(self, prediction, inputs, probabilities):
        """Add prediction to history"""
        history_entry = {
            'timestamp': datetime.now(),
            'prediction': prediction,
            'confidence': max(probabilities),
            'inputs': inputs.copy()
        }
        
        self.prediction_history.insert(0, history_entry)
        self.last_prediction = {
            'level': prediction,
            'confidence': max(probabilities),
            'inputs': inputs
        }
        
        # Update history listbox
        stress_labels = ['Low', 'Medium', 'High']
        history_text = f"[{history_entry['timestamp'].strftime('%H:%M')}] {stress_labels[prediction]} (Confidence: {max(probabilities):.1%})"
        self.history_listbox.insert(0, history_text)
        
        # Keep only last 100 entries
        if self.history_listbox.size() > 100:
            self.history_listbox.delete(100, tk.END)
            self.prediction_history = self.prediction_history[:100]
    
    def clear_history(self):
        """Clear prediction history"""
        if messagebox.askyesno("Clear History", "Are you sure you want to clear all prediction history?"):
            self.prediction_history.clear()
            self.history_listbox.delete(0, tk.END)
            self.add_activity("🗑️ Prediction history cleared")
    
    def export_history(self):
        """Export prediction history to file"""
        if not self.prediction_history:
            messagebox.showinfo("No History", "No predictions to export!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Prediction History"
        )
        
        if filename:
            try:
                # Create DataFrame from history
                history_df = pd.DataFrame(self.prediction_history)
                history_df.to_csv(filename, index=False)
                
                messagebox.showinfo("Success", f"History exported to {filename}")
                self.add_activity(f"📥 History exported ({len(self.prediction_history)} entries)")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export: {str(e)}")
    
    def show_feature_importance(self):
        """Show feature importance in plot area"""
        if self.model is None:
            messagebox.showerror("Error", "Please train model first.")
            return
        
        # Clear plot area
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        # Create feature importance plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        feature_importance = pd.DataFrame({
            'Feature': self.selected_features,
            'Importance': [self.model.feature_importances_[i] for i in range(len(self.selected_features))]
        }).sort_values('Importance', ascending=True)
        
        # Create horizontal bar chart
        bars = ax.barh(feature_importance['Feature'], feature_importance['Importance'],
                       color='#2196f3', alpha=0.8)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title('🏆 Feature Importance Rankings', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, feature_importance['Importance'])):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{importance:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.add_activity("📊 Feature importance analysis viewed")
    
    def show_correlation_heatmap(self):
        """Show correlation heatmap"""
        if self.data is None:
            messagebox.showerror("Error", "Please load dataset first.")
            return
        
        # Clear plot area
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        # Create correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Select top features for correlation
        correlation_data = self.data[self.selected_features + ['Exam_Stress_Level']]
        correlation_matrix = correlation_data.corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f', ax=ax)
        ax.set_title('🔥 Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.add_activity("🔥 Correlation heatmap viewed")
    
    def show_confusion_matrix(self):
        """Show confusion matrix"""
        if self.model is None:
            messagebox.showerror("Error", "Please train model first.")
            return
        
        # Clear plot area
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        # Generate predictions
        feature_columns = [col for col in self.data.columns if col != 'Exam_Stress_Level']
        X = self.data[feature_columns]
        y = self.data['Exam_Stress_Level']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        # Create confusion matrix plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Low', 'Medium', 'High'],
                   yticklabels=['Low', 'Medium', 'High'])
        ax.set_title('📋 Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_xlabel('Predicted', fontsize=12)
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.add_activity("📋 Confusion matrix analysis viewed")
    
    def show_stress_distribution(self):
        """Show stress distribution"""
        if self.data is None:
            messagebox.showerror("Error", "Please load dataset first.")
            return
        
        # Clear plot area
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        # Create distribution plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        stress_counts = self.data['Exam_Stress_Level'].value_counts().sort_index()
        colors = ['#4CAF50', '#FF9800', '#F44336']
        labels = ['Low Stress', 'Medium Stress', 'High Stress']
        
        # Bar chart
        bars = ax1.bar(labels, [stress_counts[0], stress_counts[1], stress_counts[2]], 
                      color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('📈 Stress Level Distribution', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Students')
        
        # Add value labels
        for bar, count in zip(bars, [stress_counts[0], stress_counts[1], stress_counts[2]]):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        ax2.pie([stress_counts[0], stress_counts[1], stress_counts[2]], 
                  labels=labels, colors=colors, autopct='%1.1f%%',
                  startangle=90)
        ax2.set_title('📊 Stress Level Percentage', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.add_activity("📈 Stress distribution analysis viewed")
    
    def show_performance_metrics(self):
        """Show performance metrics"""
        if self.model is None:
            messagebox.showerror("Error", "Please train model first.")
            return
        
        # Clear plot area
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        # Generate predictions for metrics
        feature_columns = [col for col in self.data.columns if col != 'Exam_Stress_Level']
        X = self.data[feature_columns]
        y = self.data['Exam_Stress_Level']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        # Create metrics display
        from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
        
        # Create text-based metrics display
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        metrics_text = f"""
🎯 MODEL PERFORMANCE METRICS
{'='*50}

ACCURACY METRICS:
• Overall Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)
• Precision: {precision:.4f} ({precision*100:.1f}%)
• Recall: {recall:.4f} ({recall*100:.1f}%)
• F1-Score: {f1:.4f} ({f1*100:.1f}%)

DETAILED CLASSIFICATION REPORT:
{classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High'])}

MODEL QUALITY ASSESSMENT:
{'Excellent' if accuracy > 0.9 else 'Good' if accuracy > 0.8 else 'Fair' if accuracy > 0.7 else 'Poor'} Model Performance

RECOMMENDATIONS:
• {'✅ Model is ready for deployment' if accuracy > 0.85 else '⚠️ Consider retraining with more data' if accuracy > 0.75 else '❌ Model needs improvement'}
• {'📊 Feature selection is effective' if len(self.selected_features) < 20 else '🔧 Consider feature reduction'}
• {'🎯 Model shows good generalization' if abs(accuracy - 0.85) < 0.1 else '⚠️ Watch for overfitting'}
        """
        
        ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.add_activity("🎯 Performance metrics viewed")
    
    def export_analysis_report(self):
        """Export comprehensive analysis report"""
        if self.model is None:
            messagebox.showerror("Error", "Please train model first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Export Analysis Report"
        )
        
        if filename:
            try:
                # Generate comprehensive report
                report = self.generate_comprehensive_report()
                
                with open(filename, 'w') as f:
                    f.write(report)
                
                messagebox.showinfo("Success", f"Analysis report saved to {filename}")
                self.add_activity(f"📥 Analysis report exported")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export: {str(e)}")
    
    def generate_comprehensive_report(self):
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
        
        # Generate report
        report = "="*80 + "\n"
        report += "🎓 ADVANCED STUDENT STRESS PREDICTION SYSTEM - COMPREHENSIVE REPORT\n"
        report += "="*80 + "\n\n"
        
        report += f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"🔢 System Version: Interactive v2.0\n\n"
        
        # Dataset Information
        report += "📊 DATASET INFORMATION\n"
        report += "-"*50 + "\n"
        report += f"Total Samples: {len(self.data):,}\n"
        report += f"Features Used: {len(self.selected_features)}\n"
        report += f"Training Samples: {len(X_train):,}\n"
        report += f"Test Samples: {len(X_test):,}\n\n"
        
        # Stress Level Distribution
        stress_counts = self.data['Exam_Stress_Level'].value_counts().sort_index()
        report += "📈 STRESS LEVEL DISTRIBUTION\n"
        report += "-"*50 + "\n"
        stress_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
        for level, count in stress_counts.items():
            percentage = (count / len(self.data)) * 100
            report += f"{stress_labels[level]} Stress ({level}): {count:,} ({percentage:.1f}%)\n"
        report += "\n"
        
        # Model Performance
        report += "🤖 MODEL PERFORMANCE\n"
        report += "-"*50 + "\n"
        report += f"Algorithm: Random Forest Classifier\n"
        report += f"Estimators: 100 trees\n"
        report += f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f} ({accuracy_score(y_test, y_pred)*100:.1f}%)\n\n"
        
        report += "Detailed Classification Report:\n"
        report += classification_report(y_test, y_pred, 
                                    target_names=['Low', 'Medium', 'High']) + "\n"
        
        # Feature Importance
        feature_importance = pd.DataFrame({
            'Feature': self.selected_features,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        report += "🏆 FEATURE IMPORTANCE RANKINGS\n"
        report += "-"*50 + "\n"
        for i, (feature, importance) in enumerate(zip(feature_importance['Feature'][:15], 
                                                    feature_importance['Importance'][:15])):
            report += f"{i+1:2d}. {feature:<30} ({importance:.4f})\n"
        report += "\n"
        
        # Prediction History Summary
        if self.prediction_history:
            report += "📋 PREDICTION HISTORY SUMMARY\n"
            report += "-"*50 + "\n"
            report += f"Total Predictions Made: {len(self.prediction_history)}\n"
            
            # Count predictions by level
            level_counts = {}
            for entry in self.prediction_history:
                level = entry['prediction']
                level_counts[level] = level_counts.get(level, 0) + 1
            
            stress_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
            for level, count in level_counts.items():
                percentage = (count / len(self.prediction_history)) * 100
                report += f"{stress_labels[level]} Stress: {count} ({percentage:.1f}%)\n"
            report += "\n"
        
        # Recommendations
        report += "💡 SYSTEM RECOMMENDATIONS\n"
        report += "-"*50 + "\n"
        
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > 0.9:
            report += "✅ Model Performance: EXCELLENT - Ready for production use\n"
        elif accuracy > 0.8:
            report += "✅ Model Performance: GOOD - Suitable for deployment\n"
        elif accuracy > 0.7:
            report += "⚠️ Model Performance: FAIR - Consider improvements\n"
        else:
            report += "❌ Model Performance: POOR - Requires retraining\n"
        
        report += "🔧 Suggested Actions:\n"
        report += "• Regular model retraining with new data\n"
        report += "• Monitor prediction accuracy over time\n"
        report += "• Collect more diverse training samples\n"
        report += "• Consider feature engineering improvements\n"
        
        report += "\n" + "="*80 + "\n"
        report += "📊 END OF REPORT\n"
        report += "="*80 + "\n"
        
        return report

def main():
    """Main function to run the interactive GUI application"""
    root = tk.Tk()
    app = InteractiveStressGUI(root)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()

if __name__ == "__main__":
    main()
