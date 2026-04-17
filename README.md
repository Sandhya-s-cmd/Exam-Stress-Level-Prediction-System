# Exam Stress Level Prediction System

A machine learning system that predicts exam stress levels based on academic behaviors and provides personalized counseling recommendations.

## Features

- **Stress Level Prediction**: Predicts stress levels as Low, Medium, or High
- **Personalized Counseling**: Provides tailored advice based on predicted stress level
- **Interactive Interface**: User-friendly input system for real-time predictions
- **Comprehensive Analysis**: Feature importance and model evaluation metrics

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start with Sample Data

```python
from stress_prediction import ExamStressPredictor

# Initialize the predictor
predictor = ExamStressPredictor()

# Train with sample data (automatically created)
predictor.train_from_file('sample_stress_data.csv')

# Run interactive prediction session
predictor.run_interactive_session()
```

### Using Your Own Dataset

1. Prepare your dataset with columns for academic behaviors and a target column for stress levels
2. Train the model:

```python
predictor = ExamStressPredictor()
predictor.train_from_file('your_dataset.csv', target_column='stress_level')
predictor.run_interactive_session()
```

## Dataset Requirements

Your dataset should include:
- **Features**: Academic behavior metrics (study hours, sleep, exercise, anxiety levels, etc.)
- **Target**: Stress level column (values: 'low', 'medium', 'high')

### Sample Features
- `study_hours_per_day`: Number of study hours per day
- `sleep_hours`: Hours of sleep per night
- `exercise_minutes`: Minutes of exercise per day
- `exam_anxiety`: Level of exam anxiety (low/medium/high)
- `peer_pressure`: Peer pressure level (low/medium/high)
- `time_management`: Time management quality (poor/average/good)
- `social_media_hours`: Hours spent on social media
- `caffeine_intake`: Daily caffeine intake
- `previous_gpa`: Previous academic performance

## Counseling Recommendations

### Low Stress
- Encouragement to maintain current habits
- Suggestions to share successful strategies

### Medium Stress
- Stress management techniques
- Study-life balance recommendations
- Mindfulness and relaxation exercises

### High Stress
- Immediate intervention strategies
- Professional help recommendations
- Intensive stress reduction techniques

## Model Performance

The system uses Random Forest Classifier with:
- Feature scaling for optimal performance
- Cross-validation for reliable evaluation
- Feature importance analysis for insights

## File Structure

```
projectts/
├── stress_prediction.py    # Main prediction system
├── requirements.txt        # Required packages
├── README.md              # This file
└── sample_stress_data.csv # Sample dataset (auto-generated)
```

## Running the System

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run the main script**: `python stress_prediction.py`
3. **Follow the interactive prompts** to:
   - Train the model on sample or custom data
   - Input user academic behaviors
   - Receive stress level prediction and counseling

## Customization

You can easily customize:
- Counseling advice by modifying the `counseling_advice` dictionary
- Model parameters in the `train_model` method
- Input validation in the `get_user_input` method

## Example Session

```
EXAM STRESS LEVEL PREDICTION
==================================================

Please enter your academic behavior details:
Enter study hours per day: 7
Enter sleep hours: 6
Enter exercise minutes: 20
Enter exam anxiety: medium
Enter peer pressure: low
...

Predicted Stress Level: MEDIUM

Counseling Advice:
• Your stress level is moderate. Consider implementing these strategies:
• Take regular breaks using the Pomodoro technique (25 min study, 5 min break)
• Practice deep breathing exercises for 5 minutes daily
• Ensure you're getting 7-8 hours of sleep
• Try mindfulness meditation for 10 minutes before studying
```

## Technical Details

- **Algorithm**: Random Forest Classifier
- **Preprocessing**: StandardScaler, LabelEncoder
- **Evaluation**: Accuracy, Classification Report, Confusion Matrix
- **Visualization**: Feature importance plots, confusion matrix heatmap

