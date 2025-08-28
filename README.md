# Sensor Data Classification Pipeline

A comprehensive machine learning pipeline for classifying human motion patterns using sensor data. The system supports multiple classification algorithms, automated feature selection, hyperparameter optimization, and mobile deployment through TensorFlow Lite conversion.

## 🚀 Features

- **Multi-Algorithm Support**: Decision Tree, Random Forest, LightGBM, Neural Network
- **Automated Feature Engineering**: Rolling statistics, magnitude, and rate features
- **Feature Selection**: Recursive Feature Elimination with Cross-Validation (RFECV)
- **Hyperparameter Optimization**: Grid search with cross-validation
- **Mobile Deployment**: TensorFlow Lite model conversion for Android
- **Comprehensive Evaluation**: Per-class metrics, confusion matrices, and performance analysis

### Prerequisites

```bash
Python 3.8+
```

### Required Dependencies

```bash
pip install pandas numpy scikit-learn lightgbm tensorflow matplotlib seaborn joblib
```
## 📁 Output Files

The pipeline generates a timestamped results directory with the following files:

```
model_results_YYYYMMDD_HHMMSS/
├── best_model.pkl                    # Best performing model
├── feature_selector.pkl              # Feature selection transformer
├── scaler.pkl                        # Data scaler (if needed)
├── test_predictions.csv              # Predictions on test set
├── results_summary.json              # Comprehensive results
├── [ModelName]_model.tflite         # TensorFlow Lite model
└── confusion_matrix_[ModelName].png  # Confusion matrices
```
### Results Summary Structure

```json
{
  "timestamp": "20250828_143022",
  "best_model_name": "Random Forest",
  "best_model_score": 0.9456,
  "model_comparison": {
    "Random Forest": {
      "accuracy": 0.9456,
      "auc_score": 0.9678,
      "model_size_kb": 145.2,
      "needs_scaling": false
    }
  },
  "files_created": {
    "predictions": "test_predictions.csv",
    "best_model": "best_model.pkl",
    "tflite_model": "Random_Forest_model.tflite",
    "tflite_size_kb": 89.3
  }
}
```
