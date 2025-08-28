import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
from datetime import datetime
import warnings
import tensorflow as tf
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings('ignore')

class ModelPipeline:
    
    def __init__(self, path_list=None, window_size=50):
        self.window_size = window_size
        self.path_list = path_list
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.best_needs_scaling = False
        self.best_score = 0
        self.feature_selector = None
        self.results = {}
        self.data = self._load_data()
        
    def _load_data(self):
        if not self.path_list:
            return "Please provide a list of data paths"
        
        final_df = pd.DataFrame()
        for index, path in enumerate(self.path_list):
            try:
                df = pd.read_csv(path, sep=',', header=None)
                df = df.map(lambda x: float(str(x).split(":")[-1]) if isinstance(x, str) else x)
                df.columns = ["Rotation", "Roll", "Pitch", "Yaw", "Timestamp"]
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
                df.index = df['Timestamp'].values
                df.drop(columns=['Timestamp'], inplace=True)
                df['label'] = index
                final_df = pd.concat([final_df, df.sort_index()])
                print(f"Loaded data from {path}: {len(df)} samples")
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        print(f"Total data loaded: {len(final_df)} samples, {len(final_df['label'].unique())} classes")
        return final_df
    
    def _create_features(self, df):
        """Create additional features from sensor data"""
        feature_df = df.copy()
        
        # Rolling statistics
        for col in ['Rotation', 'Roll', 'Pitch', 'Yaw']:
            feature_df[f'{col}_mean'] = df[col].rolling(window=5).mean()
            feature_df[f'{col}_std'] = df[col].rolling(window=5).std()
            feature_df[f'{col}_min'] = df[col].rolling(window=5).min()
            feature_df[f'{col}_max'] = df[col].rolling(window=5).max()
        
        # Magnitude features
        feature_df['magnitude'] = np.sqrt(df['Roll']**2 + df['Pitch']**2 + df['Yaw']**2)
        feature_df['rotation_rate'] = df['Rotation'].diff().abs()
        
        # Drop rows with NaN values created by rolling operations
        feature_df.dropna(inplace=True)
        
        return feature_df
    
    def _get_train_test_data(self, test_size=0.2):
        train_lst = []
        test_lst = []
        
        # Create features first
        featured_data = self._create_features(self.data)
        
        for index in featured_data['label'].unique():
            temp_df = featured_data[featured_data["label"] == index]
            train, test = train_test_split(temp_df, test_size=test_size, random_state=42)
            train_lst.append(train)
            test_lst.append(test)

        train_df = pd.concat(train_lst)
        test_df = pd.concat(test_lst)
        
        print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
        return train_df, test_df
    
    def _feature_selection(self, train_df):
        X = train_df.drop(columns=['label'])  
        y = train_df['label']  
        
        estimator = DecisionTreeClassifier(max_depth=10, min_samples_split=20, random_state=42)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  
        
        print("Performing feature selection...")
        selector = RFECV(estimator=estimator, step=1, cv=cv, scoring='accuracy', 
                        min_features_to_select=4, n_jobs=-1)
        selector.fit(X, y)
        
        selected_features = X.columns[selector.support_]
        print(f"Selected {len(selected_features)} features: {list(selected_features)}")
        
        cv_results = pd.DataFrame({
            'n_features': range(1, len(selector.cv_results_['mean_test_score']) + 1),
            'mean_test_score': selector.cv_results_['mean_test_score'],
            'std_test_score': selector.cv_results_['std_test_score']
        })
        
        return cv_results, selector
    
    def _quantize_model(self, model):
        try:
            # For tree-based models, we can reduce the number of estimators or depth
            if hasattr(model, 'n_estimators'):
                model.n_estimators = min(model.n_estimators, 20)  # Reduce ensemble size
            if hasattr(model, 'max_depth'):
                model.max_depth = min(model.max_depth or 10, 8)  # Reduce depth
            return model
        except:
            return model
    
    def train_test_model(self, X_train, X_test, y_train, y_test):
        models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=50, max_depth=10),
            'LightGBM': lgb.LGBMClassifier(random_state=42, n_estimators=50, verbose=-1),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        }
        
        # Models that don't need scaling (tree-based)
        tree_based_models = {'Decision Tree', 'Random Forest', 'LightGBM'}
        
        model_results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Apply scaling only for non-tree-based models
            if name in tree_based_models:
                X_train_processed = X_train
                X_test_processed = X_test
            else:
                X_train_processed = self.scaler.fit_transform(X_train)
                X_test_processed = self.scaler.transform(X_test)
            
            # Train model
            model.fit(X_train_processed, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_processed)
            y_pred_proba = model.predict_proba(X_test_processed)
            
            # Calculate metrics
            accuracy = model.score(X_test_processed, y_test)
            
            try:
                auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            except:
                auc_score = 0.0
            
            # Calculate per-class metrics
            precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
            class_names = [f'Class_{i}' for i in range(len(precision))]
            
            per_class_metrics = {}
            for i, class_name in enumerate(class_names):
                per_class_metrics[class_name] = {
                    'precision': precision[i],
                    'recall': recall[i],
                    'f1_score': f1[i],
                    'support': support[i]
                }
            
            model_results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc_score': auc_score,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'model_size': self._get_model_size(model),
                'per_class_metrics': per_class_metrics,
                'needs_scaling': name not in tree_based_models
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
            
            # Update best model
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = model
                self.best_model_name = name
                self.best_needs_scaling = name not in tree_based_models
        
        return model_results
    
    def _get_model_size(self, model):
        """Estimate model size in KB"""
        try:
            import pickle
            return len(pickle.dumps(model)) / 1024  # Size in KB
        except:
            return 0
    
    def model_evaluation(self, y_true, y_pred, y_pred_proba, model_name, per_class_metrics=None):
        print(f"\n=== {model_name} Evaluation ===")
        
        # Classification Report
        report = classification_report(y_true, y_pred, output_dict=True)
        print("Classification Report:")
        print(classification_report(y_true, y_pred))
        
        # Print per-class metrics in a formatted way
        if per_class_metrics:
            print("\nDetailed Per-Class Metrics:")
            print("-" * 60)
            print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
            print("-" * 60)
            for class_name, metrics in per_class_metrics.items():
                print(f"{class_name:<12} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                      f"{metrics['f1_score']:<12.4f} {metrics['support']:<12}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return report, cm
    
    def parameter_optimization(self, X_train, X_test, y_train, y_test):
        print("\nPerforming parameter optimization...")
        
        param_grids = {
            'Random Forest': {
                'n_estimators': [20, 50],
                'max_depth': [5, 10, 15],
                'min_samples_split': [5, 10]
            },
            'Decision Tree': {
                'max_depth': [5, 10, 15, 20],
                'min_samples_split': [5, 10, 20],
                'min_samples_leaf': [2, 5]
            },
            'Neural Network': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'alpha': [0.001, 0.01, 0.1]
            }
        }
        
        # Models that don't need scaling
        tree_based_models = {'Decision Tree', 'Random Forest', 'LightGBM'}
        
        best_models = {}
        
        for model_name, param_grid in param_grids.items():
            print(f"Optimizing {model_name}...")
            
            if model_name == 'Random Forest':
                base_model = RandomForestClassifier(random_state=42)
            elif model_name == 'Neural Network':
                base_model = MLPClassifier(random_state=42, max_iter=500)
            else:
                base_model = DecisionTreeClassifier(random_state=42)
            
            grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
            
            # Apply scaling only for non-tree-based models
            if model_name in tree_based_models:
                X_train_processed = X_train
                X_test_processed = X_test
            else:
                X_train_processed = self.scaler.fit_transform(X_train)
                X_test_processed = self.scaler.transform(X_test)
            
            grid_search.fit(X_train_processed, y_train)
            best_score = grid_search.score(X_test_processed, y_test)
            
            best_models[model_name] = {
                'model': grid_search.best_estimator_,
                'params': grid_search.best_params_,
                'score': best_score
            }
            
            print(f"Best {model_name} params: {grid_search.best_params_}")
            print(f"Best {model_name} score: {best_score:.4f}")
        
        return best_models
    
    def _convert_to_tflite(self, model, X_sample, model_name, results_dir):
        """Convert model to TensorFlow Lite for Android deployment"""
        print(f"\nConverting {model_name} to TensorFlow Lite...")
        
        try:
            # Get predictions from the original model for the training data
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_sample)
            else:
                # For models without predict_proba, create one-hot encoded predictions
                y_pred = model.predict(X_sample)
                y_proba = np.eye(len(np.unique(y_pred)))[y_pred]
            
            # Create a simple neural network to approximate the original model
            tf_model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(X_sample.shape[1],)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(y_proba.shape[1], activation='softmax')
            ])
            
            tf_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
            # Train the TensorFlow model to mimic the original model
            print("Training TensorFlow model to mimic the original model...")
            tf_model.fit(X_sample, y_proba, epochs=50, batch_size=32, verbose=0, validation_split=0.2)
            
            # Convert to TensorFlow Lite
            converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Additional optimizations for mobile deployment
            converter.target_spec.supported_types = [tf.float16]  # Use float16 for smaller size
            
            tflite_model = converter.convert()
            
            # Save the TFLite model
            tflite_path = os.path.join(results_dir, f'{model_name.replace(" ", "_")}_model.tflite')
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            tflite_size = len(tflite_model) / 1024  # Size in KB
            print(f"TensorFlow Lite model saved: {tflite_path}")
            print(f"TFLite model size: {tflite_size:.2f} KB")
            
            # Test the TFLite model
            interpreter = tf.lite.Interpreter(model_content=tflite_model)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Test with a sample
            test_sample = X_sample[:1].astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], test_sample)
            interpreter.invoke()
            tflite_prediction = interpreter.get_tensor(output_details[0]['index'])
            
            print(f"TFLite model test successful. Output shape: {tflite_prediction.shape}")
            
            return tflite_path, tflite_size
            
        except Exception as e:
            print(f"Error converting to TensorFlow Lite: {str(e)}")
            return None, 0
    
    def _save_predictions(self, y_test, model_results, results_dir):
        predictions_df = pd.DataFrame()
        predictions_df['actual'] = y_test.values
        
        # Add predictions from all models
        for model_name, results in model_results.items():
            predictions_df[f'{model_name.replace(" ", "_")}_predicted'] = results['y_pred']
            
            # Add prediction probabilities for each class
            y_proba = results['y_pred_proba']
            for class_idx in range(y_proba.shape[1]):
                predictions_df[f'{model_name.replace(" ", "_")}_prob_class_{class_idx}'] = y_proba[:, class_idx]
        
        # Save to CSV
        predictions_path = os.path.join(results_dir, 'test_predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Test predictions saved: {predictions_path}")
        
        return predictions_path
    
    def save_results(self, model_results, optimized_models, y_test, X_test_selected):
        """Save all results and models"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f'model_results_{timestamp}'
        os.makedirs(results_dir, exist_ok=True)
        
        # Save test predictions
        predictions_path = self._save_predictions(y_test, model_results, results_dir)
        
        # Save best model
        tflite_path, tflite_size = None, 0
        if self.best_model:
            quantized_model = self._quantize_model(self.best_model)
            model_path = os.path.join(results_dir, 'best_model.pkl')
            joblib.dump(quantized_model, model_path)
            
            model_size = os.path.getsize(model_path) / 1024  # Size in KB
            print(f"Model saved: {model_path} (Size: {model_size:.2f} KB)")
            
            # Convert to TensorFlow Lite for Android
            tflite_path, tflite_size = self._convert_to_tflite(
                self.best_model, X_test_selected, self.best_model_name, results_dir
            )
        
        # Save feature selector
        if self.feature_selector:
            joblib.dump(self.feature_selector, os.path.join(results_dir, 'feature_selector.pkl'))
        
        # Save scaler (only if best model needs scaling)
        if self.best_needs_scaling:
            joblib.dump(self.scaler, os.path.join(results_dir, 'scaler.pkl'))
            print("Scaler saved (required for best model)")
        else:
            print("Scaler not saved (best model doesn't require scaling)")
        
        # Create detailed results summary with per-class metrics
        model_comparison = {}
        for name, results in model_results.items():
            model_comparison[name] = {
                'accuracy': results['accuracy'],
                'auc_score': results['auc_score'],
                'model_size_kb': results['model_size'],
                'needs_scaling': results['needs_scaling'],
                'per_class_metrics': results['per_class_metrics']
            }
        
        summary = {
            'timestamp': timestamp,
            'best_model_name': self.best_model_name,
            'best_model_score': self.best_score,
            'best_model_needs_scaling': self.best_needs_scaling,
            'model_comparison': model_comparison,
            'optimized_models': {name: {'params': results['params'], 
                                      'score': results['score']} 
                               for name, results in optimized_models.items()},
            'files_created': {
                'predictions': 'test_predictions.csv',
                'best_model': 'best_model.pkl',
                'feature_selector': 'feature_selector.pkl',
                'scaler': 'scaler.pkl' if self.best_needs_scaling else None,
                'tflite_model': os.path.basename(tflite_path) if tflite_path else None,
                'tflite_size_kb': tflite_size
            }
        }
        
        with open(os.path.join(results_dir, 'results_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)  # Added default=str to handle numpy types
        
        print(f"Results saved in directory: {results_dir}")
        return results_dir
    
    def run_pipeline(self):
        """Run the complete pipeline"""
        print("Starting Model Training Pipeline...")
        print("=" * 50)
        
        # Step 1: Load and split data
        train_df, test_df = self._get_train_test_data()
        
        # Step 2: Feature selection
        cv_results, self.feature_selector = self._feature_selection(train_df)
        
        # Apply feature selection
        X_train = train_df.drop(columns=['label'])
        X_test = test_df.drop(columns=['label'])
        y_train = train_df['label']
        y_test = test_df['label']
        
        X_train_selected = self.feature_selector.transform(X_train)
        X_test_selected = self.feature_selector.transform(X_test)
        
        # Step 3: Train and test models
        model_results = self.train_test_model(X_train_selected, X_test_selected, y_train, y_test)
        
        # Step 4: Model evaluation for each model
        for name, results in model_results.items():
            self.model_evaluation(y_test, results['y_pred'], results['y_pred_proba'], 
                                name, results['per_class_metrics'])
        
        # Step 5: Parameter optimization
        optimized_models = self.parameter_optimization(X_train_selected, X_test_selected, y_train, y_test)
        
        # Step 6: Save results
        results_dir = self.save_results(model_results, optimized_models, y_test, X_test_selected)
        
        print("\n" + "=" * 50)
        print("Pipeline completed successfully!")
        print(f"Best model: {self.best_model_name}")
        print(f"Best model accuracy: {self.best_score:.4f}")
        print(f"Results saved in: {results_dir}")
        
        return model_results, optimized_models

if __name__ == '__main__':
    paths = [
        '../sensor_data/still_on_desk.txt',
        '../sensor_data/still_on_desk_perform_action.txt',
        '../sensor_data/stand_still.txt',
        '../sensor_data/moving.txt',
    ]
    
    # Initialize and run pipeline
    pipeline = ModelPipeline(path_list=paths, window_size=50)
    
    # Check if data loaded successfully
    if isinstance(pipeline.data, str):
        print(pipeline.data)
    else:
        model_results, optimized_models = pipeline.run_pipeline()
        
        # Print final summary
        print("\n=== Final Model Comparison ===")
        for name, results in model_results.items():
            print(f"{name}: Accuracy={results['accuracy']:.4f}, "
                  f"AUC={results['auc_score']:.4f}, "
                  f"Size={results['model_size']:.2f}KB")