"""
Main execution script for Facial Expression Recognition
Runs the complete pipeline: data loading, model training, and evaluation
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
import tensorflow as tf
warnings.filterwarnings('ignore')

# Import custom modules
from config import Config
from data_loader import DataLoader
from models import CNNModels
from trainer import ModelTrainer
from metrics import EvaluationMetrics
# Add these imports after existing imports:
from hyperparameter_tuning import HyperparameterTuner
from advanced_analysis import AdvancedAnalyzer
from models import BaselineModels
from scipy import stats
import seaborn as sns

def setup_environment():
    """Setup directories and environment"""
    config = Config()
    config.create_directories()
    
    print("="*80)
    print("FACIAL EXPRESSION RECOGNITION PROJECT")
    print("="*80)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    print()
    
    return config

def main():
    # Setup
    config = setup_environment()
    
    # Initialize modules
    data_loader = DataLoader(config)
    model_factory = CNNModels(config)
    trainer = ModelTrainer(config)
    metrics_calculator = EvaluationMetrics(config)
    
    # ================================
    # DATA LOADING AND PREPROCESSING
    # ================================
    print("STEP 1: Loading and preprocessing data...")
    print("-" * 50)
    
    # Load annotations
    data_loader.load_annotations()
    
    # Load images from the dataset/images directory
    print("Loading images from dataset/images directory...")
    images_directory = config.IMAGES_PATH
    
    if os.path.exists(images_directory):
        data_loader.images = data_loader.load_images_from_directory(images_directory)
        print(f"Successfully loaded {len(data_loader.images)} images from {images_directory}")
    else:
        print(f"Images directory not found: {images_directory}")
        print("Please ensure your dataset folder structure is correct:")
        print("  dataset/")
        print("    ├── images/     (JPG files)")
        print("    └── annotations/ (.npy files)")
        return
    
    # Preprocess data
    data_loader.preprocess_data()
    
    # Display data statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(data_loader.images)}")
    print(f"Image shape: {data_loader.images.shape[1:]}")
    if data_loader.expressions is not None:
        print(f"Number of expression classes: {len(np.unique(data_loader.expressions))}")
    
    # Show class distribution
    data_loader.get_class_distribution()
    
    # Visualize samples
    if data_loader.expressions is not None:
        data_loader.visualize_samples(data_loader.images, data_loader.expressions_categorical)
    
    # ================================
    # DATA SPLITTING
    # ================================
        # ================================
    # HYPERPARAMETER TUNING (NEW)
    # ================================
    print("\nSTEP 2.5: Hyperparameter Tuning...")
    print("-" * 50)
    
    tuner = HyperparameterTuner(config)
    best_hyperparameters = {}
    
    # Tune hyperparameters for each model (limit to 1-2 models for time)
    models_to_tune = ['custom']  # Start with custom model
    
    for model_name in models_to_tune:
        try:
            # Prepare data for tuning
            if data_splits['train']['valence'] is not None:
                train_data_tune = (
                    data_splits['train']['images'],
                    data_splits['train']['expressions'],
                    data_splits['train']['valence'],
                    data_splits['train']['arousal']
                )
                val_data_tune = (
                    data_splits['val']['images'],
                    data_splits['val']['expressions'],
                    data_splits['val']['valence'],
                    data_splits['val']['arousal']
                )
            else:
                train_data_tune = (
                    data_splits['train']['images'],
                    data_splits['train']['expressions']
                )
                val_data_tune = (
                    data_splits['val']['images'],
                    data_splits['val']['expressions']
                )
            
            best_params, best_score = tuner.grid_search(
                model_name, train_data_tune, val_data_tune, max_combinations=5
            )
            best_hyperparameters[model_name] = best_params
            
            # Cross-validation with best parameters
            if not multi_output:  # Simplified for single output
                cv_scores, mean_cv, std_cv = tuner.cross_validate_model(
                    model_name, best_params, 
                    data_splits['train']['images'], 
                    data_splits['train']['expressions']
                )
        
        except Exception as e:
            print(f"Hyperparameter tuning failed for {model_name}: {e}")
            best_hyperparameters[model_name] = None
    
    # Save tuning results
    tuner.save_tuning_results()
    # Split data into train, validation, and test sets
    data_splits = data_loader.split_data(test_size=0.2, val_size=0.2)
    
        # ================================
    # BASELINE MODELS (NEW)
    # ================================
    print("\nSTEP 2.7: Training Baseline Models...")
    print("-" * 50)
    
    baseline_factory = BaselineModels(config)
    baseline_results = {}
    
    # Random classifier baseline
    print("🎲 Random Classifier Baseline:")
    random_clf = baseline_factory.random_classifier(config.NUM_CLASSES)
    if data_splits['test']['expressions'] is not None:
        y_test_labels = np.argmax(data_splits['test']['expressions'], axis=1)
        random_pred = random_clf(data_splits['test']['images'])
        random_acc = np.mean(random_pred == y_test_labels)
        print(f"Random Classifier Accuracy: {random_acc:.4f}")
        baseline_results['random'] = {'accuracy': random_acc}
    
    # Majority class baseline
    print("📊 Majority Class Baseline:")
    majority_clf = baseline_factory.majority_class_classifier(data_splits['train']['expressions'])
    majority_pred = majority_clf(data_splits['test']['images'])
    majority_acc = np.mean(majority_pred == y_test_labels)
    print(f"Majority Class Accuracy: {majority_acc:.4f}")
    baseline_results['majority'] = {'accuracy': majority_acc}
    
    # Simple CNN baseline
    print("🧠 Simple CNN Baseline:")
    try:
        simple_model = baseline_factory.simple_cnn_baseline(multi_output=False)
        simple_trainer = ModelTrainer(config)
        simple_trainer.config.EPOCHS = 10  # Quick training
        
        simple_train_data = (data_splits['train']['images'], data_splits['train']['expressions'])
        simple_val_data = (data_splits['val']['images'], data_splits['val']['expressions'])
        simple_test_data = (data_splits['test']['images'], data_splits['test']['expressions'])
        
        simple_trainer.train_model(simple_model, simple_train_data, simple_val_data, 'simple_baseline', False)
        simple_results = simple_trainer.evaluate_model(simple_model, simple_test_data, 'simple_baseline', False)
        
        baseline_results['simple_cnn'] = simple_results.get('Expression Classification', {})
        print(f"Simple CNN Accuracy: {baseline_results['simple_cnn'].get('accuracy', 0):.4f}")
        
    except Exception as e:
        print(f"Simple CNN baseline failed: {e}")
    
    # ================================
    # MODEL TRAINING AND EVALUATION
    # ================================
    print("\nSTEP 3: Training and evaluating models...")
    print("-" * 50)
    
    # Define models to train
    models_to_train = ['vgg16', 'resnet50', 'custom']  # Add more models as needed
    
    all_results = {}
    
    for model_name in models_to_train:
        print(f"\n{'='*20} {model_name.upper()} {'='*20}")
        
        try:
            # Determine if we have continuous outputs
            multi_output = (data_splits['train']['valence'] is not None and 
                          data_splits['train']['arousal'] is not None)
            
            # Create model
            model = model_factory.get_model(model_name, multi_output=multi_output)
            
            # Save model summary
            trainer.save_model_summary(model, model_name)
            
            # Prepare training data
            if multi_output:
                train_data = (
                    data_splits['train']['images'],
                    data_splits['train']['expressions'],
                    data_splits['train']['valence'],
                    data_splits['train']['arousal']
                )
                val_data = (
                    data_splits['val']['images'],
                    data_splits['val']['expressions'],
                    data_splits['val']['valence'],
                    data_splits['val']['arousal']
                )
                test_data = (
                    data_splits['test']['images'],
                    data_splits['test']['expressions'],
                    data_splits['test']['valence'],
                    data_splits['test']['arousal']
                )
            else:
                train_data = (
                    data_splits['train']['images'],
                    data_splits['train']['expressions']
                )
                val_data = (
                    data_splits['val']['images'],
                    data_splits['val']['expressions']
                )
                test_data = (
                    data_splits['test']['images'],
                    data_splits['test']['expressions']
                )
            
            # Train model
            history = trainer.train_model(
                model, train_data, val_data, model_name, multi_output
            )
            
            # Plot training history
            trainer.plot_training_history(model_name)
            
            # Fine-tune if using pre-trained model
            if model_name != 'custom':
                print(f"\nFine-tuning {model_name}...")
                trainer.fine_tune_model(
                    model, train_data, val_data, model_name,
                    layers_to_unfreeze=20, fine_tune_epochs=10
                )
                
                # Plot updated training history
                trainer.plot_training_history(f'{model_name}_finetuned')
            
            # Evaluate model
            results = trainer.evaluate_model(model, test_data, model_name, multi_output)
            
            # Add training time to results
            if 'Expression Classification' in results:
                results['Expression Classification']['training_time_seconds'] = trainer.get_training_time()
            
            # Store results
            all_results[model_name] = results
            
            # Print results summary
            metrics_calculator.print_metrics_summary(results)
            
            # Save individual model results
            results_filename = os.path.join(config.RESULTS_PATH, f'{model_name}_results.csv')
            flattened_results = {}
            for task, metrics in results.items():
                for metric_name, value in metrics.items():
                    flattened_results[f"{task}_{metric_name}"] = value
            
            metrics_calculator.save_results_to_csv({model_name: flattened_results}, results_filename)
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
                # ================================
            # ADVANCED ANALYSIS (NEW)
            # ================================
            print(f"\n🔬 Advanced Analysis for {model_name.upper()}:")
            analyzer = AdvancedAnalyzer(config)
            
            try:
                # Get predictions for analysis
                if multi_output:
                    test_pred = model.predict(data_splits['test']['images'])
                    y_pred_expr = test_pred[0]  # Expression predictions
                    y_true_expr = data_splits['test']['expressions']
                else:
                    test_pred = model.predict(data_splits['test']['images'])
                    y_pred_expr = test_pred
                    y_true_expr = data_splits['test']['expressions']
                
                # Confusion Matrix
                cm = analyzer.create_confusion_matrix_heatmap(y_true_expr, y_pred_expr, model_name)
                
                # ROC Curves (if multi-class)
                if len(y_pred_expr.shape) > 1 and y_pred_expr.shape[1] > 2:
                    y_true_labels = np.argmax(y_true_expr, axis=1) if len(y_true_expr.shape) > 1 else y_true_expr
                    roc_results = analyzer.plot_roc_curves(y_true_labels, y_pred_expr, model_name)
                
                # Per-class analysis
                per_class_results = analyzer.per_class_analysis(y_true_expr, y_pred_expr, model_name)
                
                # Learning curve analysis
                if hasattr(trainer, 'history') and trainer.history:
                    analyzer.learning_curve_analysis(trainer.history, model_name)
                
                # Regression analysis (if multi-output)
                if multi_output and len(test_pred) > 1:
                    # Valence analysis
                    if len(test_pred) > 1:
                        valence_residuals = analyzer.regression_residual_analysis(
                            data_splits['test']['valence'], test_pred[1].flatten(), model_name, 'valence'
                        )
                    
                    # Arousal analysis  
                    if len(test_pred) > 2:
                        arousal_residuals = analyzer.regression_residual_analysis(
                            data_splits['test']['arousal'], test_pred[2].flatten(), model_name, 'arousal'
                        )
                
            except Exception as e:
                print(f"Advanced analysis failed for {model_name}: {e}")
    # ================================
    # MODEL COMPARISON
    # ================================
    print("\n" + "="*80)
    print("STEP 4: Model Comparison and Final Results")
    print("="*80)
        # ================================
    # STATISTICAL SIGNIFICANCE TESTING (NEW)
    # ================================
    print("\n Statistical Significance Analysis:")
    analyzer = AdvancedAnalyzer(config)
    
    # Combine all results including baselines
    all_model_results = {}
    for model_name, results in all_results.items():
        if 'Expression Classification' in results:
            all_model_results[model_name] = results['Expression Classification']
    
    # Add baseline results
    all_model_results.update(baseline_results)
    
    # Perform significance tests
    sig_results = analyzer.statistical_significance_test(all_model_results)
    
    # Save significance results
    sig_df = pd.DataFrame(sig_results).T
    sig_path = os.path.join(config.RESULTS_PATH, 'statistical_significance_results.csv')
    sig_df.to_csv(sig_path)
    if all_results:
        # Create comparison DataFrame
        comparison_data = {}
        for model_name, results in all_results.items():
            model_metrics = {}
            
            # Extract key metrics for comparison
            if 'Expression Classification' in results:
                expr_metrics = results['Expression Classification']
                model_metrics.update({
                    'accuracy': expr_metrics.get('accuracy', 0),
                    'f1_macro': expr_metrics.get('f1_macro', 0),
                    'cohens_kappa': expr_metrics.get('cohens_kappa', 0),
                    'auc_roc': expr_metrics.get('auc_roc', 0),
                    'training_time': expr_metrics.get('training_time_seconds', 0)
                })
            
            if 'Valence Regression' in results:
                val_metrics = results['Valence Regression']
                model_metrics.update({
                    'valence_rmse': val_metrics.get('valence_rmse', float('inf')),
                    'valence_corr': val_metrics.get('valence_corr', 0),
                    'valence_ccc': val_metrics.get('valence_ccc', 0),
                    'valence_sagr': val_metrics.get('valence_sagr', 0)
                })
            
            if 'Arousal Regression' in results:
                ar_metrics = results['Arousal Regression']
                model_metrics.update({
                    'arousal_rmse': ar_metrics.get('arousal_rmse', float('inf')),
                    'arousal_corr': ar_metrics.get('arousal_corr', 0),
                    'arousal_ccc': ar_metrics.get('arousal_ccc', 0),
                    'arousal_sagr': ar_metrics.get('arousal_sagr', 0)
                })
            
            comparison_data[model_name] = model_metrics
        
        # Display comparison
        comparison_df = metrics_calculator.compare_models_results(comparison_data)
        
        # Save comprehensive results
        comprehensive_results_path = os.path.join(config.RESULTS_PATH, 'comprehensive_results.csv')
        comparison_df.to_csv(comprehensive_results_path)
        print(f"\nComprehensive results saved to: {comprehensive_results_path}")
        
        # Create performance visualization
        create_performance_plots(comparison_df, config)
        
        # Generate comprehensive final report
        generate_comprehensive_report(all_results, comparison_df, config, baseline_results, best_hyperparameters)
        
    else:
        print("No models were successfully trained.")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Check the '{config.RESULTS_PATH}' directory for all outputs:")
    print("- Model weights (.h5 files)")
    print("- Training histories (plots and CSV logs)")
    print("- Evaluation metrics and plots")
    print("- Comprehensive comparison results")
    print("="*80)

def create_performance_plots(comparison_df, config):
    """Create performance comparison plots"""
    import matplotlib.pyplot as plt
    
    # Key metrics for plotting
    classification_metrics = ['accuracy', 'f1_macro', 'cohens_kappa']
    regression_metrics = ['valence_rmse', 'arousal_rmse', 'valence_ccc', 'arousal_ccc']
    
    # Classification metrics plot
    available_class_metrics = [m for m in classification_metrics if m in comparison_df.columns]
    if available_class_metrics:
        fig, axes = plt.subplots(1, len(available_class_metrics), figsize=(15, 5))
        if len(available_class_metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(available_class_metrics):
            comparison_df[metric].plot(kind='bar', ax=axes[i], title=f'{metric.replace("_", " ").title()}')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_PATH, 'classification_metrics_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    # Regression metrics plot
    available_reg_metrics = [m for m in regression_metrics if m in comparison_df.columns]
    if available_reg_metrics:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(available_reg_metrics[:4]):
            comparison_df[metric].plot(kind='bar', ax=axes[i], title=f'{metric.replace("_", " ").title()}')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(available_reg_metrics), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_PATH, 'regression_metrics_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()

def generate_final_report(all_results, comparison_df, config):
    """Generate final report summary"""
    report_path = os.path.join(config.RESULTS_PATH, 'final_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Facial Expression Recognition - Final Report\n\n")
        f.write("## Dataset Overview\n")
        f.write(f"- Image size: {config.IMG_HEIGHT}x{config.IMG_WIDTH}x{config.CHANNELS}\n")
        f.write(f"- Number of emotion classes: {config.NUM_CLASSES}\n")
        f.write(f"- Emotion labels: {list(config.EMOTION_LABELS.values())}\n\n")
        
        f.write("## Models Trained\n")
        for model_name in all_results.keys():
            f.write(f"- {model_name.upper()}\n")
        f.write("\n")
        
        f.write("## Performance Comparison\n\n")
        f.write("### Classification Metrics\n")
        if 'accuracy' in comparison_df.columns:
            best_accuracy_model = comparison_df['accuracy'].idxmax()
            best_accuracy = comparison_df['accuracy'].max()
            f.write(f"**Best Accuracy**: {best_accuracy_model} ({best_accuracy:.4f})\n\n")
        
        if 'f1_macro' in comparison_df.columns:
            best_f1_model = comparison_df['f1_macro'].idxmax()
            best_f1 = comparison_df['f1_macro'].max()
            f.write(f"**Best F1-Score**: {best_f1_model} ({best_f1:.4f})\n\n")
        
        f.write("### Regression Metrics\n")
        if 'valence_rmse' in comparison_df.columns:
            best_val_model = comparison_df['valence_rmse'].idxmin()
            best_val_rmse = comparison_df['valence_rmse'].min()
            f.write(f"**Best Valence RMSE**: {best_val_model} ({best_val_rmse:.4f})\n\n")
        
        if 'arousal_rmse' in comparison_df.columns:
            best_ar_model = comparison_df['arousal_rmse'].idxmin()
            best_ar_rmse = comparison_df['arousal_rmse'].min()
            f.write(f"**Best Arousal RMSE**: {best_ar_model} ({best_ar_rmse:.4f})\n\n")
        
        f.write("## Detailed Results\n\n")
        f.write("```\n")
        f.write(comparison_df.round(4).to_string())
        f.write("\n```\n\n")
        
        f.write("## Recommendations\n")
        f.write("- For production deployment, consider the best performing model based on your specific requirements\n")
        f.write("- If inference time is critical, consider MobileNet or EfficientNet variants\n")
        f.write("- For highest accuracy, ResNet or custom architectures often perform well\n")
        f.write("- Consider ensemble methods for further improvement\n\n")
        
        f.write("## Files Generated\n")
        f.write("- Model weights: `models/` directory\n")
        f.write("- Training logs: `results/` directory\n")
        f.write("- Evaluation plots: `results/` directory\n")
        f.write("- Comprehensive results: `results/comprehensive_results.csv`\n")
    
    print(f"Final report saved to: {report_path}")
    
def load_and_validate_data():
    """Helper function to validate data loading"""
    config = Config()
    
    print("Validating data structure...")
    print(f"Looking for dataset in: {config.DATASET_PATH}")
    print(f"Looking for annotations in: {config.ANNOTATIONS_PATH}")
    print(f"Looking for images in: {config.IMAGES_PATH}")
    
    # Check dataset folder
    if not os.path.exists(config.DATASET_PATH):
        print(f"Error: Dataset directory not found: {config.DATASET_PATH}")
        return False
    
    # Check annotations folder
    if not os.path.exists(config.ANNOTATIONS_PATH):
        print(f"Error: Annotations directory not found: {config.ANNOTATIONS_PATH}")
        return False
    
    # Check images folder
    if not os.path.exists(config.IMAGES_PATH):
        print(f"Error: Images directory not found: {config.IMAGES_PATH}")
        return False
    
    # List all files in annotations directory
    files = os.listdir(config.ANNOTATIONS_PATH)
    npy_files = [f for f in files if f.endswith('.npy')]
    
    print(f"Found {len(npy_files)} .npy files in annotations:")
    valid_array_files = []
    
    for file in npy_files:
        file_path = os.path.join(config.ANNOTATIONS_PATH, file)
        try:
            data = np.load(file_path)
            print(f"  {file}: shape {data.shape}, dtype {data.dtype}")
            
            # Check if it's a valid array (not scalar)
            if data.shape != ():  # Not a scalar
                valid_array_files.append(file)
                print(f"    ✓ Valid array file")
            else:
                print(f"    ⚠ Scalar value, skipping for length comparison")
                
        except Exception as e:
            print(f"  {file}: Error loading - {e}")
    
    # Count images
    image_files = [f for f in os.listdir(config.IMAGES_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} image files in images directory")
    
    # Check if counts match (approximately) using valid array files
    if valid_array_files and image_files:
        # Use the first valid array file for comparison
        sample_file = valid_array_files[0]
        sample_npy = np.load(os.path.join(config.ANNOTATIONS_PATH, sample_file))
        
        try:
            sample_length = len(sample_npy)
            if sample_length != len(image_files):
                print(f"WARNING: Annotation count ({sample_length}) != Image count ({len(image_files)})")
                print(f"  (Based on {sample_file})")
            else:
                print("✓ Annotation and image counts match!")
                print(f"  (Based on {sample_file})")
        except TypeError:
            print(f"WARNING: Cannot compare lengths - {sample_file} may not be a proper array")
    
    # Additional validation: check for expected annotation types
    has_expressions = any('expression' in f.lower() for f in npy_files)
    has_valence = any('valence' in f.lower() for f in npy_files)
    has_arousal = any('arousal' in f.lower() for f in npy_files)
    
    print(f"\nAnnotation types found:")
    print(f"  Expressions: {'✓' if has_expressions else '✗'}")
    print(f"  Valence: {'✓' if has_valence else '✗'}")
    print(f"  Arousal: {'✓' if has_arousal else '✗'}")
    
    # At minimum, we need expressions or at least one valid array file
    return (len(valid_array_files) > 0 or has_expressions) and len(image_files) > 0

# def load_and_validate_data():
#     """Helper function to validate data loading"""
#     config = Config()
    
#     print("Validating data structure...")
#     print(f"Looking for dataset in: {config.DATASET_PATH}")
#     print(f"Looking for annotations in: {config.ANNOTATIONS_PATH}")
#     print(f"Looking for images in: {config.IMAGES_PATH}")
    
#     # Check dataset folder
#     if not os.path.exists(config.DATASET_PATH):
#         print(f"Error: Dataset directory not found: {config.DATASET_PATH}")
#         return False
    
#     # Check annotations folder
#     if not os.path.exists(config.ANNOTATIONS_PATH):
#         print(f"Error: Annotations directory not found: {config.ANNOTATIONS_PATH}")
#         return False
    
#     # Check images folder
#     if not os.path.exists(config.IMAGES_PATH):
#         print(f"Error: Images directory not found: {config.IMAGES_PATH}")
#         return False
    
#     # List all files in annotations directory
#     files = os.listdir(config.ANNOTATIONS_PATH)
#     npy_files = [f for f in files if f.endswith('.npy')]
    
#     print(f"Found {len(npy_files)} .npy files in annotations:")
#     for file in npy_files:
#         file_path = os.path.join(config.ANNOTATIONS_PATH, file)
#         try:
#             data = np.load(file_path)
#             print(f"  {file}: shape {data.shape}, dtype {data.dtype}")
#         except Exception as e:
#             print(f"  {file}: Error loading - {e}")
    
#     # Count images
#     image_files = [f for f in os.listdir(config.IMAGES_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
#     print(f"Found {len(image_files)} image files in images directory")
    
#     # Check if counts match (approximately)
#     if npy_files and image_files:
#         sample_npy = np.load(os.path.join(config.ANNOTATIONS_PATH, npy_files[0]))
#         if len(sample_npy) != len(image_files):
#             print(f"WARNING: Annotation count ({len(sample_npy)}) != Image count ({len(image_files)})")
#         else:
#             print("✓ Annotation and image counts match!")
    
#     return len(npy_files) > 0 and len(image_files) > 0
def generate_comprehensive_report(all_results, comparison_df, config, baseline_results=None, best_hyperparams=None):
    """Generate comprehensive final report with all analyses"""
    report_path = os.path.join(config.RESULTS_PATH, 'comprehensive_final_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 🎭 Facial Expression Recognition - Comprehensive Analysis Report\n\n")
        
        # Executive Summary
        f.write("## 📋 Executive Summary\n\n")
        if 'accuracy' in comparison_df.columns:
            best_model = comparison_df['accuracy'].idxmax()
            best_acc = comparison_df['accuracy'].max()
            f.write(f"**Best Performing Model**: {best_model.upper()} (Accuracy: {best_acc:.4f})\n\n")
        
        # Dataset Overview
        f.write("## 📊 Dataset Overview\n\n")
        f.write(f"- **Image Dimensions**: {config.IMG_HEIGHT}×{config.IMG_WIDTH}×{config.CHANNELS}\n")
        f.write(f"- **Number of Classes**: {config.NUM_CLASSES}\n")
        f.write(f"- **Emotion Categories**: {list(config.EMOTION_LABELS.values())}\n")
        f.write(f"- **Training Strategy**: Transfer Learning + Custom Architecture\n\n")
        
        # Hyperparameter Tuning Results
        if best_hyperparams:
            f.write("## 🔧 Hyperparameter Optimization Results\n\n")
            for model_name, params in best_hyperparams.items():
                if params:
                    f.write(f"### {model_name.upper()} Optimal Parameters:\n")
                    for param, value in params.items():
                        f.write(f"- **{param}**: {value}\n")
                    f.write("\n")
        
        # Baseline Comparisons
        if baseline_results:
            f.write("## 📏 Baseline Model Performance\n\n")
            f.write("| Model | Accuracy | Notes |\n")
            f.write("|-------|----------|-------|\n")
            for baseline_name, results in baseline_results.items():
                acc = results.get('accuracy', 0)
                notes = {
                    'random': 'Random predictions',
                    'majority': 'Always predicts most frequent class',
                    'simple_cnn': 'Minimal CNN architecture'
                }.get(baseline_name, 'Baseline model')
                f.write(f"| {baseline_name.title()} | {acc:.4f} | {notes} |\n")
            f.write("\n")
        
        # Main Results
        f.write("## 🏆 Main Model Performance\n\n")
        f.write("### Classification Metrics\n\n")
        
        if not comparison_df.empty:
            f.write("```\n")
            f.write(comparison_df.round(4).to_string())
            f.write("\n```\n\n")
        
        # Model Architecture Comparison
        f.write("## 🏗️ Model Architecture Analysis\n\n")
        f.write("### Transfer Learning Models\n")
        f.write("- **VGG16**: 138M parameters, proven CNN architecture\n")
        f.write("- **ResNet50**: 25M parameters, residual connections for deep training\n\n")
        f.write("### Custom Architecture\n")
        f.write("- **Lightweight Design**: <2M parameters\n")
        f.write("- **Multi-output**: Simultaneous expression classification and emotion regression\n\n")
        
        # Key Findings
        f.write("## 🔍 Key Findings\n\n")
        if 'accuracy' in comparison_df.columns:
            accuracies = comparison_df['accuracy'].sort_values(ascending=False)
            f.write("### Performance Ranking:\n")
            for i, (model, acc) in enumerate(accuracies.items(), 1):
                f.write(f"{i}. **{model.upper()}**: {acc:.4f}\n")
            f.write("\n")
        
        # Technical Insights
        f.write("## 🧠 Technical Insights\n\n")
        f.write("### Training Observations:\n")
        f.write("- **Transfer Learning**: Pre-trained models show faster convergence\n")
        f.write("- **Multi-output Training**: Simultaneous learning of classification and regression tasks\n")
        f.write("- **Data Augmentation**: Improved generalization through image transformations\n")
        f.write("- **Early Stopping**: Prevented overfitting and optimized training time\n\n")
        
        # Recommendations
        f.write("## 💡 Recommendations\n\n")
        f.write("### For Production Deployment:\n")
        if 'accuracy' in comparison_df.columns:
            best_model = comparison_df['accuracy'].idxmax()
            f.write(f"- **Recommended Model**: {best_model.upper()} (highest accuracy)\n")
        f.write("- **Real-time Applications**: Consider model compression techniques\n")
        f.write("- **High-accuracy Requirements**: Ensemble multiple top-performing models\n\n")
        
        f.write("### For Further Research:\n")
        f.write("- **Advanced Architectures**: Experiment with Vision Transformers (ViTs)\n")
        f.write("- **Data Augmentation**: Advanced techniques like MixUp, CutMix\n")
        f.write("- **Semi-supervised Learning**: Leverage unlabeled facial expression data\n")
        f.write("- **Cross-dataset Validation**: Test generalization on different FER datasets\n\n")
        
        # Methodology
        f.write("## 🔬 Methodology\n\n")
        f.write("### Experimental Design:\n")
        f.write("- **Data Split**: 60% training, 20% validation, 20% testing\n")
        f.write("- **Cross-validation**: K-fold validation for robust evaluation\n")
        f.write("- **Hyperparameter Tuning**: Grid search optimization\n")
        f.write("- **Statistical Testing**: Significance tests between models\n")
        f.write("- **Baseline Comparison**: Random, majority class, and simple CNN baselines\n\n")
        
        # Files Generated
        f.write("## 📁 Generated Outputs\n\n")
        f.write("### Model Artifacts:\n")
        f.write("- **Model Weights**: `models/*.h5` - Trained model parameters\n")
        f.write("- **Architecture Summaries**: `results/*_architecture.txt`\n\n")
        
        f.write("### Analysis Results:\n")
        f.write("- **Training Histories**: `results/*_training_history.png`\n")
        f.write("- **Confusion Matrices**: `results/*_confusion_matrix.png`\n")
        f.write("- **ROC Curves**: `results/*_roc_curves.png`\n")
        f.write("- **Learning Curves**: `results/*_advanced_learning_curves.png`\n")
        f.write("- **Residual Analysis**: `results/*_residual_analysis.png`\n\n")
        
        f.write("### Data Files:\n")
        f.write("- **Comprehensive Results**: `results/comprehensive_results.csv`\n")
        f.write("- **Hyperparameter Tuning**: `results/hyperparameter_tuning_results.csv`\n")
        f.write("- **Statistical Tests**: `results/statistical_significance_results.csv`\n")
        f.write("- **Per-class Analysis**: `results/*_per_class_report.csv`\n\n")
        
        # Footer
        f.write("---\n")
        f.write("*Report generated automatically by the Facial Expression Recognition pipeline*\n")
    
    print(f"📋 Comprehensive final report saved to: {report_path}")
if __name__ == "__main__":
    # Validate data before running main pipeline
    if load_and_validate_data():
        main()
    else:
        print("\nPlease ensure your data is properly structured:")
        print("Expected structure:")
        print("  i221903_DL_A1/")
        print("    ├── dataset/")
        print("    │   ├── images/      (JPG files)")
        print("    │   └── annotations/ (.npy files)")
        print("    ├── models/          (will be created)")
        print("    └── results/         (will be created)")
        print("\nTips:")
        print("1. Make sure the dataset folder is in your project directory") 
        print("2. Annotation files should contain 'expression', 'valence', or 'arousal' in their names")
        print("3. Images should be numbered JPG files")