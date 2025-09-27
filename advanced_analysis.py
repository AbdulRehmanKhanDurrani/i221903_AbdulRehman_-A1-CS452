"""
Advanced analysis module for deep model insights
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow.keras.models import Model
import cv2
from config import Config

class AdvancedAnalyzer:
    def __init__(self, config=None):
        self.config = config or Config()
        
    def create_confusion_matrix_heatmap(self, y_true, y_pred, model_name, class_names=None):
        """Create and save confusion matrix heatmap"""
        if class_names is None:
            class_names = list(self.config.EMOTION_LABELS.values())
        
        # Convert one-hot to labels if needed
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name.upper()}', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # Save plot
        save_path = os.path.join(self.config.RESULTS_PATH, f'{model_name}_confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_roc_curves(self, y_true, y_pred_proba, model_name, class_names=None):
        """Plot ROC curves for each class"""
        if class_names is None:
            class_names = list(self.config.EMOTION_LABELS.values())
        
        n_classes = len(class_names)
        
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Compute ROC curve and ROC area for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        
        for i, color in zip(range(n_classes), colors):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curves - {model_name.upper()}', fontsize=16)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save plot
        save_path = os.path.join(self.config.RESULTS_PATH, f'{model_name}_roc_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return roc_auc
    
    def per_class_analysis(self, y_true, y_pred, model_name, class_names=None):
        """Detailed per-class performance analysis"""
        if class_names is None:
            class_names = list(self.config.EMOTION_LABELS.values())
        
        # Convert one-hot to labels if needed
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Convert to DataFrame
        report_df = pd.DataFrame(report).transpose()
        
        # Save detailed report
        report_path = os.path.join(self.config.RESULTS_PATH, f'{model_name}_per_class_report.csv')
        report_df.to_csv(report_path)
        
        # Visualize per-class metrics
        metrics_to_plot = ['precision', 'recall', 'f1-score']
        class_data = report_df.iloc[:-3]  # Exclude avg rows
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, metric in enumerate(metrics_to_plot):
            class_data[metric].plot(kind='bar', ax=axes[i], color='skyblue')
            axes[i].set_title(f'{metric.capitalize()} by Class - {model_name.upper()}', fontsize=12)
            axes[i].set_xlabel('Emotion Class')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(0, 1)
        
        plt.tight_layout()
        save_path = os.path.join(self.config.RESULTS_PATH, f'{model_name}_per_class_metrics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return report_df
    
    def learning_curve_analysis(self, history, model_name):
        """Advanced learning curve analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss curves
        if 'loss' in history.history:
            axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
            if 'val_loss' in history.history:
                axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
            axes[0, 0].set_title(f'Model Loss - {model_name.upper()}', fontsize=14)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        accuracy_key = 'expression_accuracy' if 'expression_accuracy' in history.history else 'accuracy'
        val_accuracy_key = f'val_{accuracy_key}'
        
        if accuracy_key in history.history:
            axes[0, 1].plot(history.history[accuracy_key], label='Training Accuracy', linewidth=2)
            if val_accuracy_key in history.history:
                axes[0, 1].plot(history.history[val_accuracy_key], label='Validation Accuracy', linewidth=2)
            axes[0, 1].set_title(f'Model Accuracy - {model_name.upper()}', fontsize=14)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate (if available)
        if hasattr(history, 'lr') and history.lr:
            axes[1, 0].plot(history.lr, label='Learning Rate', linewidth=2, color='red')
            axes[1, 0].set_title(f'Learning Rate Schedule - {model_name.upper()}', fontsize=14)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Overfitting analysis
        if 'val_loss' in history.history and 'loss' in history.history:
            train_loss = np.array(history.history['loss'])
            val_loss = np.array(history.history['val_loss'])
            gap = val_loss - train_loss
            
            axes[1, 1].plot(gap, label='Validation - Training Loss', linewidth=2, color='orange')
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 1].set_title(f'Overfitting Analysis - {model_name.upper()}', fontsize=14)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss Gap')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.config.RESULTS_PATH, f'{model_name}_advanced_learning_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def regression_residual_analysis(self, y_true, y_pred, model_name, target_name):
        """Analyze regression residuals"""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title(f'Residuals vs Predicted - {target_name}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals vs True
        axes[0, 1].scatter(y_true, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('True Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title(f'Residuals vs True - {target_name}')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Residuals Distribution - {target_name}')
        axes[1, 0].grid(True, alpha=0.3)
        
        # True vs Predicted scatter
        axes[1, 1].scatter(y_true, y_pred, alpha=0.6)
        axes[1, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[1, 1].set_xlabel('True Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].set_title(f'True vs Predicted - {target_name}')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.config.RESULTS_PATH, f'{model_name}_{target_name}_residual_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate residual statistics
        residual_stats = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'mae': np.mean(np.abs(residuals)),
            'rmse': np.sqrt(np.mean(residuals**2)),
            'median': np.median(residuals),
            'q25': np.percentile(residuals, 25),
            'q75': np.percentile(residuals, 75)
        }
        
        return residual_stats
    
    def statistical_significance_test(self, results_dict):
        """Perform statistical significance tests between models"""
        from scipy import stats
        
        print("\n📊 STATISTICAL SIGNIFICANCE TESTS")
        print("=" * 50)
        
        model_names = list(results_dict.keys())
        significance_results = {}
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                
                # Extract accuracies (assuming we have cross-validation scores)
                # This is a simplified version - you'd need actual CV scores
                acc1 = results_dict[model1].get('accuracy', 0)
                acc2 = results_dict[model2].get('accuracy', 0)
                
                # Simulated significance test (replace with actual CV scores)
                print(f"\n{model1.upper()} vs {model2.upper()}:")
                
                if acc1 > acc2:
                    diff = acc1 - acc2
                    print(f"  {model1.upper()} performs better by {diff:.4f}")
                    if diff > 0.02:  # Simple threshold
                        print("  ✅ Difference appears significant (>2%)")
                    else:
                        print("  ⚠️  Difference may not be significant")
                else:
                    diff = acc2 - acc1
                    print(f"  {model2.upper()} performs better by {diff:.4f}")
                    if diff > 0.02:
                        print("  ✅ Difference appears significant (>2%)")
                    else:
                        print("  ⚠️  Difference may not be significant")
                
                significance_results[f"{model1}_vs_{model2}"] = {
                    'model1_acc': acc1,
                    'model2_acc': acc2,
                    'difference': abs(acc1 - acc2),
                    'better_model': model1 if acc1 > acc2 else model2
                }
        
        return significance_results