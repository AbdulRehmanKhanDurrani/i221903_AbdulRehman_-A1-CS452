"""
Evaluation metrics module
Contains all evaluation metrics for both categorical and continuous predictions
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, roc_auc_score, 
    precision_recall_curve, auc, classification_report, confusion_matrix
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import pandas as pd
from config import Config

class EvaluationMetrics:
    def __init__(self, config=None):
        self.config = config or Config()
    
    def calculate_categorical_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate all categorical classification metrics"""
        # Convert one-hot to class indices if needed
        if len(y_true.shape) > 1:
            y_true_idx = np.argmax(y_true, axis=1)
        else:
            y_true_idx = y_true
            
        if len(y_pred.shape) > 1:
            y_pred_idx = np.argmax(y_pred, axis=1)
        else:
            y_pred_idx = y_pred
        
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(y_true_idx, y_pred_idx)
        
        # F1-Score (macro and weighted)
        metrics['f1_macro'] = f1_score(y_true_idx, y_pred_idx, average='macro')
        metrics['f1_weighted'] = f1_score(y_true_idx, y_pred_idx, average='weighted')
        
        # Cohen's Kappa
        metrics['cohens_kappa'] = cohen_kappa_score(y_true_idx, y_pred_idx)
        
        # Krippendorf's Alpha (approximated using Cohen's Kappa for binary case)
        metrics['krippendorfs_alpha'] = self.calculate_krippendorfs_alpha(y_true_idx, y_pred_idx)
        
        # AUC metrics (if probabilities are provided)
        if y_pred_proba is not None:
            try:
                # Multi-class AUC
                y_true_bin = label_binarize(y_true_idx, classes=np.arange(self.config.NUM_CLASSES))
                if self.config.NUM_CLASSES == 2:
                    metrics['auc_roc'] = roc_auc_score(y_true_bin, y_pred_proba[:, 1])
                else:
                    metrics['auc_roc'] = roc_auc_score(y_true_bin, y_pred_proba, multi_class='ovr', average='macro')
                
                # AUC-PR
                metrics['auc_pr'] = self.calculate_auc_pr(y_true_bin, y_pred_proba)
                
            except Exception as e:
                print(f"Error calculating AUC metrics: {e}")
                metrics['auc_roc'] = None
                metrics['auc_pr'] = None
        
        return metrics
    
    def calculate_continuous_metrics(self, y_true, y_pred, metric_type='valence'):
        """Calculate continuous domain metrics (RMSE, CORR, SAGR, CCC)"""
        metrics = {}
        
        # RMSE
        metrics[f'{metric_type}_rmse'] = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # Correlation
        try:
            corr_coef, p_value = pearsonr(y_true.flatten(), y_pred.flatten())
            metrics[f'{metric_type}_corr'] = corr_coef
            metrics[f'{metric_type}_corr_pvalue'] = p_value
        except:
            metrics[f'{metric_type}_corr'] = 0.0
            metrics[f'{metric_type}_corr_pvalue'] = 1.0
        
        # Sign Agreement Metric (SAGR)
        metrics[f'{metric_type}_sagr'] = self.calculate_sagr(y_true, y_pred)
        
        # Concordance Correlation Coefficient (CCC)
        metrics[f'{metric_type}_ccc'] = self.calculate_ccc(y_true, y_pred)
        
        return metrics
    
    def calculate_krippendorfs_alpha(self, y_true, y_pred):
        """
        Simplified Krippendorf's Alpha calculation
        For detailed implementation, consider using specialized libraries
        """
        # This is a simplified version - for accurate calculation, use krippendorff library
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(y_true, y_pred)
    
    def calculate_auc_pr(self, y_true_bin, y_pred_proba):
        """Calculate Area Under Precision-Recall Curve"""
        if len(y_true_bin.shape) == 1 or y_true_bin.shape[1] == 1:
            # Binary case
            precision, recall, _ = precision_recall_curve(y_true_bin, y_pred_proba[:, 1])
            return auc(recall, precision)
        else:
            # Multi-class case
            auc_scores = []
            for i in range(y_true_bin.shape[1]):
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
                auc_scores.append(auc(recall, precision))
            return np.mean(auc_scores)
    
    def calculate_sagr(self, y_true, y_pred):
        """
        Calculate Sign Agreement Metric (SAGR)
        Penalizes incorrect sign alongside deviation from value
        """
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Count agreements and disagreements
        same_sign = np.sign(y_true_flat) == np.sign(y_pred_flat)
        
        # For zero values, consider them as positive
        y_true_sign = np.where(y_true_flat == 0, 1, np.sign(y_true_flat))
        y_pred_sign = np.where(y_pred_flat == 0, 1, np.sign(y_pred_flat))
        
        same_sign = y_true_sign == y_pred_sign
        
        # SAGR is the proportion of same signs
        sagr = np.mean(same_sign)
        
        return sagr
    
    def calculate_ccc(self, y_true, y_pred):
        """
        Calculate Concordance Correlation Coefficient (CCC)
        Combines Pearson's correlation with square difference between means
        """
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Pearson correlation coefficient
        corr_coef = np.corrcoef(y_true_flat, y_pred_flat)[0, 1]
        
        # Means
        mean_true = np.mean(y_true_flat)
        mean_pred = np.mean(y_pred_flat)
        
        # Variances
        var_true = np.var(y_true_flat)
        var_pred = np.var(y_pred_flat)
        
        # CCC formula
        numerator = 2 * corr_coef * np.sqrt(var_true) * np.sqrt(var_pred)
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
        
        ccc = numerator / denominator if denominator != 0 else 0.0
        
        return ccc
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot confusion matrix"""
        # Convert one-hot to class indices if needed
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[self.config.EMOTION_LABELS[i] for i in range(self.config.NUM_CLASSES)],
                    yticklabels=[self.config.EMOTION_LABELS[i] for i in range(self.config.NUM_CLASSES)])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_regression_results(self, y_true, y_pred, metric_name, save_path=None):
        """Plot regression results"""
        plt.figure(figsize=(10, 4))
        
        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([-1, 1], [-1, 1], 'r--', lw=2)
        plt.xlabel(f'True {metric_name}')
        plt.ylabel(f'Predicted {metric_name}')
        plt.title(f'{metric_name} Prediction vs Ground Truth')
        plt.grid(True, alpha=0.3)
        
        # Error histogram
        plt.subplot(1, 2, 2)
        errors = y_pred - y_true
        plt.hist(errors, bins=50, alpha=0.7)
        plt.xlabel(f'Prediction Error ({metric_name})')
        plt.ylabel('Frequency')
        plt.title(f'{metric_name} Prediction Errors')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_classification_report(self, y_true, y_pred):
        """Generate detailed classification report"""
        # Convert one-hot to class indices if needed
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        target_names = [self.config.EMOTION_LABELS[i] for i in range(self.config.NUM_CLASSES)]
        
        report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
        
        # Print formatted report
        print("\nDetailed Classification Report:")
        print("=" * 60)
        print(classification_report(y_true, y_pred, target_names=target_names))
        
        return report
    
    def create_results_summary(self, categorical_metrics, valence_metrics=None, arousal_metrics=None):
        """Create a comprehensive results summary"""
        results = {}
        
        # Categorical metrics
        results['Expression Classification'] = categorical_metrics
        
        # Continuous metrics
        if valence_metrics:
            results['Valence Regression'] = valence_metrics
        
        if arousal_metrics:
            results['Arousal Regression'] = arousal_metrics
        
        return results
    
    def print_metrics_summary(self, results_dict):
        """Print formatted metrics summary"""
        print("\n" + "="*80)
        print("EVALUATION RESULTS SUMMARY")
        print("="*80)
        
        for task, metrics in results_dict.items():
            print(f"\n{task}:")
            print("-" * 40)
            
            for metric_name, value in metrics.items():
                if value is not None:
                    if isinstance(value, float):
                        print(f"{metric_name:25}: {value:.4f}")
                    else:
                        print(f"{metric_name:25}: {value}")
                else:
                    print(f"{metric_name:25}: N/A")
    
    def save_results_to_csv(self, results_dict, filename):
        """Save results to CSV file"""
        all_metrics = {}
        
        for task, metrics in results_dict.items():
            for metric_name, value in metrics.items():
                all_metrics[f"{task}_{metric_name}"] = [value]
        
        df = pd.DataFrame(all_metrics)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
    
    def compare_models_results(self, model_results_dict):
        """Compare results from multiple models"""
        comparison_df = pd.DataFrame(model_results_dict).T
        
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(comparison_df.round(4))
        
        # Find best performing models
        print("\nBest performing models:")
        print("-" * 40)
        
        key_metrics = ['accuracy', 'f1_macro', 'valence_rmse', 'arousal_rmse', 'valence_ccc', 'arousal_ccc']
        
        for metric in key_metrics:
            if metric in comparison_df.columns:
                if 'rmse' in metric:
                    best_model = comparison_df[metric].idxmin()
                    best_value = comparison_df[metric].min()
                else:
                    best_model = comparison_df[metric].idxmax()
                    best_value = comparison_df[metric].max()
                
                print(f"{metric:20}: {best_model} ({best_value:.4f})")
        
        return comparison_df