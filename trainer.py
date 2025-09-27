"""
Training module
Handles model training, validation, and callbacks
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, 
    CSVLogger, TensorBoard
)
from tensorflow.keras.optimizers import Adam
from config import Config
from metrics import EvaluationMetrics
class ModelTrainer:
    def __init__(self, config=None):
        self.config = config or Config()
        self.metrics_calculator = EvaluationMetrics(self.config)
        self.history = None
        self.training_time = None
    
    def create_callbacks(self, model_name, monitor='val_loss'):
        """Create training callbacks"""
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = os.path.join(
            self.config.MODEL_SAVE_PATH, 
            f'best_{model_name}_model.h5'
        )
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor=monitor,
            save_best_only=True,
            mode='min' if 'loss' in monitor else 'max',
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor=monitor,
            patience=10,
            restore_best_weights=True,
            mode='min' if 'loss' in monitor else 'max',
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            mode='min' if 'loss' in monitor else 'max',  # This is the key fix
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # CSV Logger
        csv_path = os.path.join(
            self.config.RESULTS_PATH,
            f'{model_name}_training_log.csv'
        )
        csv_logger = CSVLogger(csv_path)
        callbacks.append(csv_logger)
        
        # TensorBoard
        tb_path = os.path.join(
            self.config.RESULTS_PATH,
            'tensorboard',
            model_name
        )
        tensorboard = TensorBoard(
            log_dir=tb_path,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callbacks.append(tensorboard)
        
        return callbacks

    def train_model(self, model, train_data, val_data, model_name, multi_output=True):
        """Train the model with the given data"""
        print(f"\nTraining {model_name}...")
        print(f"Model parameters: {model.count_params():,}")
        
        start_time = time.time()
        
        # Determine the appropriate monitor metric
        if multi_output:
            monitor = 'val_expression_accuracy'  # Focus on expression accuracy for multi-output
        else:
            monitor = 'val_accuracy'
        
        # Create callbacks
        callbacks = self.create_callbacks(model_name, monitor=monitor)
        
        # Prepare data
        if multi_output:
            x_train, y_expr_train, y_val_train, y_ar_train = train_data
            x_val, y_expr_val, y_val_val, y_ar_val = val_data
            
            # Combine outputs for training
            y_train_combined = [y_expr_train, y_val_train, y_ar_train]
            y_val_combined = [y_expr_val, y_val_val, y_ar_val]
            
            validation_data = (x_val, y_val_combined)
        else:
            x_train, y_train = train_data
            x_val, y_val = val_data
            y_train_combined = y_train
            validation_data = (x_val, y_val)
        
        # Train the model
        try:
            self.history = model.fit(
                x_train,
                y_train_combined,
                batch_size=self.config.BATCH_SIZE,
                epochs=self.config.EPOCHS,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1
            )
            
            self.training_time = time.time() - start_time
            print(f"Training completed in {self.training_time:.2f} seconds")
            
            return self.history
            
        except Exception as e:
            print(f"Error during training: {e}")
            raise e
    def plot_training_history(self, model_name, save_path=None):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        history = self.history.history
        epochs = range(1, len(history['loss']) + 1)
        
        # Determine number of subplots based on available metrics
        n_plots = 1  # At least loss plot
        if 'expression_accuracy' in history:
            n_plots += 1
        if 'valence_mae' in history:
            n_plots += 1
        if 'arousal_mae' in history:
            n_plots += 1
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        plot_idx = 0
        
        # Plot training and validation loss
        axes[plot_idx].plot(epochs, history['loss'], 'b-', label='Training Loss')
        if 'val_loss' in history:
            axes[plot_idx].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        axes[plot_idx].set_title('Model Loss')
        axes[plot_idx].set_xlabel('Epochs')
        axes[plot_idx].set_ylabel('Loss')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True)
        plot_idx += 1
        
        # Plot expression accuracy
        if 'expression_accuracy' in history:
            axes[plot_idx].plot(epochs, history['expression_accuracy'], 'b-', label='Training Accuracy')
            if 'val_expression_accuracy' in history:
                axes[plot_idx].plot(epochs, history['val_expression_accuracy'], 'r-', label='Validation Accuracy')
            axes[plot_idx].set_title('Expression Classification Accuracy')
            axes[plot_idx].set_xlabel('Epochs')
            axes[plot_idx].set_ylabel('Accuracy')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True)
            plot_idx += 1
        
        # Plot valence MAE
        if 'valence_mae' in history:
            axes[plot_idx].plot(epochs, history['valence_mae'], 'b-', label='Training MAE')
            if 'val_valence_mae' in history:
                axes[plot_idx].plot(epochs, history['val_valence_mae'], 'r-', label='Validation MAE')
            axes[plot_idx].set_title('Valence Mean Absolute Error')
            axes[plot_idx].set_xlabel('Epochs')
            axes[plot_idx].set_ylabel('MAE')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True)
            plot_idx += 1
        
        # Plot arousal MAE
        if 'arousal_mae' in history:
            axes[plot_idx].plot(epochs, history['arousal_mae'], 'b-', label='Training MAE')
            if 'val_arousal_mae' in history:
                axes[plot_idx].plot(epochs, history['val_arousal_mae'], 'r-', label='Validation MAE')
            axes[plot_idx].set_title('Arousal Mean Absolute Error')
            axes[plot_idx].set_xlabel('Epochs')
            axes[plot_idx].set_ylabel('MAE')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True)
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'{model_name} Training History')
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.config.RESULTS_PATH, f'{model_name}_training_history.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def fine_tune_model(self, model, train_data, val_data, model_name, 
                       layers_to_unfreeze=20, fine_tune_epochs=20):
        """Fine-tune pre-trained model"""
        print(f"\nFine-tuning {model_name}...")
        
        # Unfreeze some layers
        if hasattr(model.layers[1], 'trainable'):  # Assuming base model is second layer
            base_model = model.layers[1]
            base_model.trainable = True
            
            # Freeze early layers
            for layer in base_model.layers[:-layers_to_unfreeze]:
                layer.trainable = False
            
            print(f"Unfroze last {layers_to_unfreeze} layers")
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE / 10),
            loss=model.loss,
            metrics=model.metrics,
            loss_weights=getattr(model, 'loss_weights', None)
        )
        
        # Prepare data
        if isinstance(train_data[1], list):  # Multi-output
            x_train, y_train = train_data[0], train_data[1]
            x_val, y_val = val_data[0], val_data[1]
            monitor = 'val_expression_accuracy'
        else:
            x_train, y_train = train_data
            x_val, y_val = val_data
            monitor = 'val_accuracy'
        
        # Create callbacks for fine-tuning
        callbacks = self.create_callbacks(f'{model_name}_finetuned', monitor=monitor)
        
        # Fine-tune
        history_fine_tune = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=fine_tune_epochs,
            batch_size=self.config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        if self.history is not None:
            for key in history_fine_tune.history:
                if key in self.history.history:
                    self.history.history[key].extend(history_fine_tune.history[key])
                else:
                    self.history.history[key] = history_fine_tune.history[key]
        else:
            self.history = history_fine_tune
        
        return history_fine_tune
    
    def evaluate_model(self, model, test_data, model_name, multi_output=True):
        """Evaluate model on test data"""
        print(f"\nEvaluating {model_name}...")
        
        if multi_output:
            x_test, y_test_expr, y_test_val, y_test_ar = test_data
            
            # Make predictions
            predictions = model.predict(x_test)
            pred_expr, pred_val, pred_ar = predictions
            
            # Calculate categorical metrics
            categorical_metrics = self.metrics_calculator.calculate_categorical_metrics(
                y_test_expr, pred_expr, pred_expr
            )
            
            # Calculate continuous metrics
            valence_metrics = self.metrics_calculator.calculate_continuous_metrics(
                y_test_val, pred_val, 'valence'
            )
            
            arousal_metrics = self.metrics_calculator.calculate_continuous_metrics(
                y_test_ar, pred_ar, 'arousal'
            )
            
            # Create results summary
            results = self.metrics_calculator.create_results_summary(
                categorical_metrics, valence_metrics, arousal_metrics
            )
            
            # Generate visualizations
            self.generate_evaluation_plots(
                y_test_expr, pred_expr, y_test_val, pred_val, 
                y_test_ar, pred_ar, model_name
            )
            
        else:
            x_test, y_test = test_data
            
            # Make predictions
            predictions = model.predict(x_test)
            
            # Calculate categorical metrics
            categorical_metrics = self.metrics_calculator.calculate_categorical_metrics(
                y_test, predictions, predictions
            )
            
            results = {'Expression Classification': categorical_metrics}
            
            # Generate visualizations
            self.generate_evaluation_plots(y_test, predictions, None, None, None, None, model_name)
        
        return results
    
    def generate_evaluation_plots(self, y_true_expr, pred_expr, y_true_val, pred_val, 
                                 y_true_ar, pred_ar, model_name):
        """Generate evaluation plots"""
        # Confusion matrix
        cm_path = os.path.join(self.config.RESULTS_PATH, f'{model_name}_confusion_matrix.png')
        self.metrics_calculator.plot_confusion_matrix(y_true_expr, pred_expr, cm_path)
        
        # Classification report
        self.metrics_calculator.generate_classification_report(y_true_expr, pred_expr)
        
        # Regression plots (if available)
        if y_true_val is not None and pred_val is not None:
            val_path = os.path.join(self.config.RESULTS_PATH, f'{model_name}_valence_regression.png')
            self.metrics_calculator.plot_regression_results(y_true_val, pred_val, 'Valence', val_path)
        
        if y_true_ar is not None and pred_ar is not None:
            ar_path = os.path.join(self.config.RESULTS_PATH, f'{model_name}_arousal_regression.png')
            self.metrics_calculator.plot_regression_results(y_true_ar, pred_ar, 'Arousal', ar_path)
    
    def save_model_summary(self, model, model_name):
        """Save model architecture summary to file"""
        summary_path = os.path.join(
            self.config.RESULTS_PATH,
            f'{model_name}_architecture.txt'
        )
        
        try:
            # Use UTF-8 encoding to handle Unicode characters
            with open(summary_path, 'w', encoding='utf-8') as f:
                # Get model summary as string
                summary_lines = []
                model.summary(print_fn=lambda x: summary_lines.append(x))
                
                # Write summary to file
                for line in summary_lines:
                    f.write(line + '\n')
                
                # Add additional model info
                f.write(f"\nTotal params: {model.count_params():,}\n")
                f.write(f"Trainable params: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}\n")
                f.write(f"Non-trainable params: {sum([tf.size(w).numpy() for w in model.non_trainable_weights]):,}\n")
            
            print(f"Model summary saved to: {summary_path}")
            
        except Exception as e:
            print(f"Warning: Could not save model summary for {model_name}: {e}")
            # Continue execution even if summary saving fails
        
    def get_training_time(self):
        """Get training time"""
        return self.training_time