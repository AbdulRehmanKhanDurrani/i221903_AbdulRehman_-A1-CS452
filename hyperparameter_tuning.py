"""
Hyperparameter tuning module for systematic optimization
"""
import os
import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import tensorflow as tf
from config import Config
from models import CNNModels
from trainer import ModelTrainer

class HyperparameterTuner:
    def __init__(self, config=None):
        self.config = config or Config()
        self.results = []
        
    def define_search_space(self):
        """Define hyperparameter search space"""
        return {
            'learning_rate': [0.001, 0.0001, 0.00001],
            'batch_size': [16, 32, 64],
            'optimizer': ['adam', 'sgd', 'rmsprop'],
            'dropout_rate': [0.3, 0.5, 0.7],
            'l2_reg': [0.01, 0.001, 0.0001]
        }
    
    def grid_search(self, model_name, train_data, val_data, max_combinations=10):
        """Perform grid search for hyperparameters"""
        print(f"\n🔍 Starting hyperparameter tuning for {model_name.upper()}...")
        print("-" * 60)
        
        search_space = self.define_search_space()
        
        # Generate all combinations
        param_names = list(search_space.keys())
        param_values = list(search_space.values())
        all_combinations = list(product(*param_values))
        
        # Limit combinations for computational feasibility
        if len(all_combinations) > max_combinations:
            print(f"Limiting to {max_combinations} random combinations from {len(all_combinations)} total")
            np.random.shuffle(all_combinations)
            all_combinations = all_combinations[:max_combinations]
        
        best_score = 0
        best_params = None
        
        for i, combination in enumerate(all_combinations):
            params = dict(zip(param_names, combination))
            print(f"\nTrial {i+1}/{len(all_combinations)}: {params}")
            
            try:
                score = self.evaluate_hyperparameters(model_name, params, train_data, val_data)
                
                self.results.append({
                    'model': model_name,
                    'trial': i+1,
                    'score': score,
                    **params
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    print(f"✅ New best score: {best_score:.4f}")
                
            except Exception as e:
                print(f"❌ Trial failed: {e}")
                continue
        
        print(f"\n🏆 Best hyperparameters for {model_name}:")
        print(f"Score: {best_score:.4f}")
        print(f"Parameters: {best_params}")
        
        return best_params, best_score
    
    def evaluate_hyperparameters(self, model_name, params, train_data, val_data):
        """Evaluate a single hyperparameter combination"""
        # Update config with new parameters
        original_batch_size = self.config.BATCH_SIZE
        original_lr = self.config.LEARNING_RATE
        
        self.config.BATCH_SIZE = params['batch_size']
        self.config.LEARNING_RATE = params['learning_rate']
        
        try:
            # Create model with new parameters
            model_factory = CNNModels(self.config)
            
            # Modify model creation to include hyperparameters
            if model_name == 'custom':
                model = model_factory.create_custom_cnn(
                    multi_output=True,
                    dropout_rate=params['dropout_rate'],
                    l2_reg=params['l2_reg']
                )
            else:
                model = model_factory.get_model(model_name, multi_output=True)
            
            # Compile with specified optimizer
            optimizer = self.get_optimizer(params['optimizer'], params['learning_rate'])
            
            if hasattr(model, 'outputs') and len(model.outputs) > 1:
                model.compile(
                    optimizer=optimizer,
                    loss={
                        'expression': 'categorical_crossentropy',
                        'valence': 'mse',
                        'arousal': 'mse'
                    },
                    metrics={
                        'expression': ['accuracy'],
                        'valence': ['mae'],
                        'arousal': ['mae']
                    },
                    loss_weights={'expression': 1.0, 'valence': 0.5, 'arousal': 0.5}
                )
            
            # Quick training (reduced epochs for tuning)
            trainer = ModelTrainer(self.config)
            trainer.config.EPOCHS = 5  # Reduced for tuning
            
            history = trainer.train_model(
                model, train_data, val_data, f'{model_name}_tune', multi_output=True
            )
            
            # Return validation accuracy as score
            if 'val_expression_accuracy' in history.history:
                return max(history.history['val_expression_accuracy'])
            else:
                return 0
                
        finally:
            # Restore original config
            self.config.BATCH_SIZE = original_batch_size
            self.config.LEARNING_RATE = original_lr
    
    def get_optimizer(self, optimizer_name, learning_rate):
        """Get optimizer by name"""
        if optimizer_name == 'adam':
            return Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            return SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer_name == 'rmsprop':
            return RMSprop(learning_rate=learning_rate)
        else:
            return Adam(learning_rate=learning_rate)
    
    def save_tuning_results(self):
        """Save hyperparameter tuning results"""
        if self.results:
            df = pd.DataFrame(self.results)
            results_path = os.path.join(self.config.RESULTS_PATH, 'hyperparameter_tuning_results.csv')
            df.to_csv(results_path, index=False)
            print(f"Hyperparameter tuning results saved to: {results_path}")
            return df
        return None
    
    def cross_validate_model(self, model_name, best_params, X, y, cv_folds=3):
        """Perform cross-validation with best hyperparameters"""
        print(f"\n🔄 Cross-validating {model_name} with best parameters...")
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        # Convert one-hot back to labels for stratification
        if len(y.shape) > 1:
            y_labels = np.argmax(y, axis=1)
        else:
            y_labels = y
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_labels)):
            print(f"Fold {fold + 1}/{cv_folds}")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Update config with best parameters
            self.config.BATCH_SIZE = best_params['batch_size']
            self.config.LEARNING_RATE = best_params['learning_rate']
            
            # Create and train model
            model_factory = CNNModels(self.config)
            model = model_factory.get_model(model_name, multi_output=False)
            
            # Quick training
            trainer = ModelTrainer(self.config)
            trainer.config.EPOCHS = 10
            
            train_data_fold = (X_train_fold, y_train_fold)
            val_data_fold = (X_val_fold, y_val_fold)
            
            history = trainer.train_model(
                model, train_data_fold, val_data_fold, f'{model_name}_cv_{fold}', multi_output=False
            )
            
            # Get best validation accuracy
            val_acc = max(history.history.get('val_accuracy', [0]))
            cv_scores.append(val_acc)
            print(f"Fold {fold + 1} validation accuracy: {val_acc:.4f}")
        
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        print(f"\n📊 Cross-validation results for {model_name}:")
        print(f"Mean accuracy: {mean_score:.4f} ± {std_score:.4f}")
        print(f"Individual scores: {[f'{s:.4f}' for s in cv_scores]}")
        
        return cv_scores, mean_score, std_score