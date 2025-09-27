"""
CNN Models module
Contains different CNN architectures for facial expression recognition
"""

import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, Xception, MobileNetV2, EfficientNetB0, DenseNet121
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, Input, Conv2D, MaxPooling2D, 
    Flatten, BatchNormalization, Activation, Add, concatenate
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from config import Config
import numpy as np

class CNNModels:
    def __init__(self, config=None):
        self.config = config or Config()
        self.input_shape = (self.config.IMG_HEIGHT, self.config.IMG_WIDTH, self.config.CHANNELS)
    
    def create_base_model(self, model_name, include_top=False, weights='imagenet'):
        """Create base model from pre-trained architectures"""
        models_dict = {
            'vgg16': VGG16,
            'resnet50': ResNet50,
            'inceptionv3': InceptionV3,
            'xception': Xception,
            'mobilenetv2': MobileNetV2,
            'efficientnetb0': EfficientNetB0,
            'densenet121': DenseNet121
        }
        
        if model_name.lower() not in models_dict:
            raise ValueError(f"Model {model_name} not supported. Available: {list(models_dict.keys())}")
        
        base_model = models_dict[model_name.lower()](
            weights=weights,
            include_top=include_top,
            input_shape=self.input_shape
        )
        
        return base_model
    
    def create_transfer_learning_model(self, base_model_name, multi_output=True):
        """Create transfer learning model with custom head"""
        base_model = self.create_base_model(base_model_name)
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom head
        inputs = Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.config.DROPOUT_RATE)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(self.config.DROPOUT_RATE)(x)
        
        # Expression classification output
        expression_output = Dense(self.config.NUM_CLASSES, activation='softmax', name='expression')(x)
        
        outputs = [expression_output]
        
        if multi_output:
            # Valence regression output
            valence_output = Dense(1, activation='tanh', name='valence')(x)  # tanh for [-1, 1] range
            
            # Arousal regression output
            arousal_output = Dense(1, activation='tanh', name='arousal')(x)  # tanh for [-1, 1] range
            
            outputs.extend([valence_output, arousal_output])
        
        model = Model(inputs, outputs)
        return model
    
    def create_custom_cnn(self, multi_output=True):
        """Create custom CNN architecture"""
        inputs = Input(shape=self.input_shape)
        
        # First block
        x = Conv2D(32, (3, 3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        # Second block
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        # Third block
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        # Fourth block
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        # Global Average Pooling
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        
        # Expression classification output
        expression_output = Dense(self.config.NUM_CLASSES, activation='softmax', name='expression')(x)
        
        outputs = [expression_output]
        
        if multi_output:
            # Valence regression output
            valence_output = Dense(1, activation='tanh', name='valence')(x)
            
            # Arousal regression output
            arousal_output = Dense(1, activation='tanh', name='arousal')(x)
            
            outputs.extend([valence_output, arousal_output])
        
        model = Model(inputs, outputs)
        return model
    
    def create_residual_block(self, x, filters, kernel_size=3, stride=1):
        """Create a residual block"""
        # Main path
        x_main = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
        x_main = BatchNormalization()(x_main)
        x_main = Activation('relu')(x_main)
        x_main = Conv2D(filters, kernel_size, padding='same')(x_main)
        x_main = BatchNormalization()(x_main)
        
        # Shortcut path
        if stride > 1 or x.shape[-1] != filters:
            x = Conv2D(filters, 1, strides=stride, padding='same')(x)
            x = BatchNormalization()(x)
        
        # Add shortcut to main path
        x = Add()([x_main, x])
        x = Activation('relu')(x)
        
        return x
    
    def create_improved_custom_cnn(self, multi_output=True):
        """Create improved custom CNN with residual connections"""
        inputs = Input(shape=self.input_shape)
        
        # Initial conv layer
        x = Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
        
        # Residual blocks
        x = self.create_residual_block(x, 64)
        x = self.create_residual_block(x, 64)
        
        x = self.create_residual_block(x, 128, stride=2)
        x = self.create_residual_block(x, 128)
        
        x = self.create_residual_block(x, 256, stride=2)
        x = self.create_residual_block(x, 256)
        
        x = self.create_residual_block(x, 512, stride=2)
        x = self.create_residual_block(x, 512)
        
        # Global Average Pooling
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.5)(x)
        
        # Expression classification output
        expression_output = Dense(self.config.NUM_CLASSES, activation='softmax', name='expression')(x)
        
        outputs = [expression_output]
        
        if multi_output:
            # Valence regression output
            valence_output = Dense(1, activation='tanh', name='valence')(x)
            
            # Arousal regression output  
            arousal_output = Dense(1, activation='tanh', name='arousal')(x)
            
            outputs.extend([valence_output, arousal_output])
        
        model = Model(inputs, outputs)
        return model
    
    def compile_model(self, model, multi_output=True):
        """Compile model with appropriate loss functions and metrics"""
        if multi_output:
            # Multi-output model
            losses = {
                'expression': 'categorical_crossentropy',
                'valence': 'mse',
                'arousal': 'mse'
            }
            
            metrics = {
                'expression': ['accuracy'],
                'valence': ['mae'],
                'arousal': ['mae']
            }
            
            loss_weights = {
                'expression': 1.0,
                'valence': 0.5,
                'arousal': 0.5
            }
            
            model.compile(
                optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
                loss=losses,
                metrics=metrics,
                loss_weights=loss_weights
            )
        else:
            # Single output model (expressions only)
            model.compile(
                optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        return model
    
    def get_model(self, model_name, multi_output=True):
        """Get and compile a specific model"""
        if model_name.lower() == 'custom':
            model = self.create_custom_cnn(multi_output)
        elif model_name.lower() == 'improved_custom':
            model = self.create_improved_custom_cnn(multi_output)
        else:
            model = self.create_transfer_learning_model(model_name, multi_output)
        
        return self.compile_model(model, multi_output)
    
    def unfreeze_base_model(self, model, layers_to_unfreeze=None):
        """Unfreeze base model layers for fine-tuning"""
        if hasattr(model.layers[1], 'trainable'):  # Assuming base model is the second layer
            base_model = model.layers[1]
            
            if layers_to_unfreeze is None:
                # Unfreeze all layers
                base_model.trainable = True
            else:
                # Unfreeze specific layers
                base_model.trainable = True
                for layer in base_model.layers[:-layers_to_unfreeze]:
                    layer.trainable = False
        
        return model
    

class BaselineModels:
    def __init__(self, config=None):
        self.config = config or Config()
    
    def random_classifier(self, num_classes):
        """Simple random classifier baseline"""
        def predict(X):
            return np.random.randint(0, num_classes, size=(len(X),))
        return predict
    
    def majority_class_classifier(self, y_train):
        """Majority class baseline"""
        if len(y_train.shape) > 1:
            y_labels = np.argmax(y_train, axis=1)
        else:
            y_labels = y_train
            
        majority_class = np.bincount(y_labels).argmax()
        
        def predict(X):
            return np.full(len(X), majority_class)
        return predict
    
    def simple_cnn_baseline(self, multi_output=False):
        """Very simple CNN baseline"""
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input
        
        inputs = Input(shape=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH, self.config.CHANNELS))
        
        # Ultra-simple architecture
        x = Conv2D(32, 3, activation='relu')(inputs)
        x = MaxPooling2D()(x)
        x = Conv2D(64, 3, activation='relu')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        
        if multi_output:
            # Expression classification
            expr_output = Dense(self.config.NUM_CLASSES, activation='softmax', name='expression')(x)
            
            # Valence regression
            val_output = Dense(1, activation='tanh', name='valence')(x)
            
            # Arousal regression  
            ar_output = Dense(1, activation='sigmoid', name='arousal')(x)
            
            model = Model(inputs, [expr_output, val_output, ar_output])
        else:
            outputs = Dense(self.config.NUM_CLASSES, activation='softmax')(x)
            model = Model(inputs, outputs)
        
        return model

# Add this to your existing CNNModels class:
def get_baseline_model(self, baseline_type, multi_output=False):
    """Get baseline model"""
    baseline_factory = BaselineModels(self.config)
    
    if baseline_type == 'simple_cnn':
        return baseline_factory.simple_cnn_baseline(multi_output)
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")   