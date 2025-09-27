"""
Data loading and preprocessing module
Handles loading images and annotations from .npy files
"""

import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from config import Config

class DataLoader:
    def __init__(self, config=None):
        self.config = config or Config()
        self.images = None
        self.expressions = None
        self.valence = None
        self.arousal = None
        self.expressions_categorical = None
        
    def load_annotations(self):
        """Load all annotation files from the annotations directory"""
        annotations_path = self.config.ANNOTATIONS_PATH
        
        try:
            # Get all .npy files
            all_files = [f for f in os.listdir(annotations_path) if f.endswith('.npy')]
            
            # Group files by type and extract indices
            expression_data = {}
            valence_data = {}
            arousal_data = {}
            
            print("Analyzing annotation files...")
            
            for file in all_files:
                file_path = os.path.join(annotations_path, file)
                try:
                    data = np.load(file_path, allow_pickle=True)
                    
                    # Extract index from filename (e.g., "123_exp.npy" -> 123)
                    base_name = os.path.splitext(file)[0]
                    parts = base_name.split('_')
                    
                    if len(parts) >= 2:
                        try:
                            index = int(parts[0])
                            file_type = parts[1].lower()
                            
                            # Store data by type and index
                            if file_type == 'exp':
                                # Convert string to integer if needed
                                if isinstance(data.item() if data.shape == () else data, str):
                                    try:
                                        expression_data[index] = int(data.item())
                                    except ValueError:
                                        print(f"Warning: Could not convert expression '{data.item()}' to integer for {file}")
                                else:
                                    expression_data[index] = int(data.item() if data.shape == () else data)
                                    
                            elif file_type == 'val':
                                # Convert string to float if needed
                                if isinstance(data.item() if data.shape == () else data, str):
                                    try:
                                        valence_data[index] = float(data.item())
                                    except ValueError:
                                        print(f"Warning: Could not convert valence '{data.item()}' to float for {file}")
                                else:
                                    valence_data[index] = float(data.item() if data.shape == () else data)
                                    
                            elif file_type == 'aro':
                                # Convert string to float if needed
                                if isinstance(data.item() if data.shape == () else data, str):
                                    try:
                                        arousal_data[index] = float(data.item())
                                    except ValueError:
                                        print(f"Warning: Could not convert arousal '{data.item()}' to float for {file}")
                                else:
                                    arousal_data[index] = float(data.item() if data.shape == () else data)
                                    
                        except ValueError:
                            print(f"Warning: Could not extract index from filename: {file}")
                            
                except Exception as e:
                    print(f"Error loading {file}: {e}")
            
            # Convert to arrays, sorted by index
            if expression_data:
                max_index = max(expression_data.keys())
                min_index = min(expression_data.keys())
                print(f"Expression data: indices {min_index} to {max_index} ({len(expression_data)} files)")
                
                # Create array with NaN for missing indices
                self.expressions = np.full(max_index + 1, -1, dtype=int)
                for idx, value in expression_data.items():
                    self.expressions[idx] = value
                
                # Remove entries with -1 (missing data) - keep only valid indices
                valid_expr_indices = [i for i in range(len(self.expressions)) if self.expressions[i] != -1]
                self.expressions = self.expressions[valid_expr_indices]
                self.valid_indices = valid_expr_indices
                
                print(f"Loaded {len(self.expressions)} expression labels")
            
            if valence_data:
                max_index = max(valence_data.keys())
                print(f"Valence data: indices {min(valence_data.keys())} to {max_index} ({len(valence_data)} files)")
                
                self.valence = np.full(max_index + 1, np.nan)
                for idx, value in valence_data.items():
                    self.valence[idx] = value
                
                if hasattr(self, 'valid_indices'):
                    self.valence = self.valence[self.valid_indices]
                else:
                    # Remove NaN values
                    valid_val_mask = ~np.isnan(self.valence)
                    self.valence = self.valence[valid_val_mask]
                    
                print(f"Loaded {len(self.valence)} valence values")
            
            if arousal_data:
                max_index = max(arousal_data.keys())
                print(f"Arousal data: indices {min(arousal_data.keys())} to {max_index} ({len(arousal_data)} files)")
                
                self.arousal = np.full(max_index + 1, np.nan)
                for idx, value in arousal_data.items():
                    self.arousal[idx] = value
                
                if hasattr(self, 'valid_indices'):
                    self.arousal = self.arousal[self.valid_indices]
                else:
                    # Remove NaN values
                    valid_ar_mask = ~np.isnan(self.arousal)
                    self.arousal = self.arousal[valid_ar_mask]
                    
                print(f"Loaded {len(self.arousal)} arousal values")
                
        except Exception as e:
            print(f"Error loading annotations: {e}")
            print("Available .npy files in annotations directory:")
            for file in os.listdir(annotations_path):
                if file.endswith('.npy'):
                    print(f"  {file}")

    def load_images_from_directory(self, images_dir):
        """Load images from directory structure"""
        images = []
        image_files = []
        
        # Get all image files
        for root, dirs, files in os.walk(images_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))
        
        # Sort numerically if files are numbered (e.g., 1.jpg, 2.jpg, ..., 10.jpg)
        def numerical_sort(filename):
            basename = os.path.basename(filename)
            name_without_ext = os.path.splitext(basename)[0]
            try:
                return int(name_without_ext)
            except ValueError:
                return float('inf')  # Put non-numeric files at the end
        
        image_files.sort(key=numerical_sort)
        
        print(f"Loading {len(image_files)} images...")
        print(f"First few files: {[os.path.basename(f) for f in image_files[:5]]}")
        
        for i, img_path in enumerate(image_files):
            try:
                # Load and resize image
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    img = cv2.resize(img, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
                    images.append(img)
                    
                    if (i + 1) % 1000 == 0:
                        print(f"Loaded {i + 1} images...")
                else:
                    print(f"Warning: Could not load image {img_path}")
                        
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        print(f"Successfully loaded {len(images)} images")
        
        # If we have valid indices from expression loading, filter images accordingly
        if hasattr(self, 'valid_indices') and len(images) > max(self.valid_indices):
            print(f"Filtering images to match annotation indices...")
            filtered_images = [images[i] for i in self.valid_indices if i < len(images)]
            print(f"Filtered to {len(filtered_images)} images matching annotations")
            return np.array(filtered_images)
        
        return np.array(images)

    def preprocess_data(self):
        """Preprocess the loaded data"""
        if self.images is None:
            print("No images loaded. Please load images first.")
            return None
        
        # Normalize images to [0, 1]
        self.images = self.images.astype('float32') / 255.0
        
        # Filter out uncertain and no-face samples (valence/arousal = -2)
        if self.valence is not None and self.arousal is not None:
            valid_mask = (self.valence != -2) & (self.arousal != -2) & (~np.isnan(self.valence)) & (~np.isnan(self.arousal))
            
            print(f"Before filtering: {len(self.images)} samples")
            self.images = self.images[valid_mask]
            if self.expressions is not None:
                self.expressions = self.expressions[valid_mask]
            self.valence = self.valence[valid_mask]
            self.arousal = self.arousal[valid_mask]
            
            print(f"After filtering: {len(self.images)} samples remain")
        
        # One-hot encode expressions
        if self.expressions is not None:
            # Ensure expressions are in valid range
            unique_expressions = np.unique(self.expressions)
            print(f"Unique expression values: {unique_expressions}")
            
            # Map expressions to 0-based indices if needed
            if np.max(self.expressions) >= self.config.NUM_CLASSES:
                print(f"Warning: Expression values exceed NUM_CLASSES ({self.config.NUM_CLASSES})")
                # Map to valid range
                self.expressions = self.expressions % self.config.NUM_CLASSES
            
            self.expressions_categorical = to_categorical(self.expressions, self.config.NUM_CLASSES)
            print(f"Expression labels shape: {self.expressions_categorical.shape}")
        
        return True
    
    def split_data(self, test_size=0.2, val_size=0.2):
        """Split data into train, validation, and test sets"""
        # First split: train+val vs test
        train_val_images, test_images = train_test_split(
            self.images, test_size=test_size, random_state=42, stratify=self.expressions
        )
        
        train_val_expr, test_expr = train_test_split(
            self.expressions_categorical, test_size=test_size, random_state=42, stratify=self.expressions
        )
        
        if self.valence is not None:
            train_val_valence, test_valence = train_test_split(
                self.valence, test_size=test_size, random_state=42, stratify=self.expressions
            )
        else:
            train_val_valence, test_valence = None, None
            
        if self.arousal is not None:
            train_val_arousal, test_arousal = train_test_split(
                self.arousal, test_size=test_size, random_state=42, stratify=self.expressions
            )
        else:
            train_val_arousal, test_arousal = None, None
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
        
        train_images, val_images = train_test_split(
            train_val_images, test_size=val_size_adjusted, random_state=42
        )
        
        train_expr, val_expr = train_test_split(
            train_val_expr, test_size=val_size_adjusted, random_state=42
        )
        
        if train_val_valence is not None:
            train_valence, val_valence = train_test_split(
                train_val_valence, test_size=val_size_adjusted, random_state=42
            )
        else:
            train_valence, val_valence = None, None
            
        if train_val_arousal is not None:
            train_arousal, val_arousal = train_test_split(
                train_val_arousal, test_size=val_size_adjusted, random_state=42
            )
        else:
            train_arousal, val_arousal = None, None
        
        print(f"Data split completed:")
        print(f"Train: {len(train_images)} samples")
        print(f"Validation: {len(val_images)} samples") 
        print(f"Test: {len(test_images)} samples")
        
        return {
            'train': {
                'images': train_images,
                'expressions': train_expr,
                'valence': train_valence,
                'arousal': train_arousal
            },
            'val': {
                'images': val_images,
                'expressions': val_expr,
                'valence': val_valence,
                'arousal': val_arousal
            },
            'test': {
                'images': test_images,
                'expressions': test_expr,
                'valence': test_valence,
                'arousal': test_arousal
            }
        }
    
    def create_data_generator(self, x_train, y_train, valence_train=None, arousal_train=None):
        """Create data generator with augmentation"""
        datagen = ImageDataGenerator(
            rotation_range=self.config.ROTATION_RANGE,
            width_shift_range=self.config.WIDTH_SHIFT_RANGE,
            height_shift_range=self.config.HEIGHT_SHIFT_RANGE,
            horizontal_flip=self.config.HORIZONTAL_FLIP,
            zoom_range=self.config.ZOOM_RANGE,
            fill_mode='nearest'
        )
        
        # If we have continuous values (valence/arousal), create multi-output generator
        if valence_train is not None and arousal_train is not None:
            def multi_output_generator():
                generator = datagen.flow(x_train, y_train, batch_size=self.config.BATCH_SIZE)
                val_gen = datagen.flow(x_train, valence_train, batch_size=self.config.BATCH_SIZE)
                ar_gen = datagen.flow(x_train, arousal_train, batch_size=self.config.BATCH_SIZE)
                
                for (x_batch, y_batch), (_, val_batch), (_, ar_batch) in zip(generator, val_gen, ar_gen):
                    yield x_batch, [y_batch, val_batch, ar_batch]
            
            return multi_output_generator()
        else:
            return datagen.flow(x_train, y_train, batch_size=self.config.BATCH_SIZE)
    
    def visualize_samples(self, images, labels, num_samples=16):
        """Visualize sample images with labels"""
        plt.figure(figsize=(12, 8))
        
        for i in range(min(num_samples, len(images))):
            plt.subplot(4, 4, i + 1)
            plt.imshow(images[i])
            
            if len(labels.shape) > 1:  # One-hot encoded
                label_idx = np.argmax(labels[i])
            else:
                label_idx = labels[i]
                
            plt.title(f'{self.config.EMOTION_LABELS[label_idx]}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_PATH, 'sample_images.png'))
        plt.show()
    
    def get_class_distribution(self):
        """Get distribution of classes in the dataset"""
        if self.expressions is None:
            return None
            
        unique, counts = np.unique(self.expressions, return_counts=True)
        
        print("Class distribution:")
        for label, count in zip(unique, counts):
            emotion_name = self.config.EMOTION_LABELS.get(label, f"Unknown_{label}")
            print(f"{emotion_name}: {count} samples ({count/len(self.expressions)*100:.1f}%)")
        
        return dict(zip(unique, counts))