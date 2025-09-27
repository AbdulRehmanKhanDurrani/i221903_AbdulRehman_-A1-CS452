"""
Configuration module for Facial Expression Recognition
Contains all hyperparameters and settings
"""

import os

class Config:
    # Dataset paths
    DATASET_PATH = "dataset"
    ANNOTATIONS_PATH = "dataset/annotations"
    IMAGES_PATH = "dataset/images"
    
    # Image settings
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    CHANNELS = 3
    
    # Training settings
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    
    # Model settings
    NUM_CLASSES = 8  # 8 emotion categories
    DROPOUT_RATE = 0.5
    
    # Data augmentation settings
    ROTATION_RANGE = 20
    WIDTH_SHIFT_RANGE = 0.2
    HEIGHT_SHIFT_RANGE = 0.2
    HORIZONTAL_FLIP = True
    ZOOM_RANGE = 0.2
    
    # Output settings
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Emotion labels
    EMOTION_LABELS = {
        0: "Neutral",
        1: "Happy", 
        2: "Sad",
        3: "Surprise",
        4: "Fear",
        5: "Disgust",
        6: "Anger",
        7: "Contempt"
    }
    
    # Create directories if they don't exist
    @staticmethod
    def create_directories():
        os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(Config.RESULTS_PATH, exist_ok=True)
