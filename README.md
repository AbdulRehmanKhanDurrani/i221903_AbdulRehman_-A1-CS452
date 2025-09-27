# i221903_AbdulRehman_-A1-CS452
This project explores deep learning techniques for facial emotion recognition and affect estimation. Multiple architectures were implemented and compared, including VGG16, ResNet50, MobileNetV2, InceptionV3, and a custom CNN model.

The models were trained using transfer learning with ImageNet-pretrained backbones (for standard architectures) and from-scratch training (for the custom model). The training objective included both categorical emotion classification (e.g., expressions) and continuous affect estimation (valence and arousal).

Key features of the project:

Architectures Implemented: VGG16, ResNet50, MobileNetV2, InceptionV3, Custom CNN

Training Strategy: Transfer learning (freezing & fine-tuning) and full training for custom model

Evaluation Metrics: Accuracy, F1-score, Cohen’s Kappa, RMSE, Correlation, CCC, and SAGR

Results: Comparison across models highlights trade-offs between accuracy, computational cost, and generalization ability

Applications: Emotion recognition, affective computing, human-computer interaction, and real-time affect monitoring

This repository contains model definitions, training scripts, evaluation results, and visualization plots.
