"""
Model definitions for Swara keyword spotting / speech recognition.
Designed for low-latency, low-memory edge deployments (TFLite / Microcontrollers).
"""

from typing import Tuple
from config import DEFAULT_INPUT_SHAPE, DEFAULT_NUM_CLASSES, DEFAULT_NUM_FILTERS


def build_dscnn_model(
    input_shape: Tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
    num_filters: int = DEFAULT_NUM_FILTERS,
):

    """
    Build a Depthwise Separable Convolutional Neural Network (DS-CNN).
    Optimized for Swara Frozen Audio Spec V0 (49 frames x 10 MFCCs).
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models

        model = models.Sequential([
            layers.Input(shape=input_shape),
            # Standard 2D Conv layer (strided over time and frequency)
            layers.Conv2D(num_filters, (5, 3), strides=(2, 1), padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            # Depthwise Separable Conv Block 1
            layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(num_filters, (1, 1), padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            # Depthwise Separable Conv Block 2
            layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(num_filters, (1, 1), padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            # Pooling & Classification Head
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax", name="output"),
        ])
        return model
    except ImportError:
        print("TensorFlow not installed. Please install tensorflow to build the model.")
        return None


if __name__ == "__main__":
    model = build_dscnn_model()
    if model:
        model.summary()
