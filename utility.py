"""
Contains helper functions
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
from sklearn.metrics import precision_score, recall_score, f1_score

def setup_logger(name, log_file, level=logging.INFO):
    """
    Creates a logging instance.
    """
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def plot_training_history(history):
    """
    Plot training and validation loss/accuracy graphs.
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.show()

def calculate_additional_metrics(predictions, labels):
    """
    Calculate precision, recall, and F1-score.
    
    Args:
    predictions: A list or array of predicted labels.
    labels: A list or array of true labels.
    
    Returns:
    A dictionary containing precision, recall, and F1-score.
    """
    # Convert predictions and labels to 1D arrays
    predictions = tf.reshape(predictions, [-1]).numpy()
    labels = tf.reshape(labels, [-1]).numpy()

    # Calculate metrics
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
