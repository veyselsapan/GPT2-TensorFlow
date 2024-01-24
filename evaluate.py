"""
Contains code for evaluating the trained model. 
"""
import tensorflow as tf
from data_loader import load_dataset
from model import GPT

def evaluate_model(model, dataset):
    """
    Evaluate the model on the given dataset.
    """
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    total_loss = 0
    num_batches = 0

    for batch in dataset:
        inputs, targets = batch[:, :-1], batch[:, 1:]
        predictions = model(inputs, training=False)
        loss = loss_fn(targets, predictions)
        total_loss += loss
        num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = tf.exp(avg_loss)

    return avg_loss, perplexity.numpy()

# Load the saved model
model = tf.keras.models.load_model('path_to_saved_model', custom_objects={'GPT': GPT})  # Include custom objects if necessary

# Adjust the parameters
batch_size = 32 
max_length = 1025

# Load the test dataset
test_dataset = load_dataset('path_to_test_data', batch_size, max_length, split_ratio=1.0)  # Load only validation data

# Evaluate the model
avg_loss, perplexity = evaluate_model(model, test_dataset)
print(f"Average Loss: {avg_loss}, Perplexity: {perplexity}")
