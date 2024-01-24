"""
Handles the training process.
It should use the model defined in model.py and the data prepared by data_loader.py. 
This file will include the training loop, loss function definition, and optimization strategy.
"""
import tensorflow as tf
from config import Config
from model import GPT
import os

output_dir = '/path/to/TFRecord/output'
checkpoint_dir = '/path/to/training_checkpoints'

def _parse_function(proto):
    keys_to_features = {'text': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)}
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    return parsed_features['text']

def load_dataset(file_pattern, batch_size, max_length, split_ratio=0.1):
    files = tf.io.gfile.glob(file_pattern)
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(_parse_function)
    
    # Shuffle and split the dataset into training and validation sets
    dataset = dataset.shuffle(100)
    val_size = int(split_ratio * len(files))
    val_dataset = dataset.take(val_size).padded_batch(batch_size, padded_shapes=(max_length,))
    train_dataset = dataset.skip(val_size).padded_batch(batch_size, padded_shapes=(max_length,))

    return train_dataset, val_dataset

def train_model(output_dir, checkpoint_dir, epochs=5, batch_size=4, max_length=1025, learning_rate=0.001):
    config = Config()
    model = GPT(config)
    train_dataset, val_dataset = load_dataset(output_dir + "/*.tfrecord", batch_size, max_length)
    PREFETCH_SIZE = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.prefetch(PREFETCH_SIZE)
    val_dataset = val_dataset.prefetch(PREFETCH_SIZE)

    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def train_step(model, inputs, targets):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            targets = tf.reshape(targets, shape=[-1])
            loss = loss_function(targets, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    @tf.function
    def val_step(model, inputs, targets):
        # Validation step implementation
        predictions = model(inputs, training=False)
        loss = loss_function(targets, predictions)
        return loss

    checkpoint_callback = tf.train.Checkpoint(optimizer=optimizer, model=model)

    for epoch in range(epochs):
        # Training phase
        for batch in train_dataset:
            inputs, targets = batch[:, :-1], batch[:, 1:]
            loss = train_step(model, inputs, targets)
            print(f"Train Epoch {epoch}, Loss: {loss.numpy()}")

        # Validation phase
        total_val_loss = 0
        num_batches = 0
        for batch in val_dataset:
            inputs, targets = batch[:, :-1], batch[:, 1:]
            predictions = model(inputs, training=False)  # No training during validation
            val_loss = loss_function(targets, predictions)
            total_val_loss += val_loss.numpy()
            num_batches += 1
        avg_val_loss = total_val_loss / num_batches
        print(f"Validation Epoch {epoch}, Avg Loss: {avg_val_loss}")

        # Save checkpoint
        checkpoint_file = os.path.join(checkpoint_dir, f"ckpt_{epoch}")
        checkpoint_callback.save(file_prefix=checkpoint_file)

    # Save the final model after training
    model.save(os.path.join(checkpoint_dir, 'my_model'))


# config = Config()
# model = GPT(config)
# batch_size = 4
# max_length = 1025
# train_dataset, val_dataset = load_dataset(output_dir + "/*.tfrecord", batch_size, max_length)
# PREFETCH_SIZE = tf.data.experimental.AUTOTUNE
# train_dataset = train_dataset.prefetch(PREFETCH_SIZE)
# val_dataset = val_dataset.prefetch(PREFETCH_SIZE)

# loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# @tf.function
# def train_step(model, inputs, targets):
#     with tf.GradientTape() as tape:
#         predictions = model(inputs, training=True)
#         targets = tf.reshape(targets, shape=[-1])
#         #print(targets.shape, predictions.shape)
#         loss = loss_function(targets, predictions)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     return loss

# @tf.function
# def val_step(model, inputs, targets):
#     # Validation step implementation
#     predictions = model(inputs, training=False)
#     loss = loss_function(targets, predictions)
#     return loss

# EPOCHS = 1  # Set the number of epochs

# #Checkpoint directory
# checkpoint_dir = '/path/to/training_checkpoints'
# #checkpoint_prefix = os.path.join(checkpoint_dir, f"ckpt_{epoch}")
# checkpoint_callback = tf.train.Checkpoint(optimizer=optimizer, model=model)

# for epoch in range(EPOCHS):
#     # Training phase
#     for batch in train_dataset:
#         inputs, targets = batch[:, :-1], batch[:, 1:]
#         loss = train_step(model, inputs, targets)
#         print(f"Train Epoch {epoch}, Loss: {loss.numpy()}")

#     # Validation phase
#     total_val_loss = 0
#     num_batches = 0
#     for batch in val_dataset:
#         inputs, targets = batch[:, :-1], batch[:, 1:]
#         predictions = model(inputs, training=False)  # No training during validation
#         val_loss = loss_function(targets, predictions)
#         total_val_loss += val_loss.numpy()
#         num_batches += 1
#     avg_val_loss = total_val_loss / num_batches
#     print(f"Validation Epoch {epoch}, Avg Loss: {avg_val_loss}")

#     # Save checkpoint
#     checkpoint_file = os.path.join(checkpoint_dir, f"ckpt_{epoch}")
#     checkpoint_callback.save(file_prefix=checkpoint_file)
# # Save the final model after training
# model.save('/path/to/model/saving/dir')