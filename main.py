"""
The entry point of the application. 
It will call other modules for training, evaluation, and prediction.
"""
import argparse
import tensorflow as tf
from train import train_model
from evaluate import load_dataset
from evaluate import evaluate_model
from model import GPT
import predict

def main():
    mode = input("Enter mode (train/evaluate/predict): ")

    if mode == 'train':
        print("Starting training...")
        output_dir = input("Enter output directory path: ")
        checkpoint_dir = input("Enter checkpoint directory path: ")
        epochs = int(input("Enter number of epochs: "))
        batch_size = int(input("Enter batch size: "))
        max_length = int(input("Enter maximum length of input sequences: "))
        learning_rate = float(input("Enter learning rate: "))
        train_model(output_dir, checkpoint_dir, epochs, batch_size, max_length, learning_rate)

    elif mode == 'evaluate':
        print("Evaluating model...")
        model_path = input("Enter path to saved model: ")
        output_dir = input("Enter output directory path for test dataset: ")
        batch_size = int(input("Enter batch size: "))
        max_length = int(input("Enter maximum length of input sequences: "))
        model = tf.keras.models.load_model(model_path, custom_objects={'GPT': GPT})
        test_dataset = load_dataset(output_dir, batch_size, max_length, split_ratio=1.0)
        avg_loss, perplexity = evaluate_model(model, test_dataset)
        print(f"Average Loss: {avg_loss}, Perplexity: {perplexity}")

    elif mode == 'predict':
        model_path = input("Enter path to saved model: ")
        input_text = input("Enter input text for prediction: ")
        max_length = int(input("Enter maximum length for generated text: "))
        model = predict.load_model(model_path)
        output = predict.predict(model, input_text, max_length)
        print("Generated Text: ", output)

    else:
        print("Invalid mode selected. Please run the script again with one of these modes: train, evaluate, predict.")


if __name__ == "__main__":
    main()
