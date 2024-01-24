"""
 For making predictions using the trained model. 
 It takes an input prompt and generates text based on it.
"""
import tensorflow as tf
from model import GPT
from config import Config
from transformers import GPT2Tokenizer

def load_model(model_path):
    """
    Load the trained GPT model from the specified path.
    """
    model = tf.keras.models.load_model('path_to_saved_model', custom_objects={'GPT': GPT})  # Include custom objects if necessary
    return model

def predict(model, input_text, max_length=50):
    """
    Generate text using the trained GPT model.

    Args:
    model: The trained GPT model.
    input_text: A string, the initial text to start generating from.
    max_length: The maximum length of the generated text.

    Returns:
    A string representing the generated text.
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input_ids = tokenizer.encode(input_text, return_tensors='tf')
    
    # Generate text
    generated_ids = model.generate(input_ids, max_length)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return generated_text

# Example usage
if __name__ == "__main__":
    model_path = '/path/to/saved/model'
    input_text = "Today is a beautiful day"
    model = load_model(model_path)
    generated_text = predict(model, input_text)
    print(generated_text)
