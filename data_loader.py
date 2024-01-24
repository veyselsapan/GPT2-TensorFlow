"""
Manages the loading, preprocessing and saving of the dataset
"""
import os
import lzma
import re
import tensorflow as tf
import tarfile
from transformers import GPT2Tokenizer

def extract_tar_xz(file_path, extract_to):
    with tarfile.open(file_path, "r:xz") as tar:
        tar.extractall(path=extract_to)

def extract_texts(file_path):
    with lzma.open(file_path, mode='rt') as file:
        return file.read()

def tokenize_text(text):
    return tokenizer.encode(text, add_special_tokens=True)

def convert_to_tfrecord(tokenized_text, filename):
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    feature = {'text': _int64_feature(tokenized_text)}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized_example = example.SerializeToString()

    with tf.io.TFRecordWriter(filename) as writer:
        writer.write(serialized_example)

def split_text_to_chunks(text, chunk_size=MAX_LENGTH):
    # Split the text into chunks of up to chunk_size tokens
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def process_and_save_text(text, file_identifier, output_dir):
    tokenized_text = tokenizer.encode(text, add_special_tokens=True)

    # Splitting the tokenized text into chunks
    tokenized_chunks = split_text_to_chunks(tokenized_text)

    for i, chunk in enumerate(tokenized_chunks):
        tfrecord_filename = os.path.join(output_dir, f'{file_identifier}_part{i}.tfrecord')
        convert_to_tfrecord(chunk, tfrecord_filename)


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Paths
tar_xz_path = '/path/to/openwebtexty.tar.xz'
extracted_dir = '/path/to/openwebtext/output/directory'
output_dir = '/path/to/TFRecord/output'

# Extract the dataset
extract_tar_xz(tar_xz_path, extracted_dir)

# Update the dataset_dir to the extracted directory
dataset_dir = extracted_dir

# Max length for GPT-2. Max length infact is 1024. 
# However, I set it to 1025 because I will shift the target 1 to the right of the input.
MAX_LENGTH = 1025 

# Data extraction, processing and converting loop
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith('.xz'):
            file_path = os.path.join(root, file)
            try:
                text = extract_texts(file_path)
                file_identifier = os.path.splitext(file)[0]  # Extract file name without extension
                process_and_save_text(text, file_identifier, output_dir)
                print(f"Processed file: {file}")
            except Exception as e:
                print(f"Error processing file {file}: {e}")
