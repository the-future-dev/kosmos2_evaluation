import string
import json
from transformers import pipeline

bart_pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

labels = []

def init_w_labels(w_labels):
    global labels
    labels = w_labels

def preprocess_string_from_model(user_input_text, model_output_text):
    processed_output = model_output_text.split(
        "primary object in the provided image is"
    )[1]

    processed_output = processed_output.strip()
    # processed_output = processed_output.replace(f'[{string.punctuation}]', '', regex=True)
    processed_output = processed_output if processed_output else "undefined"
    return processed_output

def classify_string(text):
    result = bart_pipe(text, candidate_labels=labels)
    predicted_label = result['labels'][0]
    return predicted_label
