import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy

# Load the classification model
model = tf.keras.models.load_model("model/email_classifier_model.h5")

# Load the tokenizer
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load the label encoder
with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Max length used during model training
max_length = 100

nlp = spacy.load("en_core_web_sm")

# Patterns for sensitive data
patterns = {
    'credit_debit_no': r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    'aadhar_num': r"\b\d{12}\b",
    'cvv_no': r"\b\d{3}\b",
    'expiry_no': r"\b\d{2}/\d{2}\b",
    'phone_number': r"\b\d{10}\b",
    'dob': r"\b\d{2}/\d{2}/\d{4}\b"
}

original_pii = {
    'full_name': [],
    'email': [],
    'credit_debit_no': [],
    'aadhar_num': [],
    'cvv_no': [],
    'expiry_no': [],
    'phone_number': [],
    'dob': []
}

def mask_pii(text):
    # First: Use spaCy for PERSON and EMAIL (before any replacements)
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == 'PERSON' and ent.text not in original_pii['full_name']:
            original_pii['full_name'].append(ent.text)
            text = text.replace(ent.text, '[FULL_NAME]')
        elif ent.label_ == 'EMAIL' and ent.text not in original_pii['email']:
            original_pii['email'].append(ent.text)
            text = text.replace(ent.text, '[EMAIL]')

    # Second: Use regex for other PII
    for tag, pattern in patterns.items():
        matches = re.findall(pattern, text)
        for match in matches:
            if match not in original_pii[tag]:
                original_pii[tag].append(match)
                text = text.replace(match, f'[{tag.upper()}]')

    return text



def get_masked_entities():
    """
    Collect and return all masked entities with position and type.
    """
    entity_list = []
    for tag, values in original_pii.items():
        for value in values:
            placeholder = f'[{tag.upper()}]'
            entity_list.append({
                "position": None,  # You can update this if needed by calculating index
                "classification": tag,
                "entity": value
            })
    return entity_list

# Email classification function
def predict_email_category(subject: str, body: str) -> str:
    text = subject + " " + body
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_length, padding="post")
    prediction = model.predict(padded)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]
    return predicted_label

def process_email(subject: str, body: str):
    masked_subject = mask_pii(subject)
    masked_body = mask_pii(body)
    category = predict_email_category(subject, body)
    entities = get_masked_entities()

    return {
        "subject": subject,
        "body": body,
        "masked_subject": masked_subject,
        "masked_body": masked_body,
        "category": category,
        "list_of_masked_entities": entities
    }

