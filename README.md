
# 📧 Email Classification & PII Masking API

A deep learning-based project that classifies IT service emails (e.g., **Incident**, **Request**, **Change**) while protecting user privacy by masking and restoring sensitive **Personally Identifiable Information (PII)**. Built with **TensorFlow**, **FastAPI**, and **spaCy**,**regex**.

---

## 🔍 Features

- ✅ Classifies emails into Incident, Request, or Change.
- 🔐 Detects and masks PII (Email, Phone, Aadhar, etc.) using regex + NLP
- 🔁 Restores original PII after classification
- 🧠 LSTM-based model with tokenizer and label encoder
- ⚡ FastAPI for real-time prediction API

---

## 🧠 Model Overview

| Attribute     | Details                          |
|---------------|----------------------------------|
| **Model Type**| Bidirectional LSTM (Keras)       |
| **Input**     | Combined email subject & body    |
| **Output**    | Email Type (incident/request/...)|
| **Accuracy**  | ~70% (tunable to 85%+)           |
| **Dataset**   | 24,000 labeled support emails    |

---


# 📧 Email Classification & PII Masking API

A deep learning-based project that classifies IT service emails (e.g., **Incident**, **Request**, **Change**) while protecting user privacy by masking and restoring sensitive **Personally Identifiable Information (PII)**. Built with **TensorFlow**, **FastAPI**, and **spaCy**.

---

## 🔍 Features

- ✅ Classifies emails into Incident, Request, or Change
- 🔐 Detects and masks PII (Email, Phone, Aadhar, etc.) using regex + NLP
- 🔁 Restores original PII after classification
- 🧠 LSTM-based model with tokenizer and label encoder
- ⚡ FastAPI for real-time prediction API

---

## 🧠 Model Overview

| Attribute     | Details                          |
|---------------|----------------------------------|
| **Model Type**| Bidirectional LSTM (Keras)       |
| **Input**     | Combined email subject & body    |
| **Output**    | Email Type (incident/request/...)|
| **Accuracy**  | ~70% (tunable to 85%+)           |
| **Dataset**   | 24,000 labeled support emails    |

---

## 🗂 Project Structure

- app.py (or equivalent main script)
- Requirements file (requirements.txt or environment.yml)
- README with setup and usage instructions
- models.py containing the model training and utility functions
- utils.py containing the utility function and code
- api.py to support the development of APIs.


---

## ⚙️ Setup Instructions


# 1. Clone the repo
- git clone https://github.com/hannafarsin/email-classifier-pii-api.git
- cd email-classifier-pii-api

# 2. Create and activate virtual environment
conda create -n email_env python=3.10
conda activate email_env

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the API
uvicorn main:app --reload


# Sample API Request & Response
✅ Request
{
  "subject": "Urgent: Credit card issue",
  "body": "Hello, my name is John Doe. My credit card number is 1234 5678 9012 3456 and CVV is 123. Please help!"
}

✅ Response

{
  "input_email_body": "Hello, my name is John Doe. My credit card number is 1234 5678 9012 3456 and CVV is 123. Please help!",
  "masked_email": "Hello, my name is [FULL_NAME]. My credit card number is [CREDIT_DEBIT_NO] and CVV is [CVV_NO]. Please help!",
  "list_of_masked_entities": [
    {
      "position": [18, 26],
      "classification": "full_name",
      "entity": "John Doe"
    },
    {
      "position": [51, 70],
      "classification": "credit_debit_no",
      "entity": "1234 5678 9012 3456"
    },
    {
      "position": [82, 85],
      "classification": "cvv_no",
      "entity": "123"
    }
  ],
  "category_of_the_email": "Incident"
}

# Model Training
To train or retrain the model:
python models.py

Trains the LSTM classifier and saves the model to model/email_classifier_model.h5.

# 🛠 Tech Stack
Python 3.10

TensorFlow / Keras

FastAPI

spaCy (NLP for PII detection)

Regex for pattern-based masking

scikit-learn (LabelEncoder)


