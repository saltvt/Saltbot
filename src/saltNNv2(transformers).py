import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

torch.manual_seed(42)
np.random.seed(42)

# Sample chatbot training data (Replace this with your own data)
conversations = [
    ("hello", "hi there!"),
    ("how are you?", "I'm just a bot, but I'm doing great!"),
    ("what is your name?", "I'm SaltBot!"),
    ("bye", "Goodbye! Have a great day!"),
]

# Load pre-trained model and tokenizer
model_name = 't5-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create vocabulary (word -> index)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
    return words

# Encode the dataset
training_data = [(q, a) for q, a in conversations]

# Function for chatbot interaction
def chat():
    print("SaltBot is ready! Type 'exit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input == "exit":
            print("SaltBot: Goodbye!")
            break
        
        # Tokenize user input
        inputs = tokenizer(user_input, return_tensors="pt")

        # Generate response
        outputs = model.generate(**inputs)

        # Convert response to text
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("SaltBot:", response)

# Run chatbot
if __name__ == "__main__":
    chat()