import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)

# Sample chatbot training data (Replace this with your own data)
conversations = [
    ("hello", "hi there!"),
    ("how are you?", "I'm just a bot, but I'm doing great!"),
    ("what is your name?", "I'm SaltBot!"),
    ("bye", "Goodbye! Have a great day!"),
]



# Create vocabulary (word -> index)
vocab = {word: i for i, (q, a) in enumerate(conversations) for word in q.split() + a.split()}
vocab_size = len(vocab) + 1  # +1 for padding index

# Function to convert sentences into sequences of indice
def encode_sentence(sentence):
    return [vocab[word] for word in sentence.split() if word in vocab]

# Encode the dataset
training_data = [(encode_sentence(q), encode_sentence(a)) for q, a in conversations]

# Determine the maximum sequence length
max_len = max(max(len(q), len(a)) for q, a in training_data)

# Pad sequences to match max_len
def pad_sequence(seq, max_len):
    return seq + [0] * (max_len - len(seq))

X_train = torch.tensor([pad_sequence(q, max_len) for q, _ in training_data], dtype=torch.long)
y_train = torch.stack([torch.tensor(pad_sequence(a, max_len), dtype=torch.long) for _, a in training_data])

# Define the chatbot model
class SaltBot(nn.Module):
    def __init__(self, vocab_size, embedding_dim=8, hidden_dim=16):
        super(SaltBot, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Take the last output from LSTM
        return x

# Initialize the model
model = SaltBot(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train[:, -1])  # Compare final output to last word in response
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Function for chatbot interaction
def chat():
    print("SaltBot is ready! Type 'exit' to stop.")
    while True:
        user_input = input("You: ").lower()
        if user_input == "exit":
            print("SaltBot: Goodbye!")
            break
        
        # Encode user input
        encoded_input = pad_sequence(encode_sentence(user_input), max_len)
        input_tensor = torch.tensor([encoded_input], dtype=torch.long)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_idx = torch.argmax(output, dim=1).item()

        # Find the corresponding word
        response = [word for word, idx in vocab.items() if idx == predicted_idx]
        print("SaltBot:", response[0] if response else "I don't understand.")

# Run chatbot
if __name__ == "__main__":
    chat()