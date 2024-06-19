import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

conversations = [
    ("Hello", "Hi there!"),
    ("How are you?", "I'm doing well, thanks."),
    ("What's your name?", "I'm a chatbot."),
    # Add more conversational pairs as needed
]

# Prepare the dataset
input_texts = []
target_texts = []

for input_text, target_text in conversations:
    input_texts.append(input_text)
    target_texts.append(target_text)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(input_texts + target_texts)
vocab_size = len(tokenizer.word_index) + 1

input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

max_input_len = max([len(seq) for seq in input_sequences])
max_target_len = max([len(seq) for seq in target_sequences])

input_sequences = pad_sequences(input_sequences, maxlen=max_input_len, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_target_len, padding='post')

# Model
model = Sequential([
    Embedding(vocab_size, 64, input_length=max_input_len),
    LSTM(100, return_sequences=False),
    Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Convert target sequences to a shape compatible with sparse_categorical_crossentropy
target_sequences = np.array(target_sequences)
target_sequences = target_sequences.reshape(-1, max_target_len)

# Use only the first word of the target sequence as the prediction target for simplicity
target_sequences = target_sequences[:, 0]

# Train the model
model.fit(input_sequences, target_sequences, epochs=60, verbose=1)

# Generate response function
def generate_response(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])
    padded_input = pad_sequences(input_seq, maxlen=max_input_len, padding='post')
    predicted_output = model.predict(padded_input)
    predicted_word_index = np.argmax(predicted_output[0])
    response = tokenizer.sequences_to_texts([[predicted_word_index]])[0]
    return response

# Test the chatbot
user_input = "What's your name?"
response = generate_response(user_input)
print(f"User: {user_input}")
print(f"Chatbot: {response}")
