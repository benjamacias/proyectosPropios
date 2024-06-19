import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Example dataset
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
max_len = max(max_input_len, max_target_len)

input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_len, padding='post')

# Convert target sequences to categorical
target_sequences = tf.keras.utils.to_categorical(target_sequences, num_classes=vocab_size)

# Model
model = Sequential([
    Embedding(vocab_size, 64, input_length=max_len),
    LSTM(100, return_sequences=True),
    Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(input_sequences, target_sequences, epochs=50, verbose=1)

# Generate response function
def generate_response(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])
    padded_input = pad_sequences(input_seq, maxlen=max_input_len, padding='post')
    predicted_output = model.predict(padded_input)
    predicted_word_index = np.argmax(predicted_output[0], axis=-1)
    response = ' '.join(tokenizer.index_word[idx] for idx in predicted_word_index if idx != 0)
    return response

# Test the chatbot
user_input = "Hello"
response = generate_response(user_input)
print(f"User: {user_input}")
print(f"Chatbot: {response}")
