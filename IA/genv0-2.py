import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed
from keras.layers import Dropout


# Example dataset
conversations = [
    ("hola", "Hola, como estas?"),
    ("como estas?", "Bien, porque has preguntado?"),
    ("cual es tu nombre?", "Soy genv 0.2"),
    ("cuanto es 2+2", "4"),
    ("5+5", "10"),
    ("6+6", "12"),
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
max_len = max(max_input_len, 4)  # Replace 4 with the actual length of your target sequences

input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_len, padding='post')

# Model
"""
model = Sequential([
    Embedding(vocab_size, 64, input_length=max_len),
    LSTM(100, return_sequences=True),
    TimeDistributed(Dense(vocab_size, activation='softmax'))
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(input_sequences, target_sequences, epochs=110, verbose=1)
"""

# Model
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_len),  # Increase the dimensionality of the embeddings
    LSTM(200, return_sequences=True),  # Increase the number of units in the LSTM layer
    Dropout(0.5),  # Add a dropout layer to reduce overfitting
    TimeDistributed(Dense(vocab_size, activation='softmax'))
])

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Use a different optimizer

# Train the model
model.fit(input_sequences, target_sequences, epochs=200, verbose=1, batch_size=64)  # Increase the number of epochs and adjust the batch size

def generate_response(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])
    padded_input = pad_sequences(input_seq, maxlen=max_input_len, padding='post')
    predicted_output = model.predict(padded_input)[0]

    response = []
    for word_prob in predicted_output:
        predicted_word_index = np.argmax(word_prob)
        if predicted_word_index == 0:  # Padding or end of sentence
            # Get the next most probable word
            predicted_word_index = np.argsort(word_prob)[-2]
        response.append(predicted_word_index)
    
    # Convert the list of indices to words
    response_words = tokenizer.sequences_to_texts([response])[0].split(' ')
    
    # Join the words into a sentence
    return ' '.join(response_words)

# Test the chatbot
user_input = "cuanto es 2+2"
response = generate_response(user_input)
print(f"User: {user_input}")
print(f"Chatbot: {response}")
