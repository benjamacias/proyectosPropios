import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed, MultiHeadAttention
from keras.layers import Dropout
from tensorflow.keras.models import load_model

def nonlin(x, deriv=False):
    """Función sigmoide como función de activación."""
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

#  dataset
conversations = [
    ("hola", "hola, buenas dias"),
    ("r1, cuadrado, cuadrado", "hesoyam"),
    ("bien", "Me alegro! En que puedo ayudarte?"),
    ("que haces?", "Estoy aqui para ayudarte"),
    ("quien eres?", "Soy un chatbot, tu?"),
    ("como estas?", "Bien, porque has preguntado?"),
    ("cual es tu nombre?", "Soy genv 0.2"),
    ("cuanto es 2+2", "4"),
    ("adios","adios!"),
    ("5+5", "10"),
    ("6+6", "12"),
]

# Preparar dataset
input_texts = []
target_texts = []

for input_text, target_text in conversations:
    input_texts.append(input_text)
    target_texts.append(target_text)

# Tokenizacion
filters='#&();<>@[\\]^_`{|}~\t\n'
tokenizer = Tokenizer(filters=filters)
tokenizer.fit_on_texts(input_texts + target_texts)
vocab_size = len(tokenizer.word_index) + 1

input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

max_input_len = max([len(seq) for seq in input_sequences])
max_target_len = max([len(seq) for seq in target_sequences])
max_len = max(max_input_len, 4)  # Replace 4 with the actual length of your target sequences

input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_len, padding='post')

def generarModelo(vocab_size, max_len):
    # Model
    model = Sequential([
        Embedding(vocab_size, 1800, input_length=max_len),  # Increase the dimensionality of the embeddings
        LSTM(900, return_sequences=True),  # Increase the number of units in the LSTM layer
        Dropout(0.1),  # Add a dropout layer to reduce overfitting
        TimeDistributed(Dense(vocab_size, activation='softmax'))
    ])

    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Use a different optimizer

    # Train the model
    model.fit(input_sequences, target_sequences, epochs=180, verbose=1, batch_size=64)  # Increase the number of epochs and adjust the batch size

    return model
    

def generate_response(input_text, model, tokenizer, max_input_len):
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

user_input = input()

if user_input != "cargar":
    model = generarModelo(vocab_size, max_len)
    # Test the chatbot
    while user_input != "adios":
        user_input = input()
        response = generate_response(user_input, model, tokenizer, max_len)
        print(f"User: {user_input}")
        print(f"Chatbot: {response}")
else:
    # Load the model
    model = load_model('my_model.keras')

    # Re-train the model
    #model.fit(input_sequences, target_sequences, epochs=180, verbose=1, batch_size=64)
    while user_input != "adios":
        user_input = input()
        response = generate_response(user_input, model, tokenizer, max_len)
        print(f"User: {user_input}")

model.save('my_model.keras')