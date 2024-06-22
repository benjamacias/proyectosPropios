import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, MultiHeadAttention, LayerNormalization, Dropout, TimeDistributed, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from collections import deque

# Dataset
conversations = [
    ("hola", "hola, buenos días"),
    ("hola", "buenas tardes"),
    ("hola", "buenas noches"),
    ("hola", "¿cómo estás?"),
    ("bien", "Me alegro! ¿En qué puedo ayudarte?"),
    ("estoy bien", "Me alegro de escuchar eso! ¿Cómo puedo asistirte?"),
    ("muy bien, gracias", "¡Genial! ¿Necesitas ayuda con algo?"),
    ("¿quién eres?", "Soy genv0-3, ¿y tú?"),
    ("quién eres?", "Soy genv0-3, ¿y tú?"),
    ("quien eres?", "Soy genv0-3, ¿y tú?"),
    ("¿cómo te llamas?", "Soy genv0-3, ¿y tú?"),
    ("cómo te llamas?", "Soy genv0-3, ¿y tú?"),
    ("como te llamas?", "Soy genv0-3, ¿y tú?"),
    ("¿qué eres?", "Soy un asistente virtual. ¿En qué puedo ayudarte?"),
    ("que eres?", "Soy un asistente virtual. ¿En qué puedo ayudarte?"),
    ("qué eres?", "Soy un asistente virtual. ¿En qué puedo ayudarte?"),
    ("¿qué puedes hacer?", "Puedo ayudarte con información y responder tus preguntas. ¿En qué necesitas ayuda?"),
    ("¿que puedes hacer?", "Puedo ayudarte con información y responder tus preguntas. ¿En qué necesitas ayuda?"),
    ("qué puedes hacer?", "Puedo ayudarte con información y responder tus preguntas. ¿En qué necesitas ayuda?"),
    ("qué puedes hacer?", "Puedo ayudarte con información y responder tus preguntas. ¿En qué necesitas ayuda?"),
    ("adios","que tengas un buen día"),
    ("adios","espero que tengas un buen día"),
]

# Preparar dataset
input_texts = []
target_texts = []

for input_text, target_text in conversations:
    input_texts.append(input_text)
    target_texts.append(target_text)

# Tokenización
filters = '#&();<>@[\\]^_`{|}~\t\n'
tokenizer = Tokenizer(filters=filters, lower=True)
tokenizer.fit_on_texts(input_texts + target_texts)
vocab_size = len(tokenizer.word_index) + 1

input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

max_input_len = max([len(seq) for seq in input_sequences])
max_target_len = max([len(seq) for seq in target_sequences])
max_len = max(max_input_len, max_target_len)

input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_len, padding='post')

def generarModelo(vocab_size, max_len):
    # Definir entradas
    input_layer = Input(shape=(max_len,), name='input_layer')
    
    # Embedding
    embedding_layer = Embedding(vocab_size, 2000, input_length=max_len)(input_layer)
    
    # Bidirectional LSTM
    lstm_layer_1 = Bidirectional(LSTM(1024, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01)))(embedding_layer)
    batch_norm_1 = BatchNormalization()(lstm_layer_1)
    dropout_1 = Dropout(0.3)(batch_norm_1)
    
    lstm_layer_2 = Bidirectional(LSTM(512, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01)))(dropout_1)
    batch_norm_2 = BatchNormalization()(lstm_layer_2)
    dropout_2 = Dropout(0.3)(batch_norm_2)
    
    # MultiHeadAttention
    attention_output = MultiHeadAttention(num_heads=8, key_dim=512)(dropout_2, dropout_2)
    
    # Normalización
    attention_output = LayerNormalization(epsilon=1e-7)(attention_output)
    
    # Dropout
    attention_output = Dropout(0.3)(attention_output)
    
    # TimeDistributed Dense
    output_layer = TimeDistributed(Dense(vocab_size, activation='softmax'))(attention_output)
    
    # Definir el modelo
    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.4, patience=8, min_lr=0.00001)
    
    # Entrenar el modelo
    history = model.fit(input_sequences, target_sequences, epochs=600, verbose=1, batch_size=32, callbacks=[early_stopping, reduce_lr])

    return model, history

def generate_response(input_text, model, tokenizer, max_input_len, beam_width=3, temperature=1.0, diversity=0.5):
    input_seq = tokenizer.texts_to_sequences([input_text])
    padded_input = pad_sequences(input_seq, maxlen=max_input_len, padding='post')
    predicted_output = model.predict(padded_input)[0]

    # Inicializar el beam
    beams = [(list(), 0.0)]  # Inicializar un solo beam vacío con probabilidad 0.0
    for _ in range(max_input_len):
        new_beams = []
        for beam in beams:
            word_probs = predicted_output[len(beam[0])]
            sampled_word_probs = np.log(word_probs + 1e-8) / temperature
            sampled_word_probs = np.exp(sampled_word_probs) / np.sum(np.exp(sampled_word_probs))
            sampled_word_indices = np.random.choice(len(word_probs), size=beam_width, replace=False, p=sampled_word_probs)
            for word_index in sampled_word_indices:
                new_beam = (beam[0].copy() + [word_index], beam[1] + np.log(word_probs[word_index] + 1e-8))
                new_beams.append(new_beam)
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]

    # Escoger el mejor camino y convertirlo a texto
    best_beam = beams[0][0]
    response_words = tokenizer.sequences_to_texts([best_beam])[0].split(' ')

    # Aplicar diversidad de muestreo
    for i in range(len(response_words)):
        if i < len(predicted_output) and np.random.rand() < diversity:
            sampled_word_probs = np.log(predicted_output[i] + 1e-8) / temperature
            sampled_word_probs = np.exp(sampled_word_probs) / np.sum(np.exp(sampled_word_probs))
            sampled_word_index = np.random.choice(len(sampled_word_probs), p=sampled_word_probs)
            response_words[i] = tokenizer.index_word.get(sampled_word_index, response_words[i])

    # Unir las palabras en una oración
    return ' '.join(response_words)

def generate_response_with_context(conversation_history, model, tokenizer, max_input_len, max_context_len=3, beam_width=3, temperature=1.0, diversity=0.5):
    # Limitar la longitud del historial de conversación
    if len(conversation_history) > max_context_len:
        conversation_history = conversation_history[-max_context_len:]
    
    # Concatenar el historial de conversación
    context_input = ' '.join(conversation_history)
    input_seq = tokenizer.texts_to_sequences([context_input])
    padded_input = pad_sequences(input_seq, maxlen=max_input_len, padding='post')
    predicted_output = model.predict(padded_input)[0]

    # Inicializar el beam
    beams = [(list(), 0.0)]
    for _ in range(max_input_len):
        new_beams = []
        for beam in beams:
            word_probs = predicted_output[len(beam[0])]
            sampled_word_probs = np.log(word_probs + 1e-8) / temperature
            sampled_word_probs = np.exp(sampled_word_probs) / np.sum(np.exp(sampled_word_probs))
            sampled_word_indices = np.random.choice(len(word_probs), size=beam_width, replace=False, p=sampled_word_probs)
            for word_index in sampled_word_indices:
                new_beam = (beam[0].copy() + [word_index], beam[1] + np.log(word_probs[word_index] + 1e-8))
                new_beams.append(new_beam)
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]

    best_beam = beams[0][0]
    response_words = tokenizer.sequences_to_texts([best_beam])[0].split(' ')

    for i in range(len(response_words)):
        if i < len(predicted_output) and np.random.rand() < diversity:
            sampled_word_probs = np.log(predicted_output[i] + 1e-8) / temperature
            sampled_word_probs = np.exp(sampled_word_probs) / np.sum(np.exp(sampled_word_probs))
            sampled_word_index = np.random.choice(len(sampled_word_probs), p=sampled_word_probs)
            response_words[i] = tokenizer.index_word.get(sampled_word_index, response_words[i])

    return ' '.join(response_words)

def plot_training_history(history):
    # Graficar la pérdida
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Pérdida durante el entrenamiento')
    plt.legend()

    # Graficar la precisión
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.title('Precisión durante el entrenamiento')
    plt.legend()

    plt.show()

# Manejador de conversación
conversation_history = deque(maxlen=3)
user_input = input("Escribe tu entrada: ")

if user_input != "cargar":
    model, history = generarModelo(vocab_size, max_len)
    # Graficar la historia del entrenamiento
    plot_training_history(history)
else:
    # Load the model
    model = load_model('my_model3.keras')
    history = model.fit(input_sequences, target_sequences, epochs=600, verbose=1, batch_size=32)
    plot_training_history(history)

while user_input != "adios":
    user_input = input("Usuario: ")
    conversation_history.append(user_input)
    #response = generate_response_with_context(conversation_history, model, tokenizer, max_len)
    response = generate_response(user_input, model, tokenizer, max_len)
    print(f"genv0-3: {response}")
    conversation_history.append(response)

model.save('my_model3.keras')