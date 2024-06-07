import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt


def train_and_evaluate_model(model, x_train, y_train, x_test, y_test, epochs=10, batch_size=128):
    """Entrena el modelo y evalúa su precisión."""
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {accuracy*100:.2f}%')
    return model

def make_predictions(model, x_test):
    """Hace predicciones usando el modelo entrenado."""
    return model.predict(x_test)

def plot_image(predictions_array, true_label, img):
    """Muestra una imagen y su etiqueta predicha y verdadera."""
    predictions_array, true_label, img = predictions_array, np.argmax(true_label), img.reshape(28, 28)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'
    plt.xlabel(f"{predicted_label} ({true_label})", color=color)

# Cargar y preprocesar el conjunto de datos MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Redimensionar datos para que tengan un solo canal (blanco y negro)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Normalizar los valores de los píxeles a un rango de 0 a 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Aplanar los datos de entrada
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

# Convertir etiquetas a formato categórico
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Crear el modelo
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))

# Compilar el modelo
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Entrenar y evaluar el modelo
model = train_and_evaluate_model(model, x_train, y_train, x_test, y_test)

# Hacer predicciones
predictions = make_predictions(model, x_test)

# Mostrar la primera imagen del conjunto de prueba y su predicción
plot_image(predictions[0], y_test[0], x_test[0])
print(predictions[0], y_test[0], x_test[0])
plt.show()