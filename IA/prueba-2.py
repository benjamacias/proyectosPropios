import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np


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
print(enumerate(predictions))
# Recorrer todas las predicciones
for i, prediction in enumerate(predictions):
    # La predicción es un vector de 10 elementos debido a la capa softmax
    # El índice del elemento más grande es la clase predicha
    predicted_number = np.argmax(prediction)
    print(f"Predicción para la imagen {i}: {predicted_number}")

# Mostrar la primera imagen del conjunto de prueba y su predicción
#plot_image(predictions[1], y_test[1], x_test[1])
#print(f"Predicción: {np.argmax(predictions[1])}")
#plt.show()


# Seleccionar el primer número en el conjunto de prueba
#print(x_test)

"""
# Cargar la imagen
img_path = 'ruta/a/tu/imagen.png'  # Reemplaza esto con la ruta a tu imagen
img = image.load_img(img_path, target_size=(28, 28), color_mode = "grayscale")

# Convertir la imagen a un array de numpy y aplanarla
img_array = image.img_to_array(img)
img_array = img_array.reshape((1, -1))

# Normalizar los datos de la imagen
img_array = img_array.astype('float32') / 255

# Hacer la predicción
prediction = model.predict(img_array)

# La predicción es un vector de 10 elementos debido a la capa softmax
# El índice del elemento más grande es la clase predicha
predicted_number = np.argmax(prediction)

print(f"El número predicho es: {predicted_number}")
"""