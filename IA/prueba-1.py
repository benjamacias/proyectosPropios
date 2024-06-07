import numpy as np

def nonlin(x, deriv=False):
    """Función sigmoide como función de activación."""
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def train_neural_network(X, y, iterations=10000, seed=1):
    """Entrena la red neuronal."""
    np.random.seed(seed)
    syn0 = 2 * np.random.random((X.shape[1], 1)) - 1

    for _ in range(iterations):
        l0 = X
        l1 = nonlin(np.dot(l0, syn0))
        l1_error = y - l1
        l1_delta = l1_error * nonlin(l1, True)
        syn0 += np.dot(l0.T, l1_delta)

    return l1, syn0

X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0], [0], [1], [1]])

# Asegurarse de que X y y tienen las formas correctas
assert X.shape[0] == y.shape[0], "X y y deben tener la misma cantidad de filas."

output, weights = train_neural_network(X, y)

print("Salida:")
print(output)
print("Pesos:")
print(weights)