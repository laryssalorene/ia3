import numpy as np
import os
import cv2

# Funções auxiliares
activation_func = lambda x: np.maximum(0, x)  # ReLU
activation_derivative = lambda a: np.where(a > 0, 1, 0)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def step_activation(u):
    return np.where(u >= 0, 1, 0)

# Classe Perceptron Simples
class PerceptronSimples:
    def __init__(self, learning_rate=0.01, max_epochs=10, tolerance=1e-4, n_classes=20):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.n_classes = n_classes
        self.weights = None

    def fit(self, X, y):
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        self.weights = np.random.randn(X_bias.shape[1], self.n_classes) * 0.01
        for epoch in range(self.max_epochs):
            for i in range(X.shape[0]):
                xi = X_bias[i]
                yi = y[i]
                u = np.dot(xi, self.weights)
                y_pred = step_activation(u)
                error = yi - y_pred
                self.weights += self.learning_rate * np.outer(xi, error)

    def predict(self, X):
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        u = np.dot(X_bias, self.weights)
        return step_activation(u)

# Classe Adaline
class Adaline:
    def __init__(self, learning_rate=0.01, max_epochs=1000, epsilon=1e-3):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.epsilon = epsilon
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        N, M = X.shape
        self.num_classes = y.shape[1]
        self.weights = np.random.randn(M, self.num_classes) * np.sqrt(2 / M)
        self.bias = np.zeros(self.num_classes)
        for epoch in range(self.max_epochs):
            u = np.dot(X, self.weights) + self.bias
            y_hat = np.where(u >= 0, 1, 0)
            error = y - y_hat
            for i in range(self.num_classes):
                self.weights[:, i] += self.learning_rate * np.dot(X.T, error[:, i]) / N
                self.bias[i] += self.learning_rate * np.sum(error[:, i]) / N
            if np.linalg.norm(error) < self.epsilon:
                break

    def predict(self, X):
        u = np.dot(X, self.weights) + self.bias
        return np.where(u >= 0, 1, 0)

# Classe MLP
class MLP:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01, max_epochs=1000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = [
            np.random.randn(self.layers[i + 1], self.layers[i] + 1) * np.sqrt(2 / self.layers[i])
            for i in range(len(self.layers) - 1)
        ]

    def forward(self, X):
        activations = [X.T]
        for w in self.weights[:-1]:
            a_with_bias = np.vstack([np.ones((1, activations[-1].shape[1])), activations[-1]])
            z = np.dot(w, a_with_bias)
            activations.append(activation_func(z))
        a_with_bias = np.vstack([np.ones((1, activations[-1].shape[1])), activations[-1]])
        z = np.dot(self.weights[-1], a_with_bias)
        activations.append(softmax(z))
        return activations

    def fit(self, X, y):
        for epoch in range(self.max_epochs):
            activations = self.forward(X)
            y_pred = activations[-1]
            deltas = [y_pred - y.T]
            for l in range(len(self.weights) - 2, -1, -1):
                deltas.insert(0, np.dot(self.weights[l + 1][:, 1:].T, deltas[0]) * activation_derivative(activations[l + 1]))
            for l in range(len(self.weights)):
                a_with_bias = np.vstack([np.ones((1, activations[l].shape[1])), activations[l]])
                self.weights[l] -= self.learning_rate * np.dot(deltas[l], a_with_bias.T) / X.shape[0]

    def predict(self, X):
        activations = self.forward(X)
        y_pred = activations[-1].T
        return np.argmax(y_pred, axis=1)

# Funções de Métrica
def calculate_metrics(y_test, predictions):
    accuracy = np.mean(np.argmax(y_test, axis=1) == np.argmax(predictions, axis=1))
    sensitivity = np.mean(np.diag(np.dot(y_test.T, predictions)) / np.sum(y_test, axis=0))
    specificity = np.mean(np.diag(np.dot(1 - y_test.T, 1 - predictions)) / np.sum(1 - y_test, axis=0))
    return accuracy, sensitivity, specificity

# Execução
if __name__ == "__main__":
    # Dados simulados para demonstração
    n_samples = 640
    n_features = 2500
    n_classes = 20
    X = np.random.rand(n_samples, n_features)
    y = np.eye(n_classes)[np.random.choice(n_classes, n_samples)]

    modelos = {
        "Perceptron Simples": PerceptronSimples(learning_rate=0.01, max_epochs=10, n_classes=n_classes),
        "Adaline": Adaline(learning_rate=0.01, max_epochs=1000),
        "MLP": MLP(input_size=n_features, hidden_layers=[128], output_size=n_classes, learning_rate=0.01, max_epochs=1000)
    }

    for nome_modelo, modelo in modelos.items():
        metrics = {"accuracy": [], "sensitivity": [], "specificity": []}
        print(f"\nExecutando {nome_modelo}...")

        for _ in range(10):
            indices = np.random.permutation(n_samples)
            X_train, X_test = X[indices[:512]], X[indices[512:]]
            y_train, y_test = y[indices[:512]], y[indices[512:]]

            modelo.fit(X_train, y_train)
            predictions = modelo.predict(X_test)

            if predictions.ndim == 1:
                predictions = np.eye(n_classes)[predictions]

            accuracy, sensitivity, specificity = calculate_metrics(y_test, predictions)
            metrics["accuracy"].append(accuracy)
            metrics["sensitivity"].append(sensitivity)
            metrics["specificity"].append(specificity)

        for metric_name, values in metrics.items():
            mean_value = np.mean(values)
            std_value = np.std(values)
            max_value = np.max(values)
            min_value = np.min(values)

            print(f"\nEstatísticas para {metric_name} - {nome_modelo}:")
            print(f"Média: {mean_value:.4f}")
            print(f"Desvio-Padrão: {std_value:.4f}")
            print(f"Maior Valor: {max_value:.4f}")
            print(f"Menor Valor: {min_value:.4f}")
