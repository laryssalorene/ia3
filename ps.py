import numpy as np
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Funções auxiliares
def activation_func(x):
    """Função de ativação ReLU."""
    return np.maximum(0, x)

def activation_derivative(a):
    """Derivada da função ReLU."""
    return np.where(a > 0, 1, 0)

def softmax(z):
    """Função Softmax."""
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def step_activation(u):
    """Função de ativação degrau."""
    return np.where(u >= 0, 1, -1)  # Retorna 1 para u >= 0 e -1 para u < 0

# Classe para leitura de imagens
class LerImagem:
    @staticmethod
    def obter_dados(img_size=50):
        raiz = "RecFac"
        pastas_pessoas = [os.path.join(raiz, pasta) for pasta in os.listdir(raiz) if os.path.isdir(os.path.join(raiz, pasta))]
        C = 20  # Total de classes
        X, Y = [], []

        for i, pasta in enumerate(pastas_pessoas):
            imagens_pasta = os.listdir(pasta)
            for imagem_nome in imagens_pasta:
                path_imagem = os.path.join(pasta, imagem_nome)
                imagem = cv2.imread(path_imagem, cv2.IMREAD_GRAYSCALE)
                imagem_redimensionada = cv2.resize(imagem, (img_size, img_size))
                imagem_vetorizada = imagem_redimensionada.flatten()
                X.append(imagem_vetorizada)
                rotulo = np.zeros(C)
                rotulo[i] = 1
                Y.append(rotulo)

        X = np.array(X)
        Y = np.array(Y)
        print(f"Tamanho de X: {X.shape}, Tamanho de Y: {Y.shape}")
        return X, Y

# Classe para normalização dos dados
class Normalizacao:
    @staticmethod
    def normalizar(data):
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        range_val = max_val - min_val
        normalized_data = (data - min_val) / range_val
        normalized_data = normalized_data * 2 - 1
        return normalized_data

# Função para calcular matriz de confusão
def calculate_confusion_matrix(predictions, y_test, n_classes):
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, predicted_label in zip(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1)):
        conf_matrix[true_label, predicted_label] += 1
    return conf_matrix

def calculate_accuracy(predictions, y_test):
    """Calcula a acurácia entre as predições e os rótulos reais."""
    predictions_one_hot = predictions if predictions.ndim > 1 else np.eye(y_test.shape[1])[predictions]
    return np.mean(np.argmax(predictions_one_hot, axis=1) == np.argmax(y_test, axis=1))

# Função para calcular métricas
def calculate_metrics(predictions, y_test, n_classes):
    predictions_one_hot = predictions if predictions.ndim > 1 else np.eye(y_test.shape[1])[predictions]
    conf_matrix = calculate_confusion_matrix(predictions_one_hot, y_test, n_classes)
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
    sensitivity = []
    specificity = []

    for i in range(n_classes):
        tp = conf_matrix[i, i]
        fn = np.sum(conf_matrix[i, :]) - tp
        fp = np.sum(conf_matrix[:, i]) - tp
        tn = np.sum(conf_matrix) - (tp + fn + fp)
        sensitivity.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    avg_sensitivity = np.mean(sensitivity)
    avg_specificity = np.mean(specificity)

    return accuracy, avg_sensitivity, avg_specificity, conf_matrix

# Perceptron Simples
class PerceptronSimples:
    def __init__(self, learning_rate=0.01, max_epochs=50, n_classes=20):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.n_classes = n_classes
        self.weights = None

    def fit(self, X, y):
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])  # Adiciona bias
        self.weights = np.zeros((X_bias.shape[1], self.n_classes))  # Inicializa pesos com 0

        for epoch in range(self.max_epochs):
            erro_existe = False

            for i in range(X.shape[0]):
                xi = X_bias[i]
                yi = y[i]

                u = np.dot(xi, self.weights)
                y_pred = step_activation(u)

                for j in range(self.n_classes):
                    if y_pred[j] != yi[j]:
                        erro_existe = True
                        self.weights[:, j] += self.learning_rate * (yi[j] - y_pred[j]) * xi

            if not erro_existe:
                print(f"Convergência alcançada na época {epoch + 1}")
                break

    def predict(self, X):
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        u = np.dot(X_bias, self.weights)
        return step_activation(u)

# Adaline
class Adaline:
    def __init__(self, learning_rate=0.001, max_epochs=1000, epsilon=1e-3):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.epsilon = epsilon
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        N, M = X.shape
        self.num_classes = y.shape[1]
        self.weights = np.random.randn(M, self.num_classes) * 0.001
        self.bias = np.zeros(self.num_classes)
        eqm_history = []

        for epoch in range(self.max_epochs):
            u = np.dot(X, self.weights) + self.bias
            u = np.clip(u, -1e3, 1e3)
            y_hat = u

            error = y - y_hat
            error = np.clip(error, -1e3, 1e3)
            eqm = np.mean(error**2)
            eqm_history.append(eqm)

            self.weights += self.learning_rate * np.dot(X.T, error) / N
            self.bias += self.learning_rate * np.mean(error, axis=0)

            if eqm < self.epsilon:
                print(f"Convergência alcançada na época {epoch + 1}")
                break

        return eqm_history

    def predict(self, X):
        u = np.dot(X, self.weights) + self.bias
        return u

# MLP
class MLP:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01, max_epochs=1000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = [np.random.randn(self.layers[i + 1], self.layers[i] + 1) for i in range(len(self.layers) - 1)]

    def forward(self, X):
        activations = [X.T]
        for w in self.weights[:-1]:
            a_with_bias = np.vstack([np.ones((1, activations[-1].shape[1])), activations[-1]] )
            z = np.dot(w, a_with_bias)
            activations.append(activation_func(z))
        a_with_bias = np.vstack([np.ones((1, activations[-1].shape[1])), activations[-1]] )
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
                a_with_bias = np.vstack([np.ones((1, activations[l].shape[1])), activations[l]] )
                self.weights[l] -= self.learning_rate * np.dot(deltas[l], a_with_bias.T) / X.shape[0]

    def predict(self, X):
        activations = self.forward(X)
        return activations[-1].T

# Função para exibir a matriz de confusão usando seaborn
def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Execução principal
if __name__ == "__main__":
    img_size = 50
    data, labels = LerImagem.obter_dados(img_size)
    data = Normalizacao.normalizar(data)

    n_samples, n_features = data.shape
    n_classes = labels.shape[1]

    modelos = {
        "Perceptron Simples": PerceptronSimples(learning_rate=0.01, max_epochs=10, n_classes=n_classes),
        "Adaline": Adaline(learning_rate=0.01, max_epochs=1000, epsilon=1e-3),
        "MLP": MLP(input_size=n_features, hidden_layers=[128], output_size=n_classes, learning_rate=0.01, max_epochs=10)
    }

    estatisticas_acuracia = {}

    for nome_modelo, modelo in modelos.items():
        print(f"\nExecutando {nome_modelo}...")

        acuracias = []
        conf_matrices = []

        for repeticao in range(5):
            indices = np.random.permutation(n_samples)
            X_train, X_test = data[indices[:512]], data[indices[512:]]
            y_train, y_test = labels[indices[:512]], labels[indices[512:]]

            modelo.fit(X_train, y_train)
            predictions = modelo.predict(X_test)

            y_test_one_hot = y_test if y_test.ndim > 1 else np.eye(n_classes)[y_test]
            predictions_one_hot = predictions if predictions.ndim > 1 else np.eye(n_classes)[np.argmax(predictions, axis=1)]

            conf_matrix = calculate_confusion_matrix(predictions_one_hot, y_test_one_hot, n_classes)
            conf_matrices.append(conf_matrix)
            accuracy = calculate_accuracy(predictions_one_hot, y_test_one_hot)
            acuracias.append(accuracy)

        media_acuracia = np.mean(acuracias)
        dp_acuracia = np.std(acuracias)
        max_acuracia = np.max(acuracias)
        min_acuracia = np.min(acuracias)

        estatisticas_acuracia[nome_modelo] = {
            "Média": media_acuracia,
            "Desvio Padrão": dp_acuracia,
            "Máximo": max_acuracia,
            "Mínimo": min_acuracia
        }

        print(f"\nEstatísticas de Acurácia para {nome_modelo}:")
        print(f" - Média: {media_acuracia:.4f}")
        print(f" - Desvio Padrão: {dp_acuracia:.4f}")
        print(f" - Máximo: {max_acuracia:.4f}")
        print(f" - Mínimo: {min_acuracia:.4f}")

        max_conf_matrix = conf_matrices[acuracias.index(max_acuracia)]
        min_conf_matrix = conf_matrices[acuracias.index(min_acuracia)]

        print(f"\nMatriz de confusão para maior acurácia ({max_acuracia:.4f}):")
        plot_confusion_matrix(max_conf_matrix, f"{nome_modelo} - Maior Acurácia")

        print(f"\nMatriz de confusão para menor acurácia ({min_acuracia:.4f}):")
        plot_confusion_matrix(min_conf_matrix, f"{nome_modelo} - Menor Acurácia")

    print("\nEstatísticas de Acurácia para todos os Modelos:")
    for nome_modelo, stats in estatisticas_acuracia.items():
        print(f"\n{nome_modelo}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}")
