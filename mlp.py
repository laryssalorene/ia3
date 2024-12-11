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

    return accuracy, avg_sensitivity, avg_specificity, sensitivity, specificity, conf_matrix

# MLP
class MLP:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01, max_epochs=1000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.layers = [input_size] + hidden_layers + [output_size]
        
        # Inicialização dos pesos (usando a inicialização de He para ReLU)
        self.weights = [np.random.randn(self.layers[i + 1], self.layers[i] + 1) * np.sqrt(2. / self.layers[i])
                        for i in range(len(self.layers) - 1)]  # He initialization

    def forward(self, X):
        activations = [X.T]
        for w in self.weights[:-1]:
            a_with_bias = np.vstack([np.ones((1, activations[-1].shape[1])), activations[-1]])  # Adiciona o viés
            z = np.dot(w, a_with_bias)
            activations.append(activation_func(z))  # Aplica ReLU

        # Camada de saída com softmax
        a_with_bias = np.vstack([np.ones((1, activations[-1].shape[1])), activations[-1]])  # Adiciona o viés
        z = np.dot(self.weights[-1], a_with_bias)
        activations.append(softmax(z))  # Aplica softmax na saída
        return activations

    def backward(self, X, y, activations):
        deltas = []
        y_pred = activations[-1]  # Saída da rede (última camada)

        # Calculando erro da camada de saída usando Cross-Entropy com Softmax
        delta_output = y_pred - y.T
        deltas.append(delta_output)

        # Backpropagation para camadas ocultas
        for l in range(len(self.weights) - 2, -1, -1):  # Itera de trás para frente (camadas ocultas)
            delta_hidden = np.dot(self.weights[l + 1][:, 1:].T, deltas[0]) * activation_derivative(activations[l + 1])
            deltas.insert(0, delta_hidden)

        return deltas

    def fit(self, X, y):
        for epoch in range(self.max_epochs):
            activations = self.forward(X)  # Propagação para frente

            # Backpropagation para calcular gradientes
            deltas = self.backward(X, y, activations)

            # Atualização dos pesos usando o gradiente
            for l in range(len(self.weights)):  # Para cada camada, atualiza os pesos
                a_with_bias = np.vstack([np.ones((1, activations[l].shape[1])), activations[l]])  # Adiciona o viés
                self.weights[l] -= self.learning_rate * np.dot(deltas[l], a_with_bias.T) / X.shape[0]

    def predict(self, X):
        activations = self.forward(X)
        return activations[-1].T  # Retorna a predição da camada de saída


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
        "MLP": MLP(input_size=n_features, hidden_layers=[128], output_size=n_classes, learning_rate=0.01, max_epochs=10)
    }

    estatisticas_acuracia = {}

    for nome_modelo, modelo in modelos.items():
        print(f"\nExecutando {nome_modelo}...")

        acuracias = []
        sensibilidade = []
        especificidade = []
        conf_matrices = []

        # Rodadas de Monte Carlo
        for repeticao in range(50):  # Agora rodando por 50 vezes
            indices = np.random.permutation(n_samples)
            X_train, X_test = data[indices[:int(0.8 * n_samples)]], data[indices[int(0.8 * n_samples):]]
            y_train, y_test = labels[indices[:int(0.8 * n_samples)]], labels[indices[int(0.8 * n_samples):]]

            modelo.fit(X_train, y_train)
            predictions = modelo.predict(X_test)

            accuracy, avg_sens, avg_spec, sens, spec, conf_matrix = calculate_metrics(predictions, y_test, n_classes)
            acuracias.append(accuracy)
            sensibilidade.append(avg_sens)
            especificidade.append(avg_spec)
            conf_matrices.append(conf_matrix)

        # Estatísticas das métricas
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

        # Encontre as matrizes de confusão para as maiores e menores acurácias
        max_conf_matrix = conf_matrices[acuracias.index(max_acuracia)]
        min_conf_matrix = conf_matrices[acuracias.index(min_acuracia)]

        print(f"\nEstatísticas de Acurácia para {nome_modelo}:")
        print(f" - Média: {media_acuracia:.4f}")
        print(f" - Desvio Padrão: {dp_acuracia:.4f}")
        print(f" - Máximo: {max_acuracia:.4f}")
        print(f" - Mínimo: {min_acuracia:.4f}")

        print(f"\nMatriz de confusão para maior acurácia ({max_acuracia:.4f}):")
        plot_confusion_matrix(max_conf_matrix, f"{nome_modelo} - Maior Acurácia")

        print(f"\nMatriz de confusão para menor acurácia ({min_acuracia:.4f}):")
        plot_confusion_matrix(min_conf_matrix, f"{nome_modelo} - Menor Acurácia")

    print("\nEstatísticas de Acurácia")
    for nome_modelo, stats in estatisticas_acuracia.items():
        print(f"\n{nome_modelo}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}")
