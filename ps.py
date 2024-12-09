import numpy as np
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# Funções auxiliares
activation_func = lambda x: np.maximum(0, x)  # ReLU
activation_derivative = lambda a: np.where(a > 0, 1, 0)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def step_activation(u):
    return np.where(u >= 0, 1, 0)

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
    for true_label, predicted_label in zip(y_test, predictions):
        conf_matrix[true_label, predicted_label] += 1
    return conf_matrix


def calculate_accuracy(predictions, y_test):
    """
    Calcula a acurácia entre as predições e os rótulos reais.

    :param predictions: Predições do modelo, no formato one-hot ou índices
    :param y_test: Rótulos reais, no formato de índices
    :return: Acurácia
    """
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)  # Converte one-hot para índices
    return np.mean(predictions == y_test)


# Função para calcular métricas
def calculate_metrics(predictions, y_test, n_classes):
    conf_matrix = calculate_confusion_matrix(predictions, y_test, n_classes)
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

# Classe Perceptron Simples
class PerceptronSimples:
    def __init__(self, learning_rate=0.01, max_epochs=20, n_classes=20):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.n_classes = n_classes
        self.weights = None

    def fit(self, X, y):
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        self.weights = np.random.randn(X_bias.shape[1], self.n_classes) * 0.01

        for epoch in range(self.max_epochs):
            total_error = 0
            for i in range(X.shape[0]):
                xi = X_bias[i]
                yi = y[i]
                u = np.dot(xi, self.weights)
                y_pred = step_activation(u)
                error = yi - y_pred
                total_error += np.sum(np.abs(error))
                self.weights += self.learning_rate * np.outer(xi, error)

            if total_error / X.shape[0] < 1e-3:
                break

    def predict(self, X):
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        u = np.dot(X_bias, self.weights)
        return np.argmax(u, axis=1)

# Classe Adaline
class Adaline:
    def __init__(self, learning_rate=0.001, max_epochs=1000, epsilon=1e-3):  
        """
        Construtor da classe Adaline.

        :param learning_rate: Taxa de aprendizado
        :param max_epochs: Número máximo de épocas para treinamento
        :param epsilon: Critério de parada baseado no erro quadrático médio (EQM)
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.epsilon = epsilon  # Agora epsilon é um parâmetro configurável
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Treina o modelo Adaline.

        :param X: Dados de entrada (features)
        :param y: Rótulos de saída (target)
        :return: Histórico do erro quadrático médio (EQM) para cada época
        """
        N, M = X.shape  # N = número de amostras, M = número de características
        self.num_classes = y.shape[1]  # Número de classes
        self.weights = np.random.randn(M, self.num_classes) * 0.001  # Inicializa pesos
        self.bias = np.zeros(self.num_classes)  # Inicializa o bias
        eqm_history = []  # Para armazenar o EQM de cada época

        for epoch in range(self.max_epochs):
            # Cálculo do potencial u e previsão
            u = np.dot(X, self.weights) + self.bias
            u = np.clip(u, -1e3, 1e3)  # Evita overflow
            y_hat = u  # Ativação linear

            # Erro e EQM
            error = y - y_hat  # Erro entre o valor real e a previsão
            error = np.clip(error, -1e3, 1e3)  # Limitar erro para evitar overflow
            eqm = np.mean(error**2)  # Calcula o erro quadrático médio
            eqm_history.append(eqm)  # Armazena o EQM para essa época

            # Atualiza pesos e bias
            self.weights += self.learning_rate * np.dot(X.T, error) / N
            self.bias += self.learning_rate * np.mean(error, axis=0)

              # Critério de parada baseado no erro (se o EQM for menor que epsilon, o treinamento é interrompido)
            if eqm < self.epsilon:
                print(f"Convergência alcançada na época {epoch + 1}")
                break

        return eqm_history  # Retorna o histórico do EQM

    def predict(self, X):
        """
        Faz previsões para os dados de entrada.

        :param X: Dados de entrada para prever as classes
        :return: Rótulos previstos para os dados de entrada
        """
        u = np.dot(X, self.weights) + self.bias  # Calcula a previsão
        return np.argmax(u, axis=1)  # Retorna o índice da classe com maior valor


# Classe MLP
class MLP:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01, max_epochs=1000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = [np.random.randn(self.layers[i + 1], self.layers[i] + 1) for i in range(len(self.layers) - 1)]

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

# Execução
# Execução
if __name__ == "__main__":
    import matplotlib.pyplot as plt

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

    # Dicionário para armazenar as acurácias de cada modelo
    estatisticas_acuracia = {}

    for nome_modelo, modelo in modelos.items():
        print(f"\nExecutando {nome_modelo}...")

        acuracias = []  # Lista para armazenar as acurácias de múltiplas execuções

        # Repetir o experimento para obter estatísticas
        for repeticao in range(5):  # 5 execuções para cálculo das estatísticas
            indices = np.random.permutation(n_samples)
            X_train, X_test = data[indices[:512]], data[indices[512:]]
            y_train, y_test = labels[indices[:512]], labels[indices[512:]]

            # Ajuste para capturar histórico de EQM se o modelo for Adaline
            if nome_modelo == "Adaline":
                modelo.fit(X_train, y_train)
            else:
                modelo.fit(X_train, y_train)

            predictions = modelo.predict(X_test)

            # Certifique-se de que predictions seja correto
            if predictions.ndim == 1:
                predictions = np.eye(n_classes)[predictions]

            # Calcula a acurácia
            # Certifique-se de que y_test e predictions estão em formato de índices
            accuracy = calculate_accuracy(predictions, np.argmax(y_test, axis=1))


            acuracias.append(accuracy)

        # Cálculo das estatísticas
        media_acuracia = np.mean(acuracias)
        dp_acuracia = np.std(acuracias)
        max_acuracia = np.max(acuracias)
        min_acuracia = np.min(acuracias)

        # Armazenar estatísticas
        estatisticas_acuracia[nome_modelo] = {
            "Média": media_acuracia,
            "Desvio Padrão": dp_acuracia,
            "Máximo": max_acuracia,
            "Mínimo": min_acuracia
        }

        # Exibir estatísticas
        print(f"\nEstatísticas de Acurácia para {nome_modelo}:")
        print(f" - Média: {media_acuracia:.4f}")
        print(f" - Desvio Padrão: {dp_acuracia:.4f}")
        print(f" - Máximo: {max_acuracia:.4f}")
        print(f" - Mínimo: {min_acuracia:.4f}")
