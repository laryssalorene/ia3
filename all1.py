import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2



import os
import numpy as np
import cv2

class LerImagem:
    
    @staticmethod
    def obter_dados(img_size=50):
        raiz = "RecFac"
        # Usando os.listdir e list comprehension para pegar todas as pastas
        pastas_pessoas = [os.path.join(raiz, pasta) for pasta in os.listdir(raiz) if os.path.isdir(os.path.join(raiz, pasta))]

        C = 20  # Total de classes (assumindo que há 20 pastas, uma por pessoa)
        X = []  # Lista para armazenar as imagens vetorizadas
        Y = []  # Lista para armazenar os rótulos (one-hot encoding)

        # Iterar pelas pastas (pessoas)
        for i, pasta in enumerate(pastas_pessoas):
            imagens_pasta = os.listdir(pasta)  # Lista de imagens dentro da pasta da pessoa

            for imagem_nome in imagens_pasta:
                path_imagem = os.path.join(pasta, imagem_nome)

                # Carrega a imagem em escala de cinza
                imagem = cv2.imread(path_imagem, cv2.IMREAD_GRAYSCALE)
                # Redimensiona a imagem
                imagem_redimensionada = cv2.resize(imagem, (img_size, img_size))

                # Vetorização da imagem (transforma em vetor 1D)
                imagem_vetorizada = imagem_redimensionada.flatten()

                # Armazenando a imagem vetorizada na lista X
                X.append(imagem_vetorizada)

                # Criando o vetor de rótulos one-hot encoding
                rotulo = np.zeros(C)  # Inicializa todos os valores com 0
                rotulo[i] = 1  # Marca a classe correta como 1
                Y.append(rotulo)

        # Convertendo as listas para arrays numpy
        X = np.array(X)  # Forma (n_amostras, n_features)
        Y = np.array(Y)  # Forma (n_amostras, n_classes)
        print(f"Tamanho final de X: {X.shape}, Tamanho final de Y: {Y.shape}")
        print(f"Exemplo de Y[0]: {Y[0]}")
        print(f"Soma de Y[0]: {np.sum(Y[0])}")  # Deve ser 1


        return X, Y

class Normalizacao:
    
    @staticmethod
    def normalizar(data):
        # 1. Calcular o valor mínimo e máximo para cada característica
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        
        # 2. Normalização para o intervalo [0, 1]
        range_val = max_val - min_val  # Calculando a diferença entre max e min
        normalized_data = (data - min_val) / range_val  # Normalizando para o intervalo [0, 1]
        
        # 3. Ajustando para o intervalo [-1, 1]
        normalized_data = normalized_data * 2 - 1  # Ajusta para o intervalo [-1, 1]
        
        return normalized_data
# Função para realizar a divisão manual dos dados (sem usar sklearn)
def split_data(X, y, test_size=0.2, seed=None):
    np.random.seed(seed)
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    test_size = int(n_samples * test_size)
    
    X_train = X[indices[test_size:]]
    y_train = y[indices[test_size:]]
    X_test = X[indices[:test_size]]
    y_test = y[indices[:test_size]]
    
    return X_train, X_test, y_train, y_test

# Função para calcular a acurácia
def calculate_accuracy(predictions, y_test):
    return np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))

# Função para calcular a matriz de confusão
def calculate_confusion_matrix(predictions, y_test, n_classes):
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, predicted_label in zip(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1)):
        conf_matrix[true_label, predicted_label] += 1
    return conf_matrix

# Função para realizar a validação de Monte Carlo
def monte_carlo_validation(model, X, y, n_runs=50):
    accuracies = []
    all_conf_matrices = []

    for run in range(n_runs):
        # Divisão manual dos dados
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, seed=run)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Calcula acurácia
        accuracy = calculate_accuracy(predictions, y_test)
        accuracies.append(accuracy)

        # Calcula matriz de confusão
        conf_matrix = calculate_confusion_matrix(predictions, y_test, n_classes)
        all_conf_matrices.append(conf_matrix)

    return accuracies, all_conf_matrices

# Função para plotar a curva de aprendizado
def plot_learning_curve(accuracies, model_name):
    plt.plot(accuracies, label=model_name)
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.title(f'Curva de Aprendizado - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Função para plotar a matriz de confusão usando seaborn
def plot_confusion_matrix(conf_matrix, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(conf_matrix.shape[1]), yticklabels=np.arange(conf_matrix.shape[0]))
    plt.title(f'Matriz de Confusão - {model_name}')
    plt.xlabel('Previsões')
    plt.ylabel('Real')
    plt.show()

# Perceptron Simples
class PerceptronSimples:
    def __init__(self, learning_rate=0.01, max_epochs=50, n_classes=20):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.n_classes = n_classes
        self.weights = None
        self.acuracias = []

    def fit(self, X, y):
        N, M = X.shape
        X_bias = np.hstack([np.ones((N, 1)), X])  # Adiciona bias
        self.weights = np.random.uniform(-0.5, 0.5, (M + 1, self.n_classes))  # Inicializa pesos aleatoriamente

        for epoch in range(self.max_epochs):
            erro = False

            for i in range(N):
                xi = X_bias[i].reshape(-1, 1)
                target = y[i]

                u = np.dot(self.weights.T, xi).flatten()
                y_pred = np.where(u >= 0, 1, -1)  # Função degrau bipolar

                for j in range(self.n_classes):
                    if y_pred[j] != target[j]:
                        erro = True
                        self.weights[:, j] += self.learning_rate * (target[j] - y_pred[j]) * xi.flatten()

            # Calcula a acurácia a cada época
            u = np.dot(X_bias, self.weights)
            predictions = np.argmax(u, axis=1)
            accuracy = np.mean(np.argmax(y, axis=1) == predictions)
            self.acuracias.append(accuracy)

            if not erro:
                print(f"Convergência alcançada na época {epoch + 1}")
                break

    def predict(self, X):
        N = X.shape[0]
        X_bias = np.hstack([np.ones((N, 1)), X])
        u = np.dot(X_bias, self.weights)
        return np.eye(self.n_classes)[np.argmax(u, axis=1)]


# Adaline
class Adaline:
    def __init__(self, learning_rate=0.01, max_epochs=1000, epsilon=1e-6, n_classes=20):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.epsilon = epsilon
        self.n_classes = n_classes
        self.acuracias = []

    def fit(self, X, Y):
        # Adiciona o viés (bias) e normaliza os dados
        X = np.hstack((-np.ones((X.shape[0], 1)), X))
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

        # Inicializa os pesos
        self.weights = np.random.uniform(-0.1, 0.1, (X.shape[1], Y.shape[1]))

        prev_eqm = np.inf  # Inicializa o erro quadrático médio da época anterior
        for epoch in range(self.max_epochs):
            # Cálculo das saídas
            outputs = X @ self.weights
            errors = Y - outputs

            # Verificação de estabilidade
            if not np.all(np.isfinite(errors)):
                print(f"Erro numérico detectado na época {epoch}. Finalizando.")
                break

            # Atualiza os pesos
            eqm = np.mean(errors**2) / 2
            grad = np.clip(X.T @ errors / X.shape[0], -1e3, 1e3)  # Limita o gradiente
            self.weights += self.learning_rate * grad

            # Calcula a acurácia a cada época
            accuracy = np.mean(np.argmax(Y, axis=1) == np.argmax(outputs, axis=1))
            self.acuracias.append(accuracy)

            # Critério de parada
            if epoch > 0 and abs(eqm - prev_eqm) < self.epsilon:
                print(f"Convergência atingida na época {epoch}.")
                break

            prev_eqm = eqm  # Atualiza o erro quadrático médio

    def predict(self, X):
        X = np.hstack((-np.ones((X.shape[0], 1)), X))
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
        outputs = X @ self.weights
        return np.eye(outputs.shape[1])[np.argmax(outputs, axis=1)]


# Função para execução principal
if __name__ == "__main__":
    # Carregar e normalizar os dados fixos
    img_size = 20
    data, labels = LerImagem.obter_dados(img_size)
    data = Normalizacao.normalizar(data)

    n_samples, n_features = data.shape
    n_classes = labels.shape[1]

    modelos = {
        "Perceptron Simples": PerceptronSimples(learning_rate=0.01, max_epochs=10, n_classes=n_classes),
        "Adaline": Adaline(learning_rate=0.01, max_epochs=1000, epsilon=1e-3, n_classes=n_classes),
    }

    for nome_modelo, modelo in modelos.items():
        print(f"\nExecutando {nome_modelo}...")

        # Realizando validação de Monte Carlo
        accuracies, all_conf_matrices = monte_carlo_validation(modelo, data, labels, n_runs=50)

        # Calculando as métricas para acurácia
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        max_accuracy = np.max(accuracies)
        min_accuracy = np.min(accuracies)

        print(f"Acurácia média do {nome_modelo}: {mean_accuracy:.4f}")
        print(f"Desvio padrão da acurácia do {nome_modelo}: {std_accuracy:.4f}")
        print(f"Maior acurácia do {nome_modelo}: {max_accuracy:.4f}")
        print(f"Menor acurácia do {nome_modelo}: {min_accuracy:.4f}")

        # Identificar as rodadas com maior e menor acurácia
        max_accuracy_run = np.argmax(accuracies)
        min_accuracy_run = np.argmin(accuracies)
        
        # Exibir a matriz de confusão para a rodada com maior acurácia
        print(f"\nMatriz de Confusão para a rodada com maior acurácia (Rodada {max_accuracy_run + 1}):")
        plot_confusion_matrix(all_conf_matrices[max_accuracy_run], f"{nome_modelo} - Maior Acurácia")
        
        # Exibir a matriz de confusão para a rodada com menor acurácia
        print(f"\nMatriz de Confusão para a rodada com menor acurácia (Rodada {min_accuracy_run + 1}):")
        plot_confusion_matrix(all_conf_matrices[min_accuracy_run], f"{nome_modelo} - Menor Acurácia")

        # Plotando a curva de aprendizado
        plot_learning_curve(modelo.acuracias, nome_modelo)
