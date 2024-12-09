import numpy as np
import os
import cv2

# Funções auxiliares
activation_func = lambda x: np.maximum(0, x)  # ReLU
activation_derivative = lambda a: np.where(a > 0, 1, 0)

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

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def step_activation(u):
    return np.where(u >= 0, 1, 0)

# Classe Perceptron Simples
class PerceptronSimples:
    def __init__(self, learning_rate=0.01, max_epochs=10, n_classes=20):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
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
    def __init__(self, learning_rate=0.001, max_epochs=1000, epsilon=1e-3):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.epsilon = epsilon
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        N, M = X.shape
        self.num_classes = y.shape[1]
        self.weights = np.random.randn(M, self.num_classes) * 0.01  # Reduzi a escala dos pesos
        self.bias = np.zeros(self.num_classes)

        for epoch in range(self.max_epochs):
            u = np.dot(X, self.weights) + self.bias
            u = np.clip(u, -1e3, 1e3)  # Evita overflow
            y_hat = u  # Ativação linear
            error = y - y_hat
            
            # Calcular o EQM
            eqm = np.mean(error**2)
            
            # Atualizar pesos e bias
            self.weights += self.learning_rate * np.dot(X.T, error) / N
            self.bias += self.learning_rate * np.mean(error, axis=0)
            
            
            # Critério de parada
            if np.linalg.norm(error) < self.epsilon:
                print(f"Convergência alcançada na época {epoch}")
                break

    def predict(self, X):
        u = np.dot(X, self.weights) + self.bias
        u = np.clip(u, -1e3, 1e3)  # Evita overflow
        return np.where(u >= 0.5, 1, 0)


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

# Métricas
def calculate_metrics(y_test, predictions):
    accuracy = np.mean(np.argmax(y_test, axis=1) == np.argmax(predictions, axis=1))
    sensitivity = np.mean(np.diag(np.dot(y_test.T, predictions)) / np.sum(y_test, axis=0))
    specificity = np.mean(np.diag(np.dot(1 - y_test.T, 1 - predictions)) / np.sum(1 - y_test, axis=0))
    return accuracy, sensitivity, specificity

# Execução
if __name__ == "__main__":
    img_size = 50
    data, labels = LerImagem.obter_dados(img_size)
    data = Normalizacao.normalizar(data)

    n_samples, n_features = data.shape
    n_classes = labels.shape[1]

    modelos = {
        "Perceptron Simples": PerceptronSimples(learning_rate=0.01, max_epochs=10, n_classes=n_classes),
        "Adaline": Adaline(learning_rate=0.01, max_epochs=1000),
        "MLP": MLP(input_size=n_features, hidden_layers=[128], output_size=n_classes, learning_rate=0.01, max_epochs=10)
    }

    for nome_modelo, modelo in modelos.items():
        metrics = {"accuracy": [], "sensitivity": [], "specificity": []}
        print(f"\nExecutando {nome_modelo}...")

        for _ in range(10):
            indices = np.random.permutation(n_samples)
            X_train, X_test = data[indices[:512]], data[indices[512:]]
            y_train, y_test = labels[indices[:512]], labels[indices[512:]]

            modelo.fit(X_train, y_train)
            predictions = modelo.predict(X_test)

            if predictions.ndim == 1:
                predictions = np.eye(n_classes)[predictions]

            accuracy, sensitivity, specificity = calculate_metrics(y_test, predictions)
            metrics["accuracy"].append(accuracy)
            metrics["sensitivity"].append(sensitivity)
            metrics["specificity"].append(specificity)

        for metric_name, values in metrics.items():
            print(f"\nEstatísticas para {metric_name} - {nome_modelo}:")
            print(f"Média: {np.mean(values):.4f}")
            print(f"Desvio-Padrão: {np.std(values):.4f}")
            print(f"Maior Valor: {np.max(values):.4f}")
            print(f"Menor Valor: {np.min(values):.4f}")
