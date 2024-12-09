import numpy as np
import os
import cv2
import Adaline

# Função de ativação: Retorna a classe com maior valor em formato one-hot
def step_activation(u):
    one_hot = np.zeros_like(u)  # Cria um vetor de zeros com o mesmo formato de u
    one_hot[np.argmax(u)] = 1   # Marca a classe com maior valor
    return one_hot

class PerceptronSimples:
    def __init__(self, learning_rate=0.01, max_epochs=10, tolerance=1e-4, n_classes=20):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.n_classes = n_classes  # Número de classes
        self.weights = None

    def predict(self, X, return_indices=False):
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        u = np.dot(X_bias, self.weights)
        predictions = np.zeros_like(u)
        predictions[np.arange(u.shape[0]), np.argmax(u, axis=1)] = 1
        if return_indices:
            return np.argmax(predictions, axis=1)
        return predictions


    def calculate_accuracy(self, y_true, y_pred):
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)
        return np.mean(y_true_classes == y_pred_classes)

    def fit(self, X, y):
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        self.weights = np.random.randn(X_bias.shape[1], self.n_classes) * 0.01
        for epoch in range(self.max_epochs):
            for i in range(X.shape[0]):
                xi = X_bias[i]
                yi = y[i]
                u = np.dot(xi, self.weights)
                y_pred = np.zeros_like(u)
                y_pred[np.argmax(u)] = 1
                error = yi - y_pred
                self.weights += self.learning_rate * np.outer(xi, error)


    def calculate_mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def calculate_sensitivity(self, y_test, predictions):
        # Certifique-se de que os arrays estão no formato float
        y_test = y_test.astype(float)
        predictions = predictions.astype(float)
        
        true_positives = np.sum((y_test == 1) & (predictions == 1), axis=0)
        relevant_elements = np.sum(y_test == 1, axis=0)
        sensitivity = np.divide(
            true_positives, relevant_elements, out=np.zeros_like(true_positives, dtype=float), where=relevant_elements > 0
        )
        return np.mean(sensitivity)




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


# Hiperparâmetros
R = 10
epochs = 10                                     
learning_rate = 0.01
tolerance = 1e-4

img_size = 50

# Carregar e normalizar os dados
data, labels = LerImagem.obter_dados(img_size)
data = Normalizacao.normalizar(data)

# Verifique o tamanho de 'data' e 'labels'
print("Tamanho de data:", data.shape)
print("Tamanho de labels:", labels.shape)

# Validação por Monte Carlo.
n_samples = data.shape[0]
metrics = {"accuracy": [], "sensitivity": [], "specificity": []}

# Escolher o modelo a ser utilizado: 'adaline' ou 'perceptron'
modelo = 'perceptron'  # ou 'adaline'

# Criar a instância do modelo escolhido
if modelo == 'adaline':
    model = Adaline.Adaline(learning_rate=learning_rate, max_epochs=epochs, epsilon=tolerance)
elif modelo == 'perceptron':
    model = PerceptronSimples(learning_rate=learning_rate, max_epochs=epochs, tolerance=tolerance, n_classes=labels.shape[1])
else:
    raise ValueError("Modelo inválido. Use 'adaline' ou 'perceptron'.")


# Realizar as R iterações para Monte Carlo
for _ in range(R):
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    data, labels = data[indices], labels[indices]

    train_size = int(0.8 * n_samples)
    X_train, X_test = data[:train_size], data[train_size:]
    y_train, y_test = labels[:train_size], labels[train_size:]

    model.fit(X_train, y_train)  # Treina o modelo
    predictions = model.predict(X_test)  # Faz a predição

    # Calcular métricas
    accuracy = model.calculate_accuracy(y_test, predictions)
    sensitivity = model.calculate_sensitivity(y_test, predictions)

    # Armazenar métricas
    metrics["accuracy"].append(accuracy)
    metrics["sensitivity"].append(sensitivity)

# Estatísticas para a métrica de Acurácia
accuracy_mean = np.mean(metrics["accuracy"])
accuracy_std = np.std(metrics["accuracy"])
accuracy_max = np.max(metrics["accuracy"])
accuracy_min = np.min(metrics["accuracy"])

# Exibir os resultados no terminal
print("\nEstatísticas para a métrica de Acurácia:")
print(f"Média: {accuracy_mean:.4f}")
print(f"Desvio-Padrão: {accuracy_std:.4f}")
print(f"Maior Valor: {accuracy_max:.4f}")
print(f"Menor Valor: {accuracy_min:.4f}")
