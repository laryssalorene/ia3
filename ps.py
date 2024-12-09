import numpy as np
import os
import cv2

# Funções de ativação
activation_func = lambda x: np.maximum(0, x)
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

    def predict(self, X):
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])  # Adiciona o bias
        u = np.dot(X_bias, self.weights)  # Calcula as ativações
        return step_activation(u)  # Aplica a função de ativação degrau

    def calculate_accuracy(self, y_true, y_pred):
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)
        return np.mean(y_true_classes == y_pred_classes)

    def fit(self, X, y):
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])  # Adiciona o bias
        self.weights = np.random.randn(X_bias.shape[1], self.n_classes) * 0.01  # Inicializa os pesos
        for epoch in range(self.max_epochs):
            if epoch % 100 == 0:
                print(f"PS: Época {epoch}: treinamento em andamento...")
            for i in range(X.shape[0]):
                xi = X_bias[i]
                yi = y[i]
                u = np.dot(xi, self.weights)
                y_pred = step_activation(u)
                error = yi - y_pred
                self.weights += self.learning_rate * np.outer(xi, error)

# Classe MLP
class MLP:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01, max_epochs=1000, tolerance=1e-4, patience=10):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.patience = patience

        # Inicialização das camadas
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = [
            np.random.randn(self.layers[i + 1], self.layers[i] + 1) * np.sqrt(2 / self.layers[i])
            for i in range(len(self.layers) - 1)
        ]

    def forward(self, X):
        activations = [X.T]
        zs = []

        for w in self.weights[:-1]:
            a_with_bias = np.vstack([np.ones((1, activations[-1].shape[1])), activations[-1]])
            z = np.dot(w, a_with_bias)
            zs.append(z)
            activations.append(activation_func(z))

        a_with_bias = np.vstack([np.ones((1, activations[-1].shape[1])), activations[-1]])
        z = np.dot(self.weights[-1], a_with_bias)
        zs.append(z)
        activations.append(softmax(z))

        return activations, zs

    def backward(self, activations, zs, y_true):
        deltas = [None] * len(self.weights)
        deltas[-1] = activations[-1] - y_true.T

        for l in range(len(self.weights) - 2, -1, -1):
            deltas[l] = np.dot(self.weights[l + 1][:, 1:].T, deltas[l + 1]) * activation_derivative(activations[l + 1])

        return deltas

    def update_weights(self, activations, deltas):
        for l in range(len(self.weights)):
            a_with_bias = np.vstack([np.ones((1, activations[l].shape[1])), activations[l]])
            self.weights[l] -= self.learning_rate * np.dot(deltas[l], a_with_bias.T) / activations[l].shape[1]

    def train(self, X, y):
        mse_history = []
        no_improve_count = 0

        for epoch in range(self.max_epochs):
            activations, zs = self.forward(X)
            y_pred = activations[-1]
            mse = np.mean(np.sum((y_pred - y.T) ** 2, axis=0))
            mse_history.append(mse)

            if epoch > 0 and mse >= mse_history[-2]:
                no_improve_count += 1
                if no_improve_count >= self.patience:
                    print(f"Early stopping ativado na época {epoch + 1}.")
                    break
            else:
                no_improve_count = 0

            if epoch > 0 and abs(mse_history[-1] - mse_history[-2]) < self.tolerance:
                print(f"Convergência alcançada na época {epoch + 1}.")
                break

            deltas = self.backward(activations, zs, y)
            self.update_weights(activations, deltas)

    def predict(self, X):
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=0)

# Classe LerImagem
class LerImagem:
    @staticmethod
    def obter_dados(img_size=50):
        raiz = "RecFac"
        pastas_pessoas = [os.path.join(raiz, pasta) for pasta in os.listdir(raiz) if os.path.isdir(os.path.join(raiz, pasta))]
        C = 20
        X = []
        Y = []

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
        print(f"Tamanho final de X: {X.shape}, Tamanho final de Y: {Y.shape}")
        return X, Y

class Normalizacao:
    @staticmethod
    def normalizar(data):
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        range_val = max_val - min_val
        normalized_data = (data - min_val) / range_val
        normalized_data = normalized_data * 2 - 1
        return normalized_data

# Funções de métrica
def calculate_sensitivity(y_true, y_pred):
    y_true = y_true.astype(float)  # Garante que y_true é float
    y_pred = y_pred.astype(float)  # Garante que y_pred é float
    true_positives = np.sum((y_true == 1) & (y_pred == 1), axis=0)
    relevant_elements = np.sum(y_true == 1, axis=0)
    sensitivity = np.divide(
        true_positives, relevant_elements, out=np.zeros_like(true_positives, dtype=float), where=relevant_elements > 0
    )
    return np.mean(sensitivity)


def calculate_specificity(y_true, y_pred):
    y_true = y_true.astype(float)  # Garante que y_true é float
    y_pred = y_pred.astype(float)  # Garante que y_pred é float
    true_negatives = np.sum((y_true == 0) & (y_pred == 0), axis=0)
    non_relevant_elements = np.sum(y_true == 0, axis=0)
    specificity = np.divide(
        true_negatives, non_relevant_elements, out=np.zeros_like(true_negatives, dtype=float), where=non_relevant_elements > 0
    )
    return np.mean(specificity)


# Execução
if __name__ == "__main__":
    R = 10
    epochs = 10
    learning_rate = 0.01
    tolerance = 1e-4
    img_size = 50

    # Carregar e normalizar os dados
    data, labels = LerImagem.obter_dados(img_size)
    data = Normalizacao.normalizar(data)

    n_samples = data.shape[0]

    modelos = {
        "Perceptron Simples": PerceptronSimples(learning_rate=learning_rate, max_epochs=epochs, tolerance=tolerance, n_classes=labels.shape[1]),
        "MLP": MLP(input_size=data.shape[1], hidden_layers=[128], output_size=labels.shape[1],
                   learning_rate=learning_rate, max_epochs=epochs, tolerance=tolerance, patience=10)
    }

    for nome_modelo, modelo in modelos.items():
        metrics = {"accuracy": [], "sensitivity": [], "specificity": []}
        print(f"\nExecutando {nome_modelo}...")

        for _ in range(R):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            data, labels = data[indices], labels[indices]
            train_size = int(0.8 * n_samples)
            X_train, X_test = data[:train_size], data[train_size:]
            y_train, y_test = labels[:train_size], labels[train_size:]

            # Ajuste do modelo
            if isinstance(modelo, PerceptronSimples):
                modelo.fit(X_train, y_train)  # Usar fit para o PerceptronSimples
            else:
                modelo.train(X_train, y_train)  # Usar train para os demais modelos

            predictions = modelo.predict(X_test)

            # Garantir que predictions seja bidimensional
            if predictions.ndim == 1:
                predictions = np.eye(y_test.shape[1])[predictions]

            # Calcular métricas
            accuracy = np.mean(np.argmax(y_test, axis=1) == np.argmax(predictions, axis=1))
            sensitivity = calculate_sensitivity(y_test, predictions)
            specificity = calculate_specificity(y_test, predictions)

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
