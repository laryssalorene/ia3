import numpy as np

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

    def fit(self, X, y):
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])  # Adiciona o bias
        self.weights = np.random.randn(X_bias.shape[1], self.n_classes) * 0.01  # Inicializa os pesos
        for epoch in range(self.max_epochs):
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

    def fit(self, X, y):
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
                    print(f"MLP: Early stopping ativado na época {epoch + 1}.")
                    break
            else:
                no_improve_count = 0

            if epoch > 0 and abs(mse_history[-1] - mse_history[-2]) < self.tolerance:
                print(f"MLP: Convergência alcançada na época {epoch + 1}.")
                break

            deltas = self.backward(activations, zs, y)
            self.update_weights(activations, deltas)

    def predict(self, X):
        activations, _ = self.forward(X)
        return activations[-1].T

# Classe Adaline
class Adaline:
    def __init__(self, learning_rate=0.01, max_epochs=1000, epsilon=1e-3):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.epsilon = epsilon
        self.w = None
        self.bias = None

    def step_bipolar(self, x):
        return np.where(x >= 0, 1, -1)

    def fit(self, X_train, y_train):
        N, M = X_train.shape
        self.num_classes = y_train.shape[1]
        self.w = np.random.randn(M, self.num_classes) * np.sqrt(2 / M)
        self.bias = np.zeros(self.num_classes)

        for epoch in range(self.max_epochs):
            u = np.dot(X_train, self.w) + self.bias
            y_hat = self.step_bipolar(u)
            erro = y_train - y_hat

            for i in range(self.num_classes):
                self.w[:, i] += self.learning_rate * np.dot(X_train.T, erro[:, i]) / N
                self.bias[i] += self.learning_rate * np.sum(erro[:, i]) / N

            EQM = np.mean(erro**2) / 2
            if epoch % 100 == 0:
                print(f"Adaline: Época {epoch}, EQM: {EQM}")

    def predict(self, X_test):
        u = np.dot(X_test, self.w) + self.bias
        return self.step_bipolar(u)

# Funções de métricas
def calculate_sensitivity(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1), axis=0).astype(float)
    relevant_elements = np.sum(y_true == 1, axis=0).astype(float)
    sensitivity = np.divide(true_positives, relevant_elements, out=np.zeros_like(true_positives, dtype=float), where=relevant_elements > 0)
    return np.mean(sensitivity)

def calculate_specificity(y_true, y_pred):
    true_negatives = np.sum((y_true == 0) & (y_pred == 0), axis=0).astype(float)
    non_relevant_elements = np.sum(y_true == 0, axis=0).astype(float)
    specificity = np.divide(true_negatives, non_relevant_elements, out=np.zeros_like(true_negatives, dtype=float), where=non_relevant_elements > 0)
    return np.mean(specificity)


# Execução
if __name__ == "__main__":
    R = 10
    epochs = 10
    learning_rate = 0.01
    tolerance = 1e-4
    img_size = 50

    data = np.random.randn(640, 2500)
    labels = np.eye(20)[np.random.randint(0, 20, 640)]

    n_samples = data.shape[0]

    modelos = {
        "Perceptron Simples": PerceptronSimples(learning_rate=learning_rate, max_epochs=epochs, tolerance=tolerance, n_classes=labels.shape[1]),
        "MLP": MLP(input_size=data.shape[1], hidden_layers=[128], output_size=labels.shape[1],
                   learning_rate=learning_rate, max_epochs=epochs, tolerance=tolerance, patience=10),
        "Adaline": Adaline(learning_rate=learning_rate, max_epochs=epochs, epsilon=tolerance)
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

            modelo.fit(X_train, y_train)
            predictions = modelo.predict(X_test)

            if nome_modelo == "MLP":
                predictions = np.argmax(predictions, axis=1)
                predictions = np.eye(labels.shape[1])[predictions]


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
