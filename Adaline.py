import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import MatrizConfusao as matriz_confusao

class Adaline:
    def __init__(self, learning_rate=0.01, epochs=1000, tolerance=1e-4, patience=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tolerance = tolerance
        self.patience = patience
        self.weights = None
        self.mse_history = [] #ista vazia  usada para armazenar o valor do mse durante o processo de treinamento.

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # For numerical stability
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def adaline_train(self, X, y):
        n_samples, n_features = X.shape
        n_classes = y.shape[1]  # Número de classes
        self.weights = np.zeros((n_features + 1, n_classes))  # Matriz de pesos incluindo o bias
        X = np.hstack((np.ones((n_samples, 1)), X))  # Adiciona o termo de bias ao input

        no_improve_count = 0

        for epoch in range(self.epochs):
            # Forward pass: saída linear
            linear_output = np.dot(X, self.weights)
            predictions = self.softmax(linear_output)  # Predições multi-classe

            # Calcula os erros e MSE
            errors = y - predictions
            mse = np.mean(errors ** 2)
            self.mse_history.append(mse)

            # Atualiza os pesos
            weight_update = self.learning_rate * np.dot(X.T, errors) / n_samples
            self.weights += weight_update

            # Critério de parada com base na norma da atualização dos pesos
            if np.linalg.norm(weight_update) < self.tolerance:
                print(f"Convergência alcançada na época {epoch + 1}")
                break

            # Early stopping com base na melhoria do MSE
            if epoch > 0 and mse >= self.mse_history[-2]:
                no_improve_count += 1
                if no_improve_count >= self.patience:
                    print("Early stopping ativado devido à falta de melhoria.")
                    break
            else:
                no_improve_count = 0

        return self.weights, self.mse_history

    def adaline_predict(self, X):
        n_samples = X.shape[0]
        X = np.hstack((np.ones((n_samples, 1)), X))  # Adiciona o termo de bias
        linear_output = np.dot(X, self.weights)
        predictions = self.softmax(linear_output)  # Calcula a probabilidade para cada classe
        return np.argmax(predictions, axis=1)  # Retorna a classe com maior probabilidade


    def calculate_metrics(self, predictions, y_test, n_classes):
        accuracy = np.mean(predictions == y_test)

        # Criação da matriz de confusão
        matriz = matriz_confusao.MatrizConfusao(predictions, y_test, n_classes)
        matriz_conf = matriz.get_matrix()  # Obtendo a matriz de confusão

        # Sensibilidade (Recall) e Especificidade
        sensitivity = []
        specificity = []

        for i in range(n_classes):
            tp = matriz_conf[i, i]
            fn = np.sum(matriz_conf[i, :]) - tp
            fp = np.sum(matriz_conf[:, i]) - tp
            tn = np.sum(matriz_conf) - (tp + fn + fp)

            sensitivity.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

        # Métricas médias por classe
        avg_sensitivity = np.mean(sensitivity)
        avg_specificity = np.mean(specificity)

        return accuracy, avg_sensitivity, avg_specificity, matriz_conf


    def update_results(self, metrics, results, predictions, y_test, n_classes=20):
        accuracy, sensitivity, specificity, matriz_conf = self.calculate_metrics(predictions, np.argmax(y_test, axis=1), n_classes)

        metrics["accuracy"].append(accuracy)
        metrics["sensitivity"].append(sensitivity)  # Sensibilidade média
        metrics["specificity"].append(specificity)  # Especificidade média

        results.append({
            "accuracy": accuracy,
            "conf_matrix": matriz_conf,
            "mse": self.mse_history
        })

    def build_summary(self, metrics):
        """Constrói um DataFrame com o resumo das métricas."""
        return pd.DataFrame({
            "Metric": ["Accuracy", "Sensitivity", "Specificity"],
            "Mean": [
                np.mean(metrics["accuracy"]),
                np.mean(metrics["sensitivity"]),
                np.mean(metrics["specificity"]),
            ],
            "StdDev": [
                np.std(metrics["accuracy"]),
                np.std(metrics["sensitivity"]),
                np.std(metrics["specificity"]),
            ],
            "Max": [
                np.max(metrics["accuracy"]),
                np.max(metrics["sensitivity"]),
                np.max(metrics["specificity"]),
            ],
            "Min": [
                np.min(metrics["accuracy"]),
                np.min(metrics["sensitivity"]),
                np.min(metrics["specificity"]),
            ],
        })

    def plot_conf_matrices(self, matriz_conf1, matriz_conf2, model_name):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        labels = ["an2i", "at33", "bol", "bpm", "ch4f", "cheyer", "choon", "danieln", "glickman", "karyadi", "kawamura", "kk49", "megak", "mitchell", "night", "phoebe", "saavik", "steffi", "sz24", "tammo"]

        # Melhor modelo
        sns.heatmap(matriz_conf1, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=labels, yticklabels=labels, ax=axes[0])
        axes[0].set_title(f"Melhor {model_name}")
        axes[0].set_xlabel("Previsão")
        axes[0].set_ylabel("Valor Real")

        # Pior modelo
        sns.heatmap(matriz_conf2, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=labels, yticklabels=labels, ax=axes[1])
        axes[1].set_title(f"Pior {model_name}")
        axes[1].set_xlabel("Previsão")
        axes[1].set_ylabel("Valor Real")

        plt.tight_layout()
        plt.show()

    def plot_learning_curves(self, best_case_mse, worst_case_mse, title):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # Melhor caso
        axs[0].plot(range(1, len(best_case_mse) + 1), best_case_mse, marker='o', linestyle='-', color='g')
        axs[0].set_title(f"{title} - Melhor Caso")
        axs[0].set_xlabel("Épocas")
        axs[0].set_ylabel("Erro Quadrático Médio (MSE)")
        axs[0].grid(True)

        # Pior caso
        axs[1].plot(range(1, len(worst_case_mse) + 1), worst_case_mse, marker='o', linestyle='-', color='r')
        axs[1].set_title(f"{title} - Pior Caso")
        axs[1].set_xlabel("Épocas")
        axs[1].grid(True)

        # Ajustar layout
        plt.tight_layout()
        plt.show()
