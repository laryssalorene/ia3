import numpy as np
from LerImagem import LerImagem
import Normalizacao as normalizar
import Adaline

# Hiperparâmetros
plot = True
R = 50
epochs = 200
learning_rate = 0.01
tolerance = 1e-4
patience = 10
img_size = 50

# Carregar e normalizar os dados
data, labels = LerImagem.obter_dados(img_size)  # Correção da chamada
data = normalizar.Normalizacao.normalizar(data)

# Verifique o tamanho de 'data' e 'labels' para garantir que está correto
print("Tamanho de data:", data.shape)
print("Tamanho de labels:", labels.shape)

# Validação por Monte Carlo.
n_samples = data.shape[0]  # Certifique-se de que isso é 640
metricsAdaline = {"accuracy": [], "sensitivity": [], "specificity": []}

# Criar uma instância de Adaline
adaline = Adaline.Adaline(learning_rate=learning_rate, epochs=epochs, tolerance=tolerance, patience=patience)

# Realizar as R iterações para Monte Carlo
for _ in range(R):
    # Embaralhar os índices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    data, labels = data[indices], labels[indices]

    # Dividir em conjuntos de treinamento e teste (80-20)
    train_size = int(0.8 * n_samples)
    X_train, X_test = data[:train_size], data[train_size:]
    y_train, y_test = labels[:train_size], labels[train_size:]

    # Treinamento e predição
    weights, mse_history = adaline.adaline_train(X_train, y_train)
    predictions = adaline.adaline_predict(X_test)
    
    if (_ + 1) % (R // 10) == 0:
        print(f"Rodada: {_ + 1}")
    
    # Calcular métricas
    accuracy, sensitivity, specificity, conf_matrix = adaline.calculate_metrics(predictions, np.argmax(y_test, axis=1), 20)
    metricsAdaline["accuracy"].append(accuracy)
    metricsAdaline["sensitivity"].append(sensitivity)
    metricsAdaline["specificity"].append(specificity)

# Imprimir as métricas finais de Monte Carlo após as R rodadas
print(f"Média de Acurácia: {np.mean(metricsAdaline['accuracy'])}")
print(f"Média de Sensibilidade: {np.mean(metricsAdaline['sensitivity'])}")
print(f"Média de Especificidade: {np.mean(metricsAdaline['specificity'])}")
