import numpy as np

class MatrizConfusao:
    def __init__(self, predictions, y_test, n_classes):
        # Inicializa a matriz de confusão com zeros
        self.conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
        
        # Usando numpy para contagem das ocorrências diretamente
        for i in range(len(predictions)):
            true_label = y_test[i]
            predicted_label = predictions[i]
            
            # Atualiza a posição da matriz correspondente à combinação de rótulo verdadeiro e previsto
            self.conf_matrix[true_label, predicted_label] += 1

    def get_matrix(self):
        return self.conf_matrix
