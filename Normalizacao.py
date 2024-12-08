import numpy as np

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
